"""FA2: sequence-parallel ownership with deferred output normalization.

Simplified algorithm in this module:
1. Assign ownership of one query tile to each outer-loop iteration.
2. For that owned query tile, stream over all K/V tiles and accumulate an
   unnormalized output tile together with the running row max and row sum.
3. Only after all K/V tiles have been processed for that query tile, apply the
   final normalization step.
4. In backward, keep the same ownership-oriented orchestration so the control
   flow reflects the split-Q / sequence-parallel idea instead of the FA1
   KV-outer loop structure.

Compared with FA1, the educational improvement is the work partitioning:
the code is organized around query-tile ownership, deferred normalization, and
LSE-centered saved state. Real FA2 uses this style to expose more sequence
parallelism and reduce non-matmul overhead; here we mirror that algorithmic
shape without reproducing the CUDA launch details.
"""

from __future__ import annotations

import torch

from .common import (
    BackwardResult,
    FlashAttentionConfig,
    ForwardResult,
    TiledAttentionFunction,
    assemble_forward_result,
    backward_step,
    block_scores_and_mask,
    compute_local_statistics,
    init_block_state,
    init_gradients,
    iter_block_slices,
    merge_state_unnormalized,
    prepare_inputs,
)

__all__ = ["backward", "flash_attention_v2", "forward"]


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> ForwardResult:
    """Run the FA2 tiled forward pass (query-tile ownership, deferred normalization).

    Args:
        q: Queries, shape ``(batch, heads, q_len, head_dim)``.
        k: Keys, shape ``(batch, heads, kv_len, head_dim)``.
        v: Values, shape ``(batch, heads, kv_len, value_dim)``; ``value_dim``
            may differ from ``head_dim``.
        causal: Apply a causal mask. When ``kv_len > q_len`` the diagonal is
            aligned with the end of the key sequence.
        key_padding_mask: Optional mask of shape ``(batch, kv_len)``; ``True``
            marks valid key positions.
        config: Tiling and debug knobs; ``FlashAttentionConfig()`` defaults
            are used when omitted.

    Returns:
        A `ForwardResult` with the attention output, the per-row log-sum-exp,
        and the final online-softmax statistics.
    """
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
    )

    q_slices = iter_block_slices(q.shape[2], config.block_size_q)
    k_slices = iter_block_slices(k.shape[2], config.block_size_kv)
    # FA2 changes the ownership model: the outer loop now "owns" one query tile
    # and streams every K/V tile through it. This mirrors split-Q style work
    # partitioning, which improves occupancy when sequence length is large.
    out_acc_blocks, normalizer_blocks, row_max_blocks = init_block_state(q, v, q_slices)
    query_owners: list[dict[str, int]] = []

    for owner_id, q_slice in enumerate(q_slices):
        query_owners.append(
            {
                "owner_id": owner_id,
                "q_start": q_slice.start or 0,
                "q_end": q_slice.stop or q.shape[2],
                "num_kv_tiles": len(k_slices),
            }
        )

        for k_slice in k_slices:
            # Real FA2 would launch multiple CTAs over query tiles so these owners
            # run concurrently. Here we keep the same control-flow structure but
            # execute it sequentially for clarity.
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                v[:, :, k_slice, :],
            )
            # Unlike FA1, we intentionally keep the output tile unnormalized while
            # streaming K/V tiles. The final normalization is deferred until the
            # owner has seen the full K/V sequence.
            (
                out_acc_blocks[owner_id],
                normalizer_blocks[owner_id],
                row_max_blocks[owner_id],
            ) = merge_state_unnormalized(
                out_acc_blocks[owner_id],
                normalizer_blocks[owner_id],
                row_max_blocks[owner_id],
                block_max,
                block_sum,
                weighted_values,
            )

    # Real FA2 stores log-sum-exp style state and performs sequence-parallel work
    # in different CUDA threadblocks. This simplified version keeps those ideas
    # visible without reproducing kernel-level launches.
    debug_state = {
        "query_owners": query_owners,
        "deferred_normalization": True,
    }
    # This late normalization is one of the main algorithmic cleanups in FA2:
    # fewer rescale operations are performed inside the tiled main loop. It is
    # applied inside `assemble_forward_result` (normalized=False).
    return assemble_forward_result(
        out_acc_blocks,
        normalizer_blocks,
        row_max_blocks,
        normalized=False,
        out_dtype=q.dtype,
        saved_state=debug_state if config.keep_debug_state else {},
    )


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    forward_result: ForwardResult,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> BackwardResult:
    """Run the FA2 tiled backward pass (query-tile ownership).

    Uses the same exact attention derivatives as FA1; the difference is the
    orchestration: gradients stay local to the owning query tile until the
    finished dQ tile is written back.

    Args:
        q, k, v: The same tensors the forward pass was called with.
        grad_out: Gradient w.r.t. the forward output, same shape as
            ``forward_result.out``.
        forward_result: The result returned by `forward`.
        causal, key_padding_mask, config: Must match the forward call.

    Returns:
        A `BackwardResult` with the gradients dQ, dK and dV.
    """
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
    )

    dQ, dK, dV = init_gradients(q, k, v)
    q_slices = iter_block_slices(q.shape[2], config.block_size_q)
    k_slices = iter_block_slices(k.shape[2], config.block_size_kv)
    owner_trace: list[dict[str, int]] = []

    for owner_id, q_slice in enumerate(q_slices):
        q_block = q[:, :, q_slice, :]
        # Keep gradients local to the owning query tile first, then write the
        # finished dQ tile back once all K/V tiles have been processed.
        dQ_block = torch.zeros_like(q_block, dtype=torch.float32)

        for k_slice in k_slices:
            k_block = k[:, :, k_slice, :]
            v_block = v[:, :, k_slice, :]
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # The derivative formulas are the same exact attention derivatives as
            # FA1. The educational difference is purely in the orchestration.
            local_dQ, local_dK, local_dV = backward_step(
                q_block=q_block,
                k_block=k_block,
                v_block=v_block,
                out_block=forward_result.out[:, :, q_slice, :],
                grad_out_block=grad_out[:, :, q_slice, :],
                scores=scores,
                valid_mask=valid_mask,
                lse_block=forward_result.lse[:, :, q_slice, :],
            )
            dQ_block += local_dQ
            dK[:, :, k_slice, :] += local_dK
            dV[:, :, k_slice, :] += local_dV

        dQ[:, :, q_slice, :] = dQ_block
        owner_trace.append({"owner_id": owner_id, "num_kv_tiles": len(k_slices)})

    return BackwardResult(
        dQ=dQ.to(q.dtype),
        dK=dK.to(k.dtype),
        dV=dV.to(v.dtype),
        debug_state={"query_owner_trace": owner_trace}
        if config.keep_debug_state
        else {},
    )


def flash_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    """Differentiable FA2 attention, callable like any PyTorch operation.

    Unlike calling `forward` directly, the returned tensor records the
    autograd graph: ``loss.backward()`` routes through this module's tiled
    `backward` implementation. Arguments match `forward`.
    """
    return TiledAttentionFunction.apply(
        q, k, v, causal, key_padding_mask, config, forward, backward
    )
