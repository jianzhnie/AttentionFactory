"""FA1: baseline tiled online-softmax attention.

Simplified algorithm in this module:
1. Split the query rows and key/value rows into tiles.
2. Keep one output tile together with per-row running statistics:
   the running row max and the running normalization sum.
3. Stream over K/V tiles outside the Q loop, compute the local score block,
   update the online softmax state, and immediately fold the weighted value
   contribution into the normalized output tile.
4. In backward, recompute the local probabilities from the saved LSE-style
   statistics instead of storing the full attention matrix.

This is the closest educational version to the original FlashAttention paper.
Its main improvement over naive attention is IO-awareness: it never
materializes the full score or probability matrix in memory, and instead keeps
only compact row-wise statistics plus the running output tile.
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
    merge_state_normalized,
    prepare_inputs,
)

__all__ = ["backward", "flash_attention_v1", "forward"]


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> ForwardResult:
    """Run the FA1 tiled forward pass (K/V-outer loop, per-step normalized merge).

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
    # Each query tile owns a running output tile plus the online-softmax state
    # for that tile. This is the key FA1 idea: keep row-local statistics instead
    # of materializing the full attention matrix.
    out_blocks, normalizer_blocks, row_max_blocks = init_block_state(q, v, q_slices)
    debug_state = {"loop_order": "kv_outer_q_inner"} if config.keep_debug_state else {}

    for k_slice in k_slices:
        v_block = v[:, :, k_slice, :]

        for q_index, q_slice in enumerate(q_slices):
            # In the CUDA kernel this block would be computed from SRAM-resident
            # tiles after cooperative loads. Here we just rebuild the score tile
            # directly in PyTorch to keep the math readable.
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # `compute_local_statistics` performs the local online-softmax work:
            # local row max, local row sum, and the unnormalized P @ V term.
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores, valid_mask, v_block
            )
            # `merge_state_normalized` is the exact FA1 merge step that combines
            # the old online-softmax state with the new tile contribution.
            out_blocks[q_index], normalizer_blocks[q_index], row_max_blocks[q_index] = (
                merge_state_normalized(
                    out_blocks[q_index],
                    normalizer_blocks[q_index],
                    row_max_blocks[q_index],
                    block_max,
                    block_sum,
                    weighted_values,
                )
            )

    # FA1 keeps the output tile normalized at every merge step, so the
    # concatenated tiles already form the final output (normalized=True).
    return assemble_forward_result(
        out_blocks,
        normalizer_blocks,
        row_max_blocks,
        normalized=True,
        out_dtype=q.dtype,
        saved_state=debug_state,
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
    """Run the FA1 tiled backward pass (K/V-outer loop).

    Per-tile probabilities are recomputed from the log-sum-exp stored in
    ``forward_result`` instead of a stored attention matrix.

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

    for k_slice in k_slices:
        k_block = k[:, :, k_slice, :]
        v_block = v[:, :, k_slice, :]
        # FA1 accumulates dK and dV per K/V tile while revisiting every query tile.
        # The real backward kernel similarly recomputes local probabilities instead
        # of reading a stored attention matrix back from global memory.
        dK_block = torch.zeros_like(k_block, dtype=torch.float32)
        dV_block = torch.zeros_like(v_block, dtype=torch.float32)

        for q_slice in q_slices:
            q_block = q[:, :, q_slice, :]
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # The saved LSE lets us reconstruct probabilities on demand. That is
            # the backward-side analogue of FA1's forward IO reduction.
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
            dQ[:, :, q_slice, :] += local_dQ
            dK_block += local_dK
            dV_block += local_dV

        dK[:, :, k_slice, :] = dK_block
        dV[:, :, k_slice, :] = dV_block

    return BackwardResult(dQ=dQ.to(q.dtype), dK=dK.to(k.dtype), dV=dV.to(v.dtype))


def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    """Differentiable FA1 attention, callable like any PyTorch operation.

    Unlike calling `forward` directly, the returned tensor records the
    autograd graph: ``loss.backward()`` routes through this module's tiled
    `backward` implementation. Arguments match `forward`.
    """
    return TiledAttentionFunction.apply(
        q, k, v, causal, key_padding_mask, config, forward, backward
    )
