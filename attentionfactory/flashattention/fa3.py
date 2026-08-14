"""FA3: staged producer/consumer pipeline with ping-pong buffers.

Simplified algorithm in this module:
1. Keep the FA2-style exact attention math, but reorganize execution into
   explicit pipeline stages rather than a single monolithic loop body.
2. Model a producer/consumer handoff with two logical tile buffers: one buffer
   is "active" for the current K/V tile while the next tile is logically
   prefetched into the other buffer.
3. For each query tile, run the stages in order: load tile, compute score
   block, update online softmax statistics, and apply the value contribution.
4. Mirror the same staged structure in backward so the simplified code still
   depicts the overlap-friendly orchestration.
5. In `fp8` mode, simulate FA3's block-quantized FP8 forward path with per-tile
   scales and dequantize-on-use computation.

Compared with FA2, the educational improvement is pipeline structure. Real FA3
uses Hopper features such as TMA, WGMMA, warp specialization, and ping-pong
buffering to overlap data movement and compute. This module leaves those
hardware mechanisms out, but keeps the stage boundaries and double-buffered
control flow visible in plain PyTorch. The released official FA3 path supports
FP8 forward, but not FP8 backward, and this simplified implementation follows
that same support boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .common import (
    BackwardResult,
    FlashAttentionConfig,
    ForwardResult,
    assemble_forward_result,
    backward_step,
    build_block_mask,
    compute_local_statistics,
    init_block_state,
    init_gradients,
    iter_block_slices,
    merge_state_unnormalized,
    prepare_inputs,
    scaled_scores,
)

__all__ = ["attention", "backward", "forward"]

#: Largest magnitude representable in the FP8 E4M3 format.
FP8_E4M3_MAX = 448.0


@dataclass
class TileBuffer:
    """One K/V tile in flight in the simulated pipeline.

    Attributes:
        buffer_id: Which logical ping-pong buffer this tile occupies.
        tile_id: Index of the K/V tile within the sequence.
        tile_slice: Sequence-dimension slice the tile was loaded from.
        k_block: The (possibly FP8-simulated) key tile.
        v_block: The (possibly FP8-simulated) value tile.
        fp8_meta: Per-tile quantization scales/amaxes when ``fp8`` is enabled.
    """

    buffer_id: int
    tile_id: int
    tile_slice: slice
    k_block: torch.Tensor
    v_block: torch.Tensor
    fp8_meta: dict[str, float] | None = None


def _simulate_fp8_block(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Quantize-dequantize a tile the way FA3's block-scaled FP8 path would.

    FA3's published FP8 path uses true float8 formats plus block quantization.
    PyTorch portability is better if we simulate that with a per-block scale
    and a dequantize-on-use path rather than requiring hardware float8 kernels.
    """
    amax = tensor.abs().max().item()
    scale = max(amax / FP8_E4M3_MAX, 1e-8)
    quantized = torch.clamp(torch.round(tensor / scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    dequantized = quantized * scale
    return dequantized, {"amax": amax, "scale": scale, "format_max": FP8_E4M3_MAX}


def _compute_scores_from_blocks(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    q_slice: slice,
    k_slice: slice,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score tile from already-loaded blocks (which may be FP8-simulated)."""
    scores = scaled_scores(q_block, k_block)
    valid_mask = build_block_mask(
        batch_size=q_block.shape[0],
        q_slice=q_slice,
        k_slice=k_slice,
        q_len=q_len,
        kv_len=kv_len,
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q_block.device,
    )
    return scores, valid_mask


def _load_tile(
    *,
    buffer_id: int,
    tile_id: int,
    tile_slice: slice,
    k: torch.Tensor,
    v: torch.Tensor,
    fp8: bool,
) -> TileBuffer:
    """Load one K/V tile into a logical pipeline buffer.

    In real FA3 this "load" would be backed by TMA into shared memory and then
    consumed by WGMMA-based compute warpgroups. Here it is just a lightweight
    Python object so the pipeline structure is explicit in the code.
    """
    k_block = k[:, :, tile_slice, :]
    v_block = v[:, :, tile_slice, :]
    fp8_meta = None
    if fp8:
        # A simplified depiction of FA3's block-quantized FP8 path. The official
        # implementation uses hardware FP8 fragments; we model the same idea
        # with per-tile quantize/dequantize metadata.
        k_block, k_meta = _simulate_fp8_block(k_block)
        v_block, v_meta = _simulate_fp8_block(v_block)
        fp8_meta = {
            "k_scale": k_meta["scale"],
            "k_amax": k_meta["amax"],
            "v_scale": v_meta["scale"],
            "v_amax": v_meta["amax"],
        }
    return TileBuffer(
        buffer_id=buffer_id,
        tile_id=tile_id,
        tile_slice=tile_slice,
        k_block=k_block,
        v_block=v_block,
        fp8_meta=fp8_meta,
    )


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> ForwardResult:
    """Run the FA3 staged-pipeline forward pass (ping-pong tile buffers).

    With ``config.fp8=True`` this simulates FA3's block-quantized FP8 forward
    path: each tile carries its own scale and is dequantized on use.

    Args:
        q: Queries, shape ``(batch, heads, q_len, head_dim)``.
        k: Keys, shape ``(batch, heads, kv_len, head_dim)``.
        v: Values, shape ``(batch, heads, kv_len, value_dim)``; ``value_dim``
            may differ from ``head_dim``.
        causal: Apply a causal mask. When ``kv_len > q_len`` the diagonal is
            aligned with the end of the key sequence.
        key_padding_mask: Optional mask of shape ``(batch, kv_len)``; ``True``
            marks valid key positions.
        config: Tiling, pipeline and FP8 knobs; ``FlashAttentionConfig()``
            defaults are used when omitted.

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
    # The buffers stand in for ping-pong shared-memory stages. FA3's practical
    # gain comes from overlapping movement and compute; we preserve that mental
    # model without introducing actual asynchronous execution here.
    out_acc_blocks, normalizer_blocks, row_max_blocks = init_block_state(q, v, q_slices)
    pipeline_trace: list[dict[str, Any]] = []

    for q_tile_id, q_slice in enumerate(q_slices):
        if not k_slices:
            continue

        q_block = q[:, :, q_slice, :]
        q_fp8_meta: dict[str, float] | None = None
        if config.fp8:
            # We simulate FA3's block-quantized FP8 forward path by quantizing
            # each resident Q tile independently, which is also how we expose the
            # "incoherent processing" idea: every tile carries its own scale.
            q_block, q_fp8_meta = _simulate_fp8_block(q_block)

        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
            fp8=config.fp8,
        )

        for kv_tile_id, _ in enumerate(k_slices):
            next_buffer = None
            if kv_tile_id + 1 < len(k_slices):
                # Prefetch the next logical stage into the alternate buffer.
                next_buffer = _load_tile(
                    buffer_id=(active_buffer.buffer_id + 1) % max(config.num_stages, 2),
                    tile_id=kv_tile_id + 1,
                    tile_slice=k_slices[kv_tile_id + 1],
                    k=k,
                    v=v,
                    fp8=config.fp8,
                )

            # Real FA3 overlaps producer and consumer warpgroups here. We keep the
            # stages explicit, but execute them sequentially for clarity.
            # Stage 1: consume the active K/V tile to build the score block.
            scores, valid_mask = _compute_scores_from_blocks(
                q_block=q_block,
                k_block=active_buffer.k_block,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                q_len=q.shape[2],
                kv_len=k.shape[2],
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # Stage 2: update local softmax statistics and form the unnormalized
            # tile contribution for P @ V.
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                active_buffer.v_block,
            )
            # Stage 3: merge the contribution into the running query-tile state.
            (
                out_acc_blocks[q_tile_id],
                normalizer_blocks[q_tile_id],
                row_max_blocks[q_tile_id],
            ) = merge_state_unnormalized(
                out_acc_blocks[q_tile_id],
                normalizer_blocks[q_tile_id],
                row_max_blocks[q_tile_id],
                block_max,
                block_sum,
                weighted_values,
            )
            pipeline_trace.append(
                {
                    "q_tile": q_tile_id,
                    "kv_tile": active_buffer.tile_id,
                    "buffer_id": active_buffer.buffer_id,
                    "prefetched_next_tile": next_buffer.tile_id
                    if next_buffer is not None
                    else -1,
                    "fp8": config.fp8,
                    "q_scale": q_fp8_meta["scale"] if q_fp8_meta is not None else None,
                    "k_scale": active_buffer.fp8_meta["k_scale"]
                    if active_buffer.fp8_meta is not None
                    else None,
                    "v_scale": active_buffer.fp8_meta["v_scale"]
                    if active_buffer.fp8_meta is not None
                    else None,
                }
            )
            active_buffer = next_buffer
            if active_buffer is None:
                break

    # The deferred softmax division is applied inside `assemble_forward_result`.
    return assemble_forward_result(
        out_acc_blocks,
        normalizer_blocks,
        row_max_blocks,
        normalized=False,
        out_dtype=q.dtype,
        saved_state={
            "pipeline_trace": pipeline_trace,
            "fp8_enabled": config.fp8,
            "fp8_format": "simulated-e4m3",
            "quantization_mode": "per-tile-block",
        }
        if config.keep_debug_state
        else {},
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
    """Run the FA3 staged-pipeline backward pass.

    Mirrors the forward's producer/consumer structure: consume one active
    tile, prepare the next one, then fold the local derivatives into the
    running dQ / dK / dV accumulators.

    Args:
        q, k, v: The same tensors the forward pass was called with.
        grad_out: Gradient w.r.t. the forward output, same shape as
            ``forward_result.out``.
        forward_result: The result returned by `forward`.
        causal, key_padding_mask, config: Must match the forward call.

    Returns:
        A `BackwardResult` with the gradients dQ, dK and dV.

    Raises:
        ValueError: If ``config.fp8`` is True. Like the official FA3 release,
            the FP8 path supports forward only.
    """
    config = config or FlashAttentionConfig()
    if config.fp8:
        raise ValueError("FA3 FP8 backward is unsupported in this educational repo")
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        key_padding_mask=key_padding_mask,
    )

    dQ, dK, dV = init_gradients(q, k, v)
    q_slices = iter_block_slices(q.shape[2], config.block_size_q)
    k_slices = iter_block_slices(k.shape[2], config.block_size_kv)
    pipeline_trace: list[dict[str, Any]] = []

    for q_tile_id, q_slice in enumerate(q_slices):
        if not k_slices:
            continue

        q_block = q[:, :, q_slice, :]
        # Backward keeps the same staged interpretation: consume one active tile,
        # optionally prepare the next tile, then fold the local derivatives into
        # the running dQ / dK / dV accumulators.
        dQ_block = torch.zeros_like(q_block, dtype=torch.float32)
        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
            fp8=False,
        )

        for kv_tile_id, _ in enumerate(k_slices):
            next_buffer = None
            if kv_tile_id + 1 < len(k_slices):
                next_buffer = _load_tile(
                    buffer_id=(active_buffer.buffer_id + 1) % max(config.num_stages, 2),
                    tile_id=kv_tile_id + 1,
                    tile_slice=k_slices[kv_tile_id + 1],
                    k=k,
                    v=v,
                    fp8=False,
                )

            # In the actual Hopper kernels, the backward mainloop is heavily
            # constrained by register pressure and overlap scheduling. We keep the
            # simplified stage order visible rather than reproducing those details.
            scores, valid_mask = _compute_scores_from_blocks(
                q_block=q_block,
                k_block=active_buffer.k_block,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                q_len=q.shape[2],
                kv_len=k.shape[2],
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            local_dQ, local_dK, local_dV = backward_step(
                q_block=q_block,
                k_block=active_buffer.k_block,
                v_block=active_buffer.v_block,
                out_block=forward_result.out[:, :, q_slice, :],
                grad_out_block=grad_out[:, :, q_slice, :],
                scores=scores,
                valid_mask=valid_mask,
                lse_block=forward_result.lse[:, :, q_slice, :],
            )
            dQ_block += local_dQ
            dK[:, :, active_buffer.tile_slice, :] += local_dK
            dV[:, :, active_buffer.tile_slice, :] += local_dV
            pipeline_trace.append(
                {
                    "q_tile": q_tile_id,
                    "kv_tile": active_buffer.tile_id,
                    "buffer_id": active_buffer.buffer_id,
                }
            )
            active_buffer = next_buffer
            if active_buffer is None:
                break

        dQ[:, :, q_slice, :] = dQ_block

    return BackwardResult(
        dQ=dQ.to(q.dtype),
        dK=dK.to(k.dtype),
        dV=dV.to(v.dtype),
        debug_state={"pipeline_trace": pipeline_trace}
        if config.keep_debug_state
        else {},
    )


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    """Convenience wrapper around `forward` returning only the output tensor."""
    return forward(
        q,
        k,
        v,
        causal=causal,
        key_padding_mask=key_padding_mask,
        config=config,
    ).out
