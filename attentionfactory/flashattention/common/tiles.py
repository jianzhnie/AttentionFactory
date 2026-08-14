"""Input validation, block partitioning and per-tile state allocation."""

from __future__ import annotations

import torch

from .masking import normalize_key_padding_mask


def prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Validate q/k/v shapes and normalize the padding mask.

    Expects the layout ``(batch, heads, seq, dim)`` for all three tensors.
    Q/K must share the head dimension; V's head dimension may differ.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape (batch, heads, seq, dim)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimension mismatch between q, k and v")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("head dimension mismatch between q, k and v")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[3] != k.shape[3]:
        raise ValueError("q and k must have the same hidden dimension")
    if v.shape[3] == 0:
        raise ValueError("v hidden dimension must be non-zero")

    normalized_mask = normalize_key_padding_mask(
        key_padding_mask,
        batch_size=q.shape[0],
        kv_len=k.shape[2],
        device=q.device,
    )
    return q, k, normalized_mask


def iter_block_slices(length: int, block_size: int) -> list[slice]:
    """Split ``range(length)`` into consecutive block-sized slices.

    The block size is clamped to ``[1, length]``; the final slice may be
    shorter than the others.
    """
    actual_block_size = max(1, min(block_size, length))
    return [
        slice(start, min(start + actual_block_size, length))
        for start in range(0, length, actual_block_size)
    ]


def init_block_state(
    q: torch.Tensor,
    v: torch.Tensor,
    q_slices: list[slice],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Allocate the per-query-tile running state for the online softmax.

    Returns one ``(output accumulator, normalizer, row max)`` triple per query
    tile. The output accumulator has V's head dimension, which may differ from
    Q/K's. Normalizer and row max are kept in float32 for numerical stability.
    """
    out_blocks = [
        torch.zeros(
            q.shape[0],
            q.shape[1],
            q_slice.stop - q_slice.start,
            v.shape[-1],
            device=q.device,
            dtype=q.dtype,
        )
        for q_slice in q_slices
    ]
    normalizer_blocks = [
        torch.zeros(
            q.shape[0],
            q.shape[1],
            q_slice.stop - q_slice.start,
            1,
            device=q.device,
            dtype=torch.float32,
        )
        for q_slice in q_slices
    ]
    row_max_blocks = [
        torch.full_like(block, float("-inf")) for block in normalizer_blocks
    ]
    return out_blocks, normalizer_blocks, row_max_blocks
