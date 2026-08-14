"""Boolean mask construction for causal and padding-masked attention.

Masks use the ``True = allowed`` convention and are broadcastable against
score tensors of shape ``(batch, heads, q_len, kv_len)``.
"""

from __future__ import annotations

import torch


def normalize_key_padding_mask(
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Validate a ``(batch_size, kv_len)`` padding mask and move it to bool/device."""
    if key_padding_mask is None:
        return None
    if key_padding_mask.shape != (batch_size, kv_len):
        raise ValueError(
            "key_padding_mask must have shape (batch_size, kv_len); "
            f"got {tuple(key_padding_mask.shape)}"
        )
    return key_padding_mask.to(device=device, dtype=torch.bool)


def build_block_mask(
    *,
    batch_size: int,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the validity mask for one (query tile, key tile) pair.

    Returns ``None`` when no masking applies at all, so callers can skip the
    masking path entirely. When ``kv_len > q_len`` the causal diagonal is
    aligned with the *end* of the key sequence, matching the usual convention
    for causal cross-attention.
    """
    block_mask = None

    if key_padding_mask is not None:
        block_mask = key_padding_mask[:, None, None, k_start:k_end]

    if causal:
        q_positions = torch.arange(q_start, q_end, device=device) + (kv_len - q_len)
        k_positions = torch.arange(k_start, k_end, device=device)
        causal_mask = (q_positions[:, None] >= k_positions[None, :])[None, None]
        block_mask = causal_mask if block_mask is None else block_mask & causal_mask

    if block_mask is None:
        return None
    return block_mask.expand(batch_size, 1, q_end - q_start, k_end - k_start)


def build_full_mask(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Build the validity mask for the whole score matrix at once."""
    return build_block_mask(
        batch_size=batch_size,
        q_start=0,
        q_end=q_len,
        k_start=0,
        k_end=kv_len,
        q_len=q_len,
        kv_len=kv_len,
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=device,
    )
