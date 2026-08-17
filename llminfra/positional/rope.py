"""Rotary Position Embedding (RoPE) core: rotation kernel and base module."""

from __future__ import annotations

import torch

from .base import BasePositionalEncoding


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to the last dimension of ``x``.

    Args:
        x: Tensor with an even final dimension.
        cos: Cosine frequencies broadcastable to ``x``.
        sin: Sine frequencies broadcastable to ``x``.
    """
    if x.size(-1) % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if cos.size(-1) == x.size(-1) // 2:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def _default_inv_freq(
    dim: int, base: float, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Compute the standard inverse frequency for RoPE."""
    if dim % 2 != 0:
        raise ValueError("RoPE dimension must be even")
    indices = torch.arange(0, dim, 2, dtype=dtype)
    return 1.0 / (base ** (indices / dim))


class RotaryPositionEmbedding(BasePositionalEncoding):
    """Rotary Position Embedding.

    Args:
        dim: Rotated feature dimension, normally ``head_dim``.
        base: RoPE base frequency.
        max_seq_len: Maximum sequence length used for precomputed frequencies.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.base = float(base)
        self.max_seq_len = int(max_seq_len)
        inv_freq = _default_inv_freq(self.dim, self.base, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate ``x`` by position indices derived from its sequence length."""
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("s,f->sf", positions, self.inv_freq.to(x.dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return apply_rotary_pos_emb(x, cos, sin)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, max_seq_len={self.max_seq_len}"
