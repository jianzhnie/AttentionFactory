"""Educational positional encoding and long-context scaling utilities.

This module provides PyTorch implementations of the position-related
components referenced by mainstream attention architectures:

- Rotary Position Embedding (RoPE)
- YaRN-scaled RoPE
- Dynamic NTK-aware RoPE
- Attention with Linear Biases (ALiBi)

The implementations are intended for teaching and small-scale experiments.
Production deployments should compare against the official kernels and
Transformers implementations, especially for exact YaRN coefficients.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


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


class BasePositionalEncoding(nn.Module, ABC):
    """Base class for position encoders used by attention implementations."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional information to ``x``."""


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


@dataclass(frozen=True)
class YaRNParameters:
    """Parameters used by the YaRN frequency scaling approximation."""

    factor: float = 4.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0


def _yarn_find_correction_dim(
    num_rotations: float,
    dim: int,
    base: float,
    original_max_position_embeddings: int,
) -> float:
    return (
        dim
        * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        return torch.ones(dim)
    indices = torch.arange(dim, dtype=torch.float32)
    return 1.0 - ((max_val - indices) / (max_val - min_val)).clamp(0.0, 1.0)


class YaRNScaledRotaryEmbedding(BasePositionalEncoding):
    """YaRN-scaled RoPE for long-context extension.

    This is a teaching implementation of the interpolation/extrapolation
    blending formula. Exact numerical behavior should be checked against
    Transformers and the official YaRN repository.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        params: YaRNParameters,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.params = params
        self.base = float(base)

        inv_freq = _default_inv_freq(self.dim, self.base, dtype)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._register_ramp_mask()

    def _register_ramp_mask(self) -> None:
        low = _yarn_find_correction_dim(
            self.params.beta_fast,
            self.dim,
            self.base,
            self.params.original_max_position_embeddings,
        )
        high = _yarn_find_correction_dim(
            self.params.beta_slow,
            self.dim,
            self.base,
            self.params.original_max_position_embeddings,
        )
        low = max(0, min(low, self.dim // 2 - 1))
        high = max(0, min(high, self.dim // 2 - 1))
        ramp = _yarn_linear_ramp_mask(low, high, self.dim // 2)
        self.register_buffer("ramp", ramp, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply YaRN-scaled RoPE to ``x``."""
        scale = self.max_seq_len / self.params.original_max_position_embeddings
        linear_factor = max(1.0, 1.0 / scale)
        extrapolation = self.inv_freq / self.params.factor
        interpolation = self.inv_freq * linear_factor
        inv_freq = interpolation * (1.0 - self.ramp) + extrapolation * self.ramp

        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("s,f->sf", positions, inv_freq.to(x.dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return apply_rotary_pos_emb(x, cos, sin)


class DynamicNTKRotaryEmbedding(BasePositionalEncoding):
    """Dynamic NTK-aware RoPE scaling.

    The base frequency is increased when the sequence length exceeds the
    original training context, following the NTK-aware scaling idea used by
    several long-context models.
    """

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.max_seq_len = int(max_seq_len)
        self.base = float(base)
        self.dtype = dtype
        self.register_buffer(
            "base_inv_freq",
            _default_inv_freq(self.dim, self.base, dtype),
            persistent=False,
        )

    def _scaled_inv_freq(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.original_max_position_embeddings:
            return self.base_inv_freq
        scale = seq_len / self.original_max_position_embeddings
        adjusted_base = self.base * (scale ** (self.dim / (self.dim - 2)))
        return _default_inv_freq(self.dim, adjusted_base, self.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic NTK-scaled RoPE to ``x``."""
        seq_len = x.size(-2)
        inv_freq = self._scaled_inv_freq(seq_len).to(x.device)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("s,f->sf", positions, inv_freq.to(x.dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return apply_rotary_pos_emb(x, cos, sin)


class PartialRotaryPositionEmbedding(RotaryPositionEmbedding):
    """Partial RoPE used by Gemma 4 and DeepSeek-V4-style p-RoPE designs.

    Only the first ``rotated_dim`` channels are rotated; the remaining
    channels pass through unchanged.
    """

    def __init__(
        self,
        dim: int,
        partial_rotary_factor: float = 0.25,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not 0.0 < partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1]")
        rotated_dim = int(dim * partial_rotary_factor)
        rotated_dim = max(2, rotated_dim - rotated_dim % 2)
        if rotated_dim > dim:
            rotated_dim = dim - dim % 2
        self.full_dim = int(dim)
        self.partial_rotary_factor = float(partial_rotary_factor)
        self.rotated_dim = int(rotated_dim)
        super().__init__(
            self.rotated_dim,
            base=base,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate only the first ``rotated_dim`` channels of ``x``."""
        if x.size(-1) != self.full_dim:
            raise ValueError(
                f"x last dim {x.size(-1)} must equal full_dim {self.full_dim}"
            )
        rotated = super().forward(x[..., : self.rotated_dim])
        return torch.cat([rotated, x[..., self.rotated_dim :]], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"full_dim={self.full_dim}, rotated_dim={self.rotated_dim}, "
            f"partial_rotary_factor={self.partial_rotary_factor}"
        )


class PositionInterpolation(RotaryPositionEmbedding):
    """Simple position interpolation for RoPE long-context extension."""

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(dim, base=base, max_seq_len=max_seq_len, dtype=dtype)
        self.original_max_position_embeddings = int(original_max_position_embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interpolated positions to ``x``."""
        scale = self.original_max_position_embeddings / self.max_seq_len
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype) * scale
        freqs = torch.einsum("s,f->sf", positions, self.inv_freq.to(x.dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return apply_rotary_pos_emb(x, cos, sin)

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, "
            f"original_max_position_embeddings="
            f"{self.original_max_position_embeddings}"
        )


class LongRoPEScaledRotaryEmbedding(RotaryPositionEmbedding):
    """Simplified LongRoPE with separate short/long frequency factors."""

    def __init__(
        self,
        dim: int,
        original_max_position_embeddings: int,
        max_seq_len: int,
        long_factor: list[float] | tuple[float, ...],
        short_factor: list[float] | tuple[float, ...],
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(dim, base=base, max_seq_len=max_seq_len, dtype=dtype)
        if len(long_factor) != dim // 2 or len(short_factor) != dim // 2:
            raise ValueError("long_factor and short_factor must have dim//2 entries")
        self.original_max_position_embeddings = int(original_max_position_embeddings)
        self.register_buffer(
            "long_factor",
            torch.tensor(long_factor, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "short_factor",
            torch.tensor(short_factor, dtype=dtype),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LongRoPE with the appropriate frequency factor."""
        seq_len = x.size(-2)
        factors = (
            self.long_factor
            if seq_len > self.original_max_position_embeddings
            else self.short_factor
        )
        inv_freq = self.inv_freq * factors.to(self.inv_freq.device)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("s,f->sf", positions, inv_freq.to(x.dtype))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return apply_rotary_pos_emb(x, cos, sin)


class TwoDimensionalPositionEmbedding(BasePositionalEncoding):
    """Simple 2D position embedding for long-document layouts."""

    def __init__(
        self,
        embedding_dim: int,
        max_blocks: int,
        max_positions_per_block: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_blocks = int(max_blocks)
        self.max_positions_per_block = int(max_positions_per_block)
        self.block_embeddings = nn.Embedding(max_blocks, embedding_dim)
        self.position_embeddings = nn.Embedding(max_positions_per_block, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        block_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add block and within-block embeddings to ``x``."""
        seq_len = x.size(-2)
        if block_ids is None:
            block_ids = (
                torch.arange(seq_len, device=x.device) // self.max_positions_per_block
            )
        if positions is None:
            positions = (
                torch.arange(seq_len, device=x.device) % self.max_positions_per_block
            )
        if block_ids.max() >= self.max_blocks:
            raise ValueError("block_ids exceeds max_blocks")
        if positions.max() >= self.max_positions_per_block:
            raise ValueError("positions exceeds max_positions_per_block")
        return (
            x
            + self.block_embeddings(block_ids).unsqueeze(0)
            + self.position_embeddings(positions).unsqueeze(0)
        )


class ALiBiBias(BasePositionalEncoding):
    """Attention with Linear Biases (ALiBi).

    ``forward`` returns a bias tensor of shape
    ``(1, num_heads, seq_len, seq_len)`` that can be added to attention
    scores. When ``causal`` is True, future positions receive ``-inf``.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        causal: bool = True,
        slope_base: float = 2.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.slope_base = float(slope_base)
        slopes = [
            slope_base ** (-8 * (head + 1) / self.num_heads)
            for head in range(self.num_heads)
        ]
        self.register_buffer("slopes", torch.tensor(slopes), persistent=False)

    def forward(self, x: torch.Tensor | int | None = None) -> torch.Tensor:
        """Return ALiBi bias for the sequence length of ``x`` if provided."""
        if isinstance(x, int):
            seq_len = x
        else:
            seq_len = self.max_seq_len if x is None else x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        q_pos = torch.arange(seq_len).view(-1, 1)
        k_pos = torch.arange(seq_len).view(1, -1)
        distance = q_pos - k_pos
        bias = -self.slopes[:, None, None] * distance.abs().unsqueeze(0)
        if self.causal:
            future = q_pos < k_pos
            bias = bias.masked_fill(future[None], float("-inf"))
        return bias.unsqueeze(0)


def get_positional_encoding(
    name: str,
    *,
    dim: int,
    num_heads: int | None = None,
    max_seq_len: int = 4096,
    **kwargs: object,
) -> BasePositionalEncoding:
    """Create a positional encoding module by name.

    Supported names: ``rope``, ``yarn``, ``ntk``, ``alibi``.
    """
    if name == "rope":
        return RotaryPositionEmbedding(dim, max_seq_len=max_seq_len, **kwargs)
    if name == "yarn":
        if "params" not in kwargs:
            raise ValueError("yarn requires a YaRNParameters instance")
        return YaRNScaledRotaryEmbedding(dim, max_seq_len, **kwargs)
    if name == "ntk":
        if "original_max_position_embeddings" not in kwargs:
            raise ValueError("ntk requires original_max_position_embeddings")
        return DynamicNTKRotaryEmbedding(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    if name == "partial_rope":
        return PartialRotaryPositionEmbedding(dim, max_seq_len=max_seq_len, **kwargs)
    if name == "interpolation":
        if "original_max_position_embeddings" not in kwargs:
            raise ValueError("interpolation requires original_max_position_embeddings")
        return PositionInterpolation(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    if name == "longrope":
        if not {"long_factor", "short_factor"} <= set(kwargs):
            raise ValueError("longrope requires long_factor and short_factor")
        if "original_max_position_embeddings" not in kwargs:
            raise ValueError("longrope requires original_max_position_embeddings")
        return LongRoPEScaledRotaryEmbedding(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    if name == "alibi":
        if num_heads is None:
            raise ValueError("alibi requires num_heads")
        return ALiBiBias(num_heads, max_seq_len, **kwargs)
    raise ValueError(f"Unknown positional encoding: {name}")
