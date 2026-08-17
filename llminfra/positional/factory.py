"""Name-based factory for positional encoding modules."""

from __future__ import annotations

from .alibi import ALiBiBias
from .base import BasePositionalEncoding
from .rope import RotaryPositionEmbedding
from .scaling import (
    DynamicNTKRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    YaRNScaledRotaryEmbedding,
)
from .two_d import TwoDimensionalPositionEmbedding


def get_positional_encoding(
    name: str,
    *,
    dim: int,
    num_heads: int | None = None,
    max_seq_len: int = 4096,
    **kwargs: object,
) -> BasePositionalEncoding:
    """Create a positional encoding module by name.

    Supported names: ``rope``, ``yarn``, ``ntk``, ``partial_rope``,
    ``interpolation``, ``longrope``, ``2d`` and ``alibi``.
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
    if name == "2d":
        if not {"max_blocks", "max_positions_per_block"} <= set(kwargs):
            raise ValueError("2d requires max_blocks and max_positions_per_block")
        return TwoDimensionalPositionEmbedding(
            dim,
            max_blocks=kwargs["max_blocks"],
            max_positions_per_block=kwargs["max_positions_per_block"],
        )
    if name == "alibi":
        if num_heads is None:
            raise ValueError("alibi requires num_heads")
        return ALiBiBias(num_heads, max_seq_len, **kwargs)
    raise ValueError(f"Unknown positional encoding: {name}")
