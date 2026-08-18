"""Name-based factory for positional encoding modules."""

from __future__ import annotations

import torch

from .alibi import ALiBiBias
from .base_position import BasePositionalEncoding
from .classic_position import (
    LearnedAbsolutePositionEmbedding,
    NoPositionEncoding,
    SinusoidalPositionEmbedding,
    T5RelativePositionBias,
)
from .multimodal_rope import MultiModalRotaryPositionEmbedding
from .rope_scaling import (
    DynamicNTKRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    YaRNScaledRotaryEmbedding,
)
from .rotary import RotaryPositionEmbedding
from .two_dimensional import TwoDimensionalPositionEmbedding


def get_positional_encoding(
    name: str,
    *,
    dim: int,
    num_heads: int | None = None,
    max_seq_len: int = 4096,
    **kwargs: object,
) -> BasePositionalEncoding:
    """Create a positional encoding module by name.

    Supported names are returned by :func:`list_positional_encodings`.
    """
    if name in {"none", "nope"}:
        return NoPositionEncoding(dim, max_seq_len=max_seq_len)
    if name in {"learned", "absolute"}:
        return LearnedAbsolutePositionEmbedding(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    if name in {"sinusoidal", "sinusoid"}:
        return SinusoidalPositionEmbedding(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
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
        if "preset" in kwargs:
            preset = kwargs.pop("preset")
            if not isinstance(preset, str):
                raise ValueError("longrope preset must be a string")
            dtype = kwargs.pop("dtype", torch.float32)
            base = kwargs.pop("base", 10000.0)
            if kwargs:
                raise ValueError(
                    f"unsupported longrope preset arguments: {sorted(kwargs)}"
                )
            if not isinstance(dtype, torch.dtype) or not isinstance(base, (int, float)):
                raise ValueError("longrope dtype/base have invalid types")
            return LongRoPEScaledRotaryEmbedding.from_preset(
                preset,
                dim=dim,
                base=float(base),
                dtype=dtype,
            )
        if not {"long_factor", "short_factor"} <= set(kwargs):
            raise ValueError("longrope requires long_factor and short_factor")
        if "original_max_position_embeddings" not in kwargs:
            raise ValueError("longrope requires original_max_position_embeddings")
        return LongRoPEScaledRotaryEmbedding(
            dim,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    if name == "mrope":
        if "mrope_section" not in kwargs:
            raise ValueError("mrope requires mrope_section")
        return MultiModalRotaryPositionEmbedding(
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
    if name in {"t5_bias", "t5_relative_bias"}:
        if num_heads is None:
            raise ValueError("t5_bias requires num_heads")
        return T5RelativePositionBias(
            num_heads,
            max_seq_len=max_seq_len,
            **kwargs,
        )
    raise ValueError(f"Unknown positional encoding: {name}")


def list_positional_encodings() -> list[str]:
    """Return canonical names accepted by :func:`get_positional_encoding`."""
    return [
        "none",
        "learned",
        "sinusoidal",
        "rope",
        "yarn",
        "ntk",
        "partial_rope",
        "interpolation",
        "longrope",
        "mrope",
        "2d",
        "alibi",
        "t5_bias",
    ]
