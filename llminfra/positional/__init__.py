"""Educational positional encoding and long-context scaling utilities.

This package provides PyTorch implementations of the position-related
components referenced by mainstream attention architectures:

- `rope`: Rotary Position Embedding (RoPE) core
- `scaling`: YaRN / dynamic-NTK / partial / interpolation / LongRoPE variants
- `alibi`: Attention with Linear Biases (ALiBi)
- `two_d`: two-dimensional block position embedding
- `factory`: name-based `get_positional_encoding` factory

The implementations are intended for teaching and small-scale experiments.
Production deployments should compare against the official kernels and
Transformers implementations, especially for exact YaRN coefficients.
"""

from .alibi import ALiBiBias
from .base import BasePositionalEncoding
from .factory import get_positional_encoding
from .mrope import MultiModalRotaryPositionEmbedding
from .rope import RotaryPositionEmbedding, apply_rotary_pos_emb
from .scaling import (
    DynamicNTKRotaryEmbedding,
    LONGROPE_PRESETS,
    LongRoPEPreset,
    LongRoPEScaledRotaryEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    get_longrope_preset,
    register_longrope_preset,
)
from .two_d import TwoDimensionalPositionEmbedding

__all__ = [
    "ALiBiBias",
    "BasePositionalEncoding",
    "DynamicNTKRotaryEmbedding",
    "LongRoPEScaledRotaryEmbedding",
    "LONGROPE_PRESETS",
    "LongRoPEPreset",
    "MultiModalRotaryPositionEmbedding",
    "PartialRotaryPositionEmbedding",
    "PositionInterpolation",
    "RotaryPositionEmbedding",
    "TwoDimensionalPositionEmbedding",
    "YaRNParameters",
    "YaRNScaledRotaryEmbedding",
    "apply_rotary_pos_emb",
    "get_positional_encoding",
    "get_longrope_preset",
    "register_longrope_preset",
]
