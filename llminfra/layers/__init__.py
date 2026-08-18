"""Reusable network layers: FFN, normalization, SSM and transformer blocks."""

from .activations import ACTIVATIONS, get_activation
from .ffn import FeedForward, SwiGLUFFN
from .ffn_variants import ClampedSwiGLUFFN, GeGLUFFN, ReGLUFFN, ffn_factory
from .hybrid_block import HybridLayerStack, HybridSSMBlock
from .hyperconnection import ManifoldConstrainedHyperConnection
from .norm import DeepNorm, LayerNorm, LayerScale, RMSNorm
from .ssm import Mamba2Layer, Mamba2State
from .transformer import TransformerBlock

__all__ = [
    "ACTIVATIONS",
    "ClampedSwiGLUFFN",
    "DeepNorm",
    "FeedForward",
    "GeGLUFFN",
    "HybridLayerStack",
    "HybridSSMBlock",
    "LayerNorm",
    "LayerScale",
    "Mamba2Layer",
    "Mamba2State",
    "ManifoldConstrainedHyperConnection",
    "RMSNorm",
    "ReGLUFFN",
    "SwiGLUFFN",
    "TransformerBlock",
    "ffn_factory",
    "get_activation",
]
