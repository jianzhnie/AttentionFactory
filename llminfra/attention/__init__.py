"""Attention mechanisms: classic variants, sparse/linear forms and hybrids.

All full-attention modules share the `BaseAttention` interface; linear and
state-space-flavored variants (``linear_attention``, ``lightning_attention``,
``gated_delta_net``) live here too since they are drop-in attention layers.
"""

from .alibi_attention import AlibiAttention
from .attention_residual import AttentionResidual
from .base import BaseAttention, validate_attention_inputs
from .block_sparse_attention import BlockSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention
from .flash_mla import FlashMLA
from .gated_delta_net import GatedDeltaNet
from .gqa import GroupQueryAttention
from .hybrid_attention import HybridAttention
from .lightning_attention import LightningAttention
from .linear_attention import LinearAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention
from .ring_attention import RingAttention, ring_attention
from .sliding_window_attention import SlidingWindowAttention

__all__ = [
    "AlibiAttention",
    "AttentionResidual",
    "BaseAttention",
    "BlockSparseAttention",
    "CompressedSparseAttention",
    "FlashMLA",
    "GatedDeltaNet",
    "GroupQueryAttention",
    "HybridAttention",
    "LightningAttention",
    "LinearAttention",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "RingAttention",
    "SlidingWindowAttention",
    "ring_attention",
    "validate_attention_inputs",
]
