"""Attention mechanisms: classic variants, sparse/linear forms and hybrids.

All full-attention modules share the `BaseAttention` interface; linear and
state-space-flavored variants (``linear``, ``lightning``, ``gated_delta_net``)
live here too since they are drop-in attention layers. File names match the
mechanism name without a redundant ``_attention`` suffix.
"""

from .alibi import ALiBiAttention, AlibiAttention
from .base import BaseAttention, validate_attention_inputs
from .block_sparse import BlockSparseAttention
from .compressed_sparse import CompressedSparseAttention
from .flash_mla import FlashMLA
from .gated_delta_net import GatedDeltaNet
from .gqa import GroupQueryAttention
from .hybrid import HybridAttention
from .kda import KDAAttention, KimiDeltaAttention
from .lightning import LightningAttention
from .linear import LinearAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention
from .residual import AttentionResidual
from .ring import RingAttention, distributed_ring_attention, ring_attention
from .sliding_window import SlidingWindowAttention
from .sparse_variants import (
    DeepSeekSparseAttention,
    DynamicSparseAttention,
    HierarchicalCompressedAttention,
    MiniMaxSparseAttention,
    QueryKeyBlockIndexer,
)

__all__ = [
    "ALiBiAttention",
    "AlibiAttention",
    "AttentionResidual",
    "BaseAttention",
    "BlockSparseAttention",
    "CompressedSparseAttention",
    "DeepSeekSparseAttention",
    "DynamicSparseAttention",
    "FlashMLA",
    "GatedDeltaNet",
    "GroupQueryAttention",
    "HierarchicalCompressedAttention",
    "HybridAttention",
    "KDAAttention",
    "KimiDeltaAttention",
    "LightningAttention",
    "LinearAttention",
    "MiniMaxSparseAttention",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiQueryAttention",
    "QueryKeyBlockIndexer",
    "RingAttention",
    "SlidingWindowAttention",
    "distributed_ring_attention",
    "ring_attention",
    "validate_attention_inputs",
]
