"""Attention mechanisms: classic variants, sparse/linear forms and hybrids.

All full-attention modules share the `BaseAttention` interface; linear and
state-space-flavored variants (``linear``, ``lightning``, ``gated_delta_net``)
live here too since they are drop-in attention layers. File names match the
mechanism name without a redundant ``_attention`` suffix.
"""

from .alibi_attention import ALiBiAttention, AlibiAttention
from .attention_residual import AttentionResidual
from .base_attention import BaseAttention, validate_attention_inputs
from .block_sparse_attention import BlockSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention
from .flash_mla_attention import FlashMLA
from .gated_delta_net import GatedDeltaNet
from .grouped_query_attention import GroupedQueryAttention, GroupQueryAttention
from .hybrid_attention import HybridAttention
from .kimi_delta_attention import KDAAttention, KimiDeltaAttention
from .lightning_attention import LightningAttention
from .linear_attention import LinearAttention
from .multi_head_attention import MultiHeadAttention
from .multi_head_latent_attention import MultiHeadLatentAttention
from .multi_query_attention import MultiQueryAttention
from .ring_attention import RingAttention, distributed_ring_attention, ring_attention
from .sliding_window_attention import SlidingWindowAttention
from .sparse_attention import (
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
    "GroupedQueryAttention",
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
