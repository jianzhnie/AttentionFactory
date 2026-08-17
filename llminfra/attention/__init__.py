"""Attention mechanisms: classic variants, sparse/linear forms and hybrids.

All full-attention modules share the `BaseAttention` interface; linear and
state-space-flavored variants (``linear``, ``lightning``, ``gated_delta_net``)
live here too since they are drop-in attention layers. File names match the
mechanism name without a redundant ``_attention`` suffix.
"""

from .alibi import AlibiAttention
from .base import BaseAttention, validate_attention_inputs
from .block_sparse import BlockSparseAttention
from .compressed_sparse import CompressedSparseAttention
from .flash_mla import FlashMLA
from .gated_delta_net import GatedDeltaNet
from .gqa import GroupQueryAttention
from .hybrid import HybridAttention
from .lightning import LightningAttention
from .linear import LinearAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention
from .residual import AttentionResidual
from .ring import RingAttention, ring_attention
from .sliding_window import SlidingWindowAttention

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
