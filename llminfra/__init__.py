"""LLMInfra: attention, positional encoding, MoE and model building blocks.

The package is organized into subpackages by role:

- `attention`: classic, sparse, linear and hybrid attention variants
- `flashattention`: educational FlashAttention v1-v4 implementations
- `layers`: FFN / norm / SSM / transformer block layers
- `moe`: Mixture-of-Experts components
- `inference`: KV cache, paging and decoding utilities

All public names are re-exported here, so ``from llminfra import X`` works
regardless of the internal layout.
"""

from .attention import (
    AlibiAttention,
    AttentionResidual,
    BaseAttention,
    BlockSparseAttention,
    CompressedSparseAttention,
    FlashMLA,
    GatedDeltaNet,
    GroupQueryAttention,
    HybridAttention,
    LightningAttention,
    LinearAttention,
    MultiHeadAttention,
    MultiHeadLatentAttention,
    MultiQueryAttention,
    RingAttention,
    SlidingWindowAttention,
    ring_attention,
)
from .flashattention import FlashAttention, flash_attention
from .inference import (
    BlockSparseIndexer,
    EagleSpeculator,
    MultiTokenPredictionHead,
    OnDiskKVStore,
    PagedAttentionCache,
    PagedKVBlockAllocator,
    SpeculativeDecoder,
    mtp_loss,
    paged_attention,
)
from .layers import (
    FeedForward,
    Mamba2Layer,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
)
from .model import CausalLMModel, CausalLMOutput
from .moe import (
    DeepSeekMoE,
    ExpertFFN,
    ExpertParallelMoE,
    LatentMoE,
    MixtureOfExperts,
    TopKRouter,
    load_balance_loss,
)
from .positional import (
    ALiBiBias,
    DynamicNTKRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    MultiModalRotaryPositionEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    RotaryPositionEmbedding,
    TwoDimensionalPositionEmbedding,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    apply_rotary_pos_emb,
    get_positional_encoding,
)
from .quantization import (
    FakeQuantizer,
    QATWrapper,
    QuantizationConfig,
    build_quantized,
)
from .registry import build_attention, build_positional_encoding, list_attentions

__all__ = [
    "ALiBiBias",
    "AlibiAttention",
    "AttentionResidual",
    "BaseAttention",
    "BlockSparseAttention",
    "BlockSparseIndexer",
    "CausalLMModel",
    "CausalLMOutput",
    "CompressedSparseAttention",
    "DeepSeekMoE",
    "DynamicNTKRotaryEmbedding",
    "EagleSpeculator",
    "ExpertFFN",
    "ExpertParallelMoE",
    "FakeQuantizer",
    "FeedForward",
    "FlashAttention",
    "FlashMLA",
    "GatedDeltaNet",
    "GroupQueryAttention",
    "HybridAttention",
    "LatentMoE",
    "LightningAttention",
    "LinearAttention",
    "LongRoPEScaledRotaryEmbedding",
    "Mamba2Layer",
    "MixtureOfExperts",
    "MultiHeadAttention",
    "MultiHeadLatentAttention",
    "MultiModalRotaryPositionEmbedding",
    "MultiQueryAttention",
    "MultiTokenPredictionHead",
    "OnDiskKVStore",
    "PagedAttentionCache",
    "PagedKVBlockAllocator",
    "PartialRotaryPositionEmbedding",
    "PositionInterpolation",
    "QATWrapper",
    "QuantizationConfig",
    "RMSNorm",
    "RingAttention",
    "RotaryPositionEmbedding",
    "SlidingWindowAttention",
    "SpeculativeDecoder",
    "SwiGLUFFN",
    "TopKRouter",
    "TransformerBlock",
    "TwoDimensionalPositionEmbedding",
    "YaRNParameters",
    "YaRNScaledRotaryEmbedding",
    "apply_rotary_pos_emb",
    "build_attention",
    "build_positional_encoding",
    "build_quantized",
    "flash_attention",
    "get_positional_encoding",
    "list_attentions",
    "load_balance_loss",
    "mtp_loss",
    "paged_attention",
    "ring_attention",
]
