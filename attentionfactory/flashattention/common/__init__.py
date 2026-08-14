"""Shared building blocks for the educational FlashAttention versions.

This package collects everything the per-version modules (``fa1``-``fa4``)
have in common: configuration, result types, mask construction, blocking and
input validation, the online-softmax math primitives, an autograd bridge, and
a dense reference implementation used for correctness checks.
"""

from .autograd import TiledAttentionFunction
from .config import FlashAttentionConfig
from .masking import (
    build_block_mask,
    build_full_mask,
    normalize_key_padding_mask,
)
from .ops import (
    assemble_forward_result,
    block_scores_and_mask,
    compute_block_gradients,
    compute_block_softmax,
    finalize_output,
    lse_from_state,
    merge_normalized_block,
    merge_unnormalized_block,
    probabilities_from_lse,
    scaled_scores,
)
from .reference import reference_attention
from .tiles import (
    block_slices,
    init_block_state,
    init_gradients,
    prepare_inputs,
)
from .types import BackwardResult, ForwardResult

__all__ = [
    "BackwardResult",
    "FlashAttentionConfig",
    "ForwardResult",
    "TiledAttentionFunction",
    "assemble_forward_result",
    "block_scores_and_mask",
    "block_slices",
    "build_block_mask",
    "build_full_mask",
    "compute_block_gradients",
    "compute_block_softmax",
    "finalize_output",
    "init_block_state",
    "init_gradients",
    "lse_from_state",
    "merge_normalized_block",
    "merge_unnormalized_block",
    "normalize_key_padding_mask",
    "prepare_inputs",
    "probabilities_from_lse",
    "reference_attention",
    "scaled_scores",
]
