"""Shared building blocks for the educational FlashAttention versions.

This package collects everything the per-version modules (``fa1``-``fa4``)
have in common: configuration, result types, mask construction, tiling and
input validation, the online-softmax math primitives, and a dense reference
implementation used for correctness checks.
"""

from .config import FlashAttentionConfig
from .masking import (
    build_block_mask,
    build_full_mask,
    normalize_key_padding_mask,
)
from .ops import (
    assemble_forward_result,
    backward_step,
    block_scores_and_mask,
    compute_local_statistics,
    finalize_unnormalized,
    lse_from_state,
    merge_state_normalized,
    merge_state_unnormalized,
    probabilities_from_lse,
    scaled_scores,
)
from .reference import reference_attention
from .tiles import (
    init_block_state,
    init_gradients,
    iter_block_slices,
    prepare_inputs,
)
from .types import BackwardResult, ForwardResult

__all__ = [
    "BackwardResult",
    "FlashAttentionConfig",
    "ForwardResult",
    "assemble_forward_result",
    "backward_step",
    "block_scores_and_mask",
    "build_block_mask",
    "build_full_mask",
    "compute_local_statistics",
    "finalize_unnormalized",
    "init_block_state",
    "init_gradients",
    "iter_block_slices",
    "lse_from_state",
    "merge_state_normalized",
    "merge_state_unnormalized",
    "normalize_key_padding_mask",
    "prepare_inputs",
    "probabilities_from_lse",
    "reference_attention",
    "scaled_scores",
]
