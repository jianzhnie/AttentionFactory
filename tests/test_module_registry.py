"""Tests for the package-level module registries."""

import pytest

from llminfra import (
    ALiBiAttention,
    CompressedSparseAttention,
    GatedDeltaNet,
    MultiHeadAttention,
    RingAttention,
    SlidingWindowAttention,
    build_attention,
    list_attentions,
)

HIDDEN_SIZE = 32
NUM_HEADS = 4


def test_build_attention_registry() -> None:
    assert isinstance(
        build_attention("mha", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        MultiHeadAttention,
    )
    assert isinstance(
        build_attention(
            "swa",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            window_size=4,
        ),
        SlidingWindowAttention,
    )
    assert isinstance(
        build_attention(
            "gated_delta",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            feature_dim=8,
        ),
        GatedDeltaNet,
    )
    assert "hybrid" in list_attentions()

    with pytest.raises(ValueError, match="Unknown attention"):
        build_attention("unknown", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS)


def test_extended_attention_modules_are_registered() -> None:
    assert isinstance(
        build_attention("ring", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        RingAttention,
    )
    assert isinstance(
        build_attention("alibi", hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS),
        ALiBiAttention,
    )
    assert isinstance(
        build_attention(
            "compressed_sparse",
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            compress_ratio=2,
        ),
        CompressedSparseAttention,
    )
