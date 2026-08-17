"""Tests for hybrid attention, norm, FFN and transformer block modules."""

import pytest
import torch

from attentionfactory import (
    FeedForward,
    HybridAttention,
    MixtureOfExperts,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
    get_positional_encoding,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def make_input():
    return torch.randn(BATCH, SEQ, HIDDEN)


def test_hybrid_attention_routes_by_layer_index():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=3,
        full_interval=1,
        linear_feature_dim=8,
        num_kv_groups=2,
    )
    x = make_input()
    assert hybrid.is_linear_layer(0)
    assert hybrid.is_linear_layer(2)
    assert not hybrid.is_linear_layer(3)
    assert hybrid(x, layer_index=0).shape == x.shape
    assert hybrid(x, layer_index=3).shape == x.shape


def test_hybrid_attention_gradient_flows():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=1,
        full_interval=1,
        linear_feature_dim=8,
    )
    x = make_input().requires_grad_(True)
    hybrid(x, layer_index=1).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hybrid_attention_returns_full_attention_weights():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=1,
        full_interval=1,
        num_kv_groups=2,
    )
    out, weights = hybrid(make_input(), return_attention_weights=True, layer_index=1)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert weights.shape == (BATCH, HEADS, SEQ, SEQ)


def test_partial_rope_shape_and_norm_preserved():
    rope = PartialRotaryPositionEmbedding(
        dim=8, partial_rotary_factor=0.5, max_seq_len=16
    )
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_position_interpolation_is_finite():
    interpolate = PositionInterpolation(
        dim=8,
        original_max_position_embeddings=2048,
        max_seq_len=4096,
    )
    x = torch.randn(1, 1, 128, 8)
    assert torch.isfinite(interpolate(x)).all()


def test_positional_factory_new_modes():
    assert isinstance(
        get_positional_encoding("partial_rope", dim=8, partial_rotary_factor=0.5),
        PartialRotaryPositionEmbedding,
    )
    assert isinstance(
        get_positional_encoding(
            "interpolation",
            dim=8,
            original_max_position_embeddings=2048,
        ),
        PositionInterpolation,
    )


def test_rms_norm_normalizes_last_dimension():
    norm = RMSNorm(HIDDEN)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    y = norm(x)
    mean_square = y.pow(2).mean(dim=-1)
    torch.testing.assert_close(
        mean_square, torch.ones_like(mean_square), atol=1e-5, rtol=1e-4
    )


def test_swiglu_ffn_shape_and_gradient():
    ffn = SwiGLUFFN(HIDDEN, intermediate_size=64)
    x = make_input().requires_grad_(True)
    out = ffn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_feed_forward_shape():
    ffn = FeedForward(HIDDEN, intermediate_size=64, activation="relu")
    out = ffn(make_input())
    assert out.shape == (BATCH, SEQ, HIDDEN)


def test_transformer_block_shape_and_gradient():
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
    )
    x = make_input().requires_grad_(True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_block_with_hybrid_attention():
    hybrid = HybridAttention(
        HIDDEN,
        HEADS,
        linear_interval=3,
        full_interval=1,
        linear_feature_dim=8,
        num_kv_groups=2,
    )
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        attention=hybrid,
    )
    x = make_input()
    assert block(x, layer_index=0).shape == x.shape
    assert block(x, layer_index=3).shape == x.shape


def test_transformer_block_with_moe_ffn():
    moe = MixtureOfExperts(
        hidden_size=HIDDEN,
        num_experts=4,
        intermediate_size=64,
        top_k=2,
    )
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
        ffn=moe,
    )
    x = make_input()
    assert block(x).shape == x.shape


def test_hybrid_attention_rejects_bad_intervals():
    with pytest.raises(ValueError, match="must be >= 1"):
        HybridAttention(HIDDEN, HEADS, linear_interval=0, full_interval=1)
