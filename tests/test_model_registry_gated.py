"""Tests for Gated DeltaNet, registry helpers and CausalLMModel."""

import pytest
import torch

from attentionfactory import (
    CausalLMModel,
    GatedDeltaNet,
    MultiHeadAttention,
    SlidingWindowAttention,
    build_attention,
    list_attentions,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def test_gated_delta_net_shape_and_gradient():
    layer = GatedDeltaNet(
        HIDDEN,
        HEADS,
        feature_dim=8,
        normalize=True,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gated_delta_net_does_not_return_weights():
    layer = GatedDeltaNet(HIDDEN, HEADS, feature_dim=8)
    with pytest.raises(ValueError, match="does not materialize"):
        layer(torch.randn(BATCH, SEQ, HIDDEN), return_attention_weights=True)


def test_gated_delta_net_masked_input_is_finite():
    layer = GatedDeltaNet(HIDDEN, HEADS, feature_dim=8)
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = layer(torch.randn(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


def test_build_attention_registry():
    assert isinstance(
        build_attention("mha", hidden_size=HIDDEN, num_heads=HEADS),
        MultiHeadAttention,
    )
    assert isinstance(
        build_attention(
            "swa",
            hidden_size=HIDDEN,
            num_heads=HEADS,
            window_size=4,
        ),
        SlidingWindowAttention,
    )
    assert isinstance(
        build_attention(
            "gated_delta",
            hidden_size=HIDDEN,
            num_heads=HEADS,
            feature_dim=8,
        ),
        GatedDeltaNet,
    )
    assert "hybrid" in list_attentions()
    with pytest.raises(ValueError, match="Unknown attention"):
        build_attention("unknown", hidden_size=HIDDEN, num_heads=HEADS)


def test_causal_lm_shape_and_gradient():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_seq_len=16,
        attention_name="gqa",
    )
    input_ids = torch.randint(0, 32, (2, 7))
    logits = model(input_ids)
    assert logits.shape == (2, 7, 32)
    logits.sum().backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_causal_lm_with_padding_mask():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="gqa",
    )
    input_ids = torch.randint(0, 32, (2, 6))
    padding = torch.ones(2, 6, dtype=torch.bool)
    padding[0, -2:] = False
    logits = model(input_ids, attention_mask=padding)
    assert torch.isfinite(logits).all()


def test_causal_lm_with_moe():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="gqa",
        use_moe=True,
        num_experts=4,
        expert_top_k=2,
    )
    input_ids = torch.randint(0, 32, (2, 5))
    assert model(input_ids).shape == (2, 5, 32)


def test_causal_lm_with_hybrid_attention_returns_weights():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        attention_name="hybrid",
        attention_kwargs={
            "linear_interval": 1,
            "full_interval": 1,
            "linear_feature_dim": 8,
            "num_kv_groups": 1,
        },
    )
    input_ids = torch.randint(0, 32, (2, 5))
    logits, weights = model(input_ids, return_attention_weights=True)
    assert logits.shape == (2, 5, 32)
    assert weights.shape == (2, 2, 5, 5)


def test_causal_lm_ties_embeddings():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="mha",
        tie_word_embeddings=True,
    )
    assert model.lm_head.weight is model.embed_tokens.weight


def test_causal_lm_with_alibi():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        attention_name="mha",
        positional="alibi",
    )
    input_ids = torch.randint(0, 32, (2, 6))
    logits = model(input_ids)
    assert logits.shape == (2, 6, 32)
