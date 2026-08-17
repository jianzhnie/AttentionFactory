"""Tests for the attention registry helpers and CausalLMModel.

Covers ``build_attention``/``list_attentions`` and the CausalLMModel wrapper:
output shapes, gradient flow, padding masks, MoE and hybrid-attention wiring,
tied embeddings and positional-encoding selection.
"""

import pytest
import torch

from llminfra import (
    AlibiAttention,
    CausalLMModel,
    CompressedSparseAttention,
    GatedDeltaNet,
    MultiHeadAttention,
    RingAttention,
    SlidingWindowAttention,
    build_attention,
    list_attentions,
)

HIDDEN = 32
HEADS = 4


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


def test_gap_modules_in_registry():
    assert isinstance(
        build_attention("ring", hidden_size=HIDDEN, num_heads=HEADS),
        RingAttention,
    )
    assert isinstance(
        build_attention("alibi", hidden_size=HIDDEN, num_heads=HEADS),
        AlibiAttention,
    )
    assert isinstance(
        build_attention(
            "compressed_sparse",
            hidden_size=HIDDEN,
            num_heads=HEADS,
            compress_ratio=2,
        ),
        CompressedSparseAttention,
    )


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


def test_causal_lm_with_longrope():
    factors = [1.0] * 16
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=32,
        attention_name="mha",
        positional="longrope",
        positional_kwargs={
            "original_max_position_embeddings": 16,
            "long_factor": factors,
            "short_factor": factors,
        },
    )
    logits = model(torch.randint(0, 32, (2, 24)))
    assert logits.shape == (2, 24, 32)


def test_causal_lm_with_2d_position():
    model = CausalLMModel(
        vocab_size=32,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=16,
        attention_name="mha",
        positional="2d",
        positional_kwargs={
            "max_blocks": 4,
            "max_positions_per_block": 4,
        },
    )
    logits = model(torch.randint(0, 32, (2, 8)))
    assert logits.shape == (2, 8, 32)
