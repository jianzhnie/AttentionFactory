"""Tests for positional encoding and MoE modules."""

import pytest
import torch

from llminfra import (
    ALiBiBias,
    DeepSeekMoE,
    DynamicNTKRotaryEmbedding,
    ExpertFFN,
    MixtureOfExperts,
    RotaryPositionEmbedding,
    TopKRouter,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    apply_rotary_pos_emb,
    get_positional_encoding,
)


def test_rotary_embedding_preserves_norm_and_shape():
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_apply_rotary_pos_emb_rejects_odd_dim():
    x = torch.randn(1, 1, 3, 7)
    cos = torch.ones(1, 1, 3, 7)
    sin = torch.zeros(1, 1, 3, 7)
    with pytest.raises(ValueError, match="even"):
        apply_rotary_pos_emb(x, cos, sin)


def test_yarn_and_dynamic_ntk_are_finite():
    params = YaRNParameters(
        factor=4.0,
        original_max_position_embeddings=2048,
    )
    yarn = YaRNScaledRotaryEmbedding(8, max_seq_len=4096, params=params)
    ntk = DynamicNTKRotaryEmbedding(
        8,
        original_max_position_embeddings=2048,
        max_seq_len=8192,
    )
    x = torch.randn(1, 2, 64, 8)
    assert torch.isfinite(yarn(x)).all()
    assert torch.isfinite(ntk(x)).all()


def test_alibi_shape_and_causal_mask():
    alibi = ALiBiBias(num_heads=4, max_seq_len=8)
    x = torch.randn(1, 4, 6, 8)
    bias = alibi(x)
    assert bias.shape == (1, 4, 6, 6)
    assert torch.isinf(bias[0, 0, 0, 1])
    assert bias[0, 0, 5, 4] < 0
    assert bias[0, 0, 5, 4] < bias[0, 1, 5, 4]


def test_positional_encoding_factory():
    assert isinstance(get_positional_encoding("rope", dim=8), RotaryPositionEmbedding)
    assert isinstance(get_positional_encoding("alibi", dim=8, num_heads=4), ALiBiBias)
    with pytest.raises(ValueError, match="Unknown"):
        get_positional_encoding("unknown", dim=8)


def test_expert_ffn_shape_and_gradient():
    expert = ExpertFFN(hidden_size=16, intermediate_size=32)
    x = torch.randn(3, 5, 16, requires_grad=True)
    out = expert(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_topk_router_weights_sum_to_one():
    router = TopKRouter(hidden_size=16, num_experts=8, top_k=2)
    x = torch.randn(4, 16)
    weights, indices = router(x)
    assert weights.shape == (4, 2)
    assert indices.shape == (4, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))
    assert indices.max().item() < 8


def test_mixture_of_experts_shape_and_gradient():
    moe = MixtureOfExperts(
        hidden_size=16,
        num_experts=8,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_deepseek_moe_includes_shared_experts():
    moe = DeepSeekMoE(
        hidden_size=16,
        num_routed_experts=8,
        num_shared_experts=2,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(2, 3, 16, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
