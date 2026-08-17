"""Tests for the mixture-of-experts modules.

Covers ExpertFFN, the top-k router, MixtureOfExperts, DeepSeekMoE, LatentMoE,
ExpertParallelMoE and the load-balancing auxiliary loss.
"""

import torch
from helpers import make_hidden_state

from llminfra import (
    DeepSeekMoE,
    ExpertFFN,
    ExpertParallelMoE,
    LatentMoE,
    MixtureOfExperts,
    TopKRouter,
    load_balance_loss,
)

HIDDEN = 32
HEADS = 4
SEQ = 8
BATCH = 2


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


def test_latent_moe_shape_and_gradient():
    moe = LatentMoE(
        hidden_size=HIDDEN,
        latent_size=16,
        num_experts=4,
        intermediate_size=32,
        top_k=2,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_expert_parallel_moe_shape_and_gradient():
    moe = ExpertParallelMoE(
        hidden_size=HIDDEN,
        num_experts=8,
        intermediate_size=64,
        top_k=2,
        world_size=2,
        rank=0,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = moe(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_load_balance_loss_is_finite():
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2)
    x = torch.randn(16, HIDDEN)
    logits = router.routing_logits(x)
    _, indices = router(x)
    loss = load_balance_loss(logits, indices, num_experts=8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_router_noise_epsilon_actually_perturbs_routing():
    """noise_epsilon must scale the noise (adding a constant would be a no-op)."""
    router = TopKRouter(
        HIDDEN, num_experts=8, top_k=2, add_noise=True, noise_epsilon=1.0
    )
    router.train()
    x = make_hidden_state(16, 1, HIDDEN)[:, 0]

    weights_1, indices_1 = router(x)
    weights_2, indices_2 = router(x)

    assert not torch.equal(indices_1, indices_2) or not torch.allclose(
        weights_1, weights_2
    )
