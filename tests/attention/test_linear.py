"""Tests for the linear-complexity attention variants.

Covers kernel-based linear attention, Lightning Attention and Gated DeltaNet:
output shapes, gradient flow, determinism, mask handling and the refusal to
materialize quadratic attention weights.
"""

import pytest
import torch
from helpers import make_hidden_state

from llminfra import (
    GatedDeltaNet,
    LightningAttention,
    LinearAttention,
    build_attention,
)

HIDDEN = 64
HEADS = 4
BATCH = 2
SEQ = 7


@pytest.fixture()
def lin_attn():
    return LinearAttention(
        HIDDEN,
        HEADS,
        feature_dim=16,
        kernel="elu",
        causal=True,
        dropout=0.0,
    )


def test_linear_attention_shape_and_gradient(lin_attn):
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = lin_attn(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_linear_attention_is_deterministic(lin_attn):
    lin_attn.eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    torch.testing.assert_close(lin_attn(x), lin_attn(x))


def test_linear_attention_does_not_return_weights(lin_attn):
    with pytest.raises(ValueError, match="does not materialize"):
        lin_attn(make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True)


def test_linear_attention_masked_row_is_finite(lin_attn):
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = lin_attn(make_hidden_state(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


def test_lightning_attention_shape_and_gradient():
    layer = LightningAttention(
        HIDDEN,
        HEADS,
        feature_dim=8,
        block_size=2,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_lightning_attention_does_not_return_weights():
    layer = LightningAttention(HIDDEN, HEADS, block_size=2)
    with pytest.raises(ValueError, match="does not materialize"):
        layer(make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True)


def test_lightning_attention_masked_input_is_finite():
    layer = LightningAttention(HIDDEN, HEADS, block_size=2)
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = layer(make_hidden_state(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


def test_lightning_attention_in_registry():
    layer = build_attention(
        "lightning",
        hidden_size=HIDDEN,
        num_heads=HEADS,
        block_size=2,
    )
    assert isinstance(layer, LightningAttention)


def test_gated_delta_net_shape_and_gradient():
    layer = GatedDeltaNet(
        HIDDEN,
        HEADS,
        feature_dim=8,
        normalize=True,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_gated_delta_net_does_not_return_weights():
    layer = GatedDeltaNet(HIDDEN, HEADS, feature_dim=8)
    with pytest.raises(ValueError, match="does not materialize"):
        layer(make_hidden_state(BATCH, SEQ, HIDDEN), return_attention_weights=True)


def test_gated_delta_net_masked_input_is_finite():
    layer = GatedDeltaNet(HIDDEN, HEADS, feature_dim=8)
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = layer(make_hidden_state(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


def test_linear_attention_non_causal_runs():
    """The non-causal path must work (it previously crashed on an einsum)."""
    module = LinearAttention(HIDDEN, HEADS, causal=False).eval()
    out = module(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert torch.isfinite(out).all()


def test_linear_attention_causal_does_not_see_future():
    """Perturbing the last token must not change earlier outputs."""
    module = LinearAttention(HIDDEN, HEADS, kernel="linear", causal=True).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, -1] += 10.0

    torch.testing.assert_close(module(x)[:, :-1], module(perturbed)[:, :-1])


def test_lightning_attention_causal_does_not_see_future():
    """Intra-block softmax must be causally masked within each block."""
    module = LightningAttention(HIDDEN, HEADS, block_size=4, causal=True).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    perturbed = x.clone()
    perturbed[:, 1] += 10.0  # inside the first block, ahead of position 0

    torch.testing.assert_close(module(x)[:, 0], module(perturbed)[:, 0])
