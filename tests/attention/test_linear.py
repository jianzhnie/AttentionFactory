"""Tests for the linear-complexity attention variants.

Covers kernel-based linear attention, Lightning Attention and Gated DeltaNet:
output shapes, gradient flow, determinism, mask handling and the refusal to
materialize quadratic attention weights.
"""

import math

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


def test_lightning_attention_accepts_combined_causal_mask():
    """A dense causal mask must reduce to its (empty) key-padding component."""
    layer = LightningAttention(
        HIDDEN, HEADS, feature_dim=8, block_size=2, causal=True
    ).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(layer(x, attention_mask=causal_mask), layer(x))


def test_lightning_attention_accepts_3d_dense_mask():
    """A dense (batch, seq, seq) mask must reduce over query rows, not crash."""
    layer = LightningAttention(HIDDEN, HEADS, block_size=2, causal=True).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(layer(x, attention_mask=causal_mask), layer(x))


def test_lightning_attention_intra_block_scale_uses_feature_dim():
    """Intra-block softmax scores contract over feature_dim, not head_dim."""
    feature_dim = 8
    assert feature_dim != HIDDEN // HEADS
    layer = LightningAttention(
        HIDDEN, HEADS, feature_dim=feature_dim, block_size=SEQ, causal=False
    ).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    # Single block and empty initial state: output is the intra-block term.
    q = layer.q_proj(x).view(BATCH, SEQ, HEADS, feature_dim).transpose(1, 2)
    k = layer.k_proj(x).view(BATCH, SEQ, HEADS, feature_dim).transpose(1, 2)
    v = layer.split_head(layer.v_proj(x))
    scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(feature_dim)
    ref = layer.o_proj(layer.combine_head(torch.softmax(scores, dim=-1) @ v))

    torch.testing.assert_close(layer(x), ref)


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


def test_gated_delta_net_accepts_combined_causal_mask():
    """A dense causal mask must reduce to its key-padding component."""
    layer = GatedDeltaNet(HIDDEN, HEADS, feature_dim=8).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(layer(x, attention_mask=causal_mask), layer(x))


def test_linear_attention_accepts_combined_causal_mask():
    """A dense causal mask must reduce to its (empty) key-padding component."""
    module = LinearAttention(HIDDEN, HEADS, feature_dim=16, causal=True).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, 1, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(module(x, attention_mask=causal_mask), module(x))


def test_linear_attention_accepts_3d_dense_mask():
    """A dense (batch, seq, seq) mask must reduce over query rows, not crash."""
    module = LinearAttention(HIDDEN, HEADS, causal=True).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    causal_mask = torch.tril(torch.ones(BATCH, SEQ, SEQ, dtype=torch.bool))
    torch.testing.assert_close(module(x, attention_mask=causal_mask), module(x))


def test_linear_attention_non_causal_runs():
    """The non-causal path must work (it previously crashed on an einsum)."""
    module = LinearAttention(HIDDEN, HEADS, causal=False).eval()
    out = module(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)
    assert torch.isfinite(out).all()


def _causal_cumsum_reference(module, x, attention_mask=None):
    """Slow reference: the previous per-step ``(s, f, d)`` cumsum implementation."""
    query = module._split(module.q_proj(x), module.feature_dim)
    key = module._split(module.k_proj(x), module.feature_dim)
    value = module.split_head(module.v_proj(x))
    mask = module._key_padding_mask(attention_mask, x.size(0))

    query = module._feature_map(query)
    key = module._feature_map(key)
    if mask is not None:
        key = key * mask.unsqueeze(-1)
        value = value * mask.unsqueeze(-1)

    kv_state = torch.einsum("bhsf,bhsd->bhsfd", key, value).cumsum(dim=2)
    out_unnorm = torch.einsum("bhsf,bhsfd->bhsd", query, kv_state)
    normalizer = torch.einsum("bhsf,bhsf->bhs", query, key.cumsum(dim=2))
    eps = torch.finfo(normalizer.dtype).eps
    safe = torch.where(
        normalizer >= 0, normalizer.clamp_min(eps), normalizer.clamp_max(-eps)
    )
    out = out_unnorm / safe.unsqueeze(-1)
    out = out.where(normalizer.unsqueeze(-1) != 0, torch.zeros_like(out))
    return module.o_proj(module.combine_head(out))


@pytest.mark.parametrize("seq_len", [1, 5, 16, 17, 40])
@pytest.mark.parametrize("kernel", ["elu", "relu", "linear"])
def test_linear_attention_chunked_matches_cumsum_reference(seq_len, kernel):
    """The chunked causal scan must equal the per-step cumsum reference.

    Compared in float64: the identity ("linear") kernel amplifies float32
    accumulation-order noise through small normalizers, which is benign
    rounding, not an algorithmic difference.
    """
    module = (
        LinearAttention(
            HIDDEN, HEADS, feature_dim=16, kernel=kernel, causal=True, chunk_size=16
        )
        .double()
        .eval()
    )
    x = make_hidden_state(BATCH, seq_len, HIDDEN).double()
    mask = torch.ones(BATCH, 1, 1, seq_len, dtype=torch.bool)
    mask[0, 0, 0, -2:] = False  # padding must contribute zero to the state

    for m in (None, mask):
        reference = _causal_cumsum_reference(module, x, m)
        torch.testing.assert_close(
            module(x, attention_mask=m), reference, rtol=1e-5, atol=1e-6
        )


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
