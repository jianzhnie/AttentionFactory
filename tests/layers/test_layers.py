"""Tests for the non-attention layer modules.

Covers the FFN variants (FeedForward, SwiGLU), RMSNorm, the Mamba2 SSM layer
and TransformerBlock, including blocks wired with custom attention or MoE FFNs.
"""

import torch
from helpers import make_hidden_state

from llminfra import (
    FeedForward,
    HybridAttention,
    Mamba2Layer,
    MixtureOfExperts,
    RMSNorm,
    SwiGLUFFN,
    TransformerBlock,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


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
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out = ffn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_feed_forward_shape():
    ffn = FeedForward(HIDDEN, intermediate_size=64, activation="relu")
    out = ffn(make_hidden_state(BATCH, SEQ, HIDDEN))
    assert out.shape == (BATCH, SEQ, HIDDEN)


def test_mamba2_layer_shape_and_gradient():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
    out, state = layer(x)
    assert out.shape == x.shape
    assert state.shape == (BATCH, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mamba2_state_threading_matches_full_forward():
    """Chunked decoding with a threaded state must equal one full pass."""
    layer = Mamba2Layer(HIDDEN, d_state=8).eval()
    x = make_hidden_state(BATCH, SEQ, HIDDEN)

    full_out, _ = layer(x)
    first_out, state = layer(x[:, :3])
    second_out, state = layer(x[:, 3:], state=state)
    torch.testing.assert_close(torch.cat([first_out, second_out], dim=1), full_out)


def test_mamba2_empty_sequence():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    out, state = layer(x[:, :0])
    assert out.shape == (BATCH, 0, HIDDEN)
    assert state.shape == (BATCH, 8)


def test_transformer_block_shape_and_gradient():
    block = TransformerBlock(
        HIDDEN,
        HEADS,
        intermediate_size=64,
    )
    x = make_hidden_state(BATCH, SEQ, HIDDEN).requires_grad_(True)
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
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
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
    x = make_hidden_state(BATCH, SEQ, HIDDEN)
    assert block(x).shape == x.shape
