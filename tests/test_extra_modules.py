"""Tests for Lightning Attention, LatentMoE and supporting modules."""

import pytest
import torch

from attentionfactory import (
    AttentionResidual,
    BlockSparseAttention,
    BlockSparseIndexer,
    LatentMoE,
    LightningAttention,
    MultiTokenPredictionHead,
    build_attention,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def test_lightning_attention_shape_and_gradient():
    layer = LightningAttention(
        HIDDEN,
        HEADS,
        feature_dim=8,
        block_size=2,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_lightning_attention_does_not_return_weights():
    layer = LightningAttention(HIDDEN, HEADS, block_size=2)
    with pytest.raises(ValueError, match="does not materialize"):
        layer(torch.randn(BATCH, SEQ, HIDDEN), return_attention_weights=True)


def test_lightning_attention_masked_input_is_finite():
    layer = LightningAttention(HIDDEN, HEADS, block_size=2)
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = layer(torch.randn(BATCH, SEQ, HIDDEN), attention_mask=mask)
    assert torch.isfinite(out).all()


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


def test_attention_residual_shape_and_gradient():
    residual = AttentionResidual(HIDDEN)
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    attention_output = torch.randn(BATCH, SEQ, HIDDEN)
    out = residual(x, attention_output)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_block_sparse_indexer_shape_and_causality():
    indexer = BlockSparseIndexer(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        block_size=2,
        top_k=2,
        max_seq_len=16,
        causal=True,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN)
    indices = indexer(x)
    assert indices.shape == (BATCH, HEADS, 4, 2)
    # Causal indexer must never select a future block.
    for query_block in range(4):
        assert (indices[:, :, query_block] <= query_block).all()


def test_block_sparse_indexer_integrates_with_sparse_attention():
    indexer = BlockSparseIndexer(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        block_size=2,
        top_k=4,
        max_seq_len=16,
        causal=True,
    )
    sparse = BlockSparseAttention(
        HIDDEN,
        HEADS,
        block_size=2,
        num_kv_groups=2,
        top_k=4,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN)
    indices = indexer(x)
    assert sparse(x, block_indices=indices).shape == x.shape


def test_multi_token_prediction_returns_multiple_logits():
    head = MultiTokenPredictionHead(
        hidden_size=HIDDEN,
        vocab_size=64,
        num_predictions=3,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN)
    logits = head(x)
    assert len(logits) == 3
    assert all(item.shape == (BATCH, SEQ, 64) for item in logits)


def test_lightning_attention_in_registry():
    layer = build_attention(
        "lightning",
        hidden_size=HIDDEN,
        num_heads=HEADS,
        block_size=2,
    )
    assert isinstance(layer, LightningAttention)
