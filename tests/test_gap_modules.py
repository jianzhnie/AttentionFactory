"""Tests for gap-filling modules from the 2026 attention review."""

import math

import pytest
import torch

from attentionfactory import (
    AlibiAttention,
    CausalLMModel,
    CompressedSparseAttention,
    FlashMLA,
    LongRoPEScaledRotaryEmbedding,
    Mamba2Layer,
    OnDiskKVStore,
    RingAttention,
    SpeculativeDecoder,
    TopKRouter,
    TwoDimensionalPositionEmbedding,
    build_attention,
    load_balance_loss,
    ring_attention,
)

HIDDEN = 32
HEADS = 4
SEQ = 8
BATCH = 2


def dense_reference(q, k, v, causal=True):
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    if causal:
        mask = torch.tril(torch.ones(q.size(2), k.size(2), dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def test_ring_attention_matches_dense():
    q = torch.randn(BATCH, HEADS, SEQ, 8)
    k = torch.randn(BATCH, HEADS, SEQ, 8)
    v = torch.randn(BATCH, HEADS, SEQ, 8)
    actual = ring_attention(q, k, v, causal=True, num_chunks=3)
    torch.testing.assert_close(actual, dense_reference(q, k, v), atol=1e-5, rtol=1e-4)


def test_ring_attention_module_shape_and_gradient():
    layer = RingAttention(HIDDEN, HEADS, num_chunks=3)
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_compressed_sparse_attention_shape_and_gradient():
    layer = CompressedSparseAttention(
        HIDDEN,
        HEADS,
        compress_ratio=2,
        num_kv_groups=2,
        top_k=4,
    )
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_alibi_attention_shape_and_causal_weights():
    layer = AlibiAttention(HIDDEN, HEADS, num_kv_groups=2)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    out, weights = layer(x, return_attention_weights=True)
    assert out.shape == x.shape
    future = torch.triu(torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
    assert weights[:, :, future].abs().max().item() == 0.0


def test_flash_mla_interface():
    layer = FlashMLA(HIDDEN, HEADS, q_latent_size=8, kv_latent_size=12)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    out = layer.prefill(x)
    assert out.shape == x.shape
    assert len(layer.latent_cache) == 1
    assert layer.decode(x).shape == x.shape
    layer.reset_cache()
    assert len(layer.latent_cache) == 0


def test_speculative_decoder_accepts_deterministic_tokens():
    vocab = 16

    def draft(input_ids):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab)
        logits[..., 1] = 1.0
        return logits

    def target(input_ids):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab)
        logits[..., 1] = 1.0
        return logits

    decoder = SpeculativeDecoder(draft, target, num_speculative_tokens=3)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    output = decoder(input_ids)
    assert output.size(1) == 7
    assert (output[:, 4:] == 1).all()


def test_mamba2_layer_shape_and_gradient():
    layer = Mamba2Layer(HIDDEN, d_state=8)
    x = torch.randn(BATCH, SEQ, HIDDEN, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_longrope_and_2d_position_are_finite():
    factors = [1.0, 2.0, 3.0, 4.0]
    longrope = LongRoPEScaledRotaryEmbedding(
        dim=8,
        original_max_position_embeddings=4,
        max_seq_len=16,
        long_factor=factors,
        short_factor=factors,
    )
    x = torch.randn(1, 1, 8, 8)
    assert torch.isfinite(longrope(x)).all()
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    y = torch.randn(1, 8, 8)
    assert torch.isfinite(two_d(y)).all()


def test_on_disk_kv_store(tmp_path):
    store = OnDiskKVStore(tmp_path / "kv")
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    store.save(1, key, value)
    loaded_key, loaded_value = store.load(1)
    torch.testing.assert_close(loaded_key, key)
    torch.testing.assert_close(loaded_value, value)
    store.delete(1)
    with pytest.raises(FileNotFoundError):
        store.load(1)


def test_load_balance_loss_is_finite():
    router = TopKRouter(HIDDEN, num_experts=8, top_k=2)
    x = torch.randn(16, HIDDEN)
    logits = router.routing_logits(x)
    _, indices = router(x)
    loss = load_balance_loss(logits, indices, num_experts=8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


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
