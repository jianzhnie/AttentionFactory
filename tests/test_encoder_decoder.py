"""Tests for the T5-style encoder-decoder modules.

Covers ``EncoderBlock``/``DecoderBlock``/``CrossAttention`` and the
``EncoderDecoderModel`` skeleton: output shapes, bidirectional encoder
visibility, causal decoder self-attention, cross-attention dependence on the
source sequence, padding masks and gradient flow.
"""

import torch

from llminfra.encoder_decoder_model import (
    CrossAttention,
    DecoderBlock,
    EncoderBlock,
    EncoderDecoderModel,
)


def _make_model(**overrides: object) -> EncoderDecoderModel:
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "num_heads": 2,
        "intermediate_size": 32,
        "max_seq_len": 16,
    }
    kwargs.update(overrides)
    model = EncoderDecoderModel(**kwargs)
    model.eval()
    return model


def test_cross_attention_supports_different_kv_len():
    attention = CrossAttention(hidden_size=16, num_heads=2)
    query_state = torch.randn(2, 3, 16)
    key_value_state = torch.randn(2, 7, 16)
    output, weights = attention(
        query_state, key_value_state, return_attention_weights=True
    )
    assert output.shape == (2, 3, 16)
    assert weights.shape == (2, 2, 3, 7)


def test_encoder_block_is_bidirectional():
    block = EncoderBlock(hidden_size=16, num_heads=2, intermediate_size=32)
    block.eval()
    hidden_state = torch.randn(1, 6, 16)
    perturbed = hidden_state.clone()
    perturbed[0, -1] += 1.0
    with torch.no_grad():
        base = block(hidden_state)
        changed = block(perturbed)
    # No causal structure: the last position influences the first one.
    assert not torch.allclose(base[0, 0], changed[0, 0])


def test_decoder_block_self_attention_is_causal():
    block = DecoderBlock(hidden_size=16, num_heads=2, intermediate_size=32)
    block.eval()
    hidden_state = torch.randn(1, 6, 16)
    encoder_output = torch.randn(1, 5, 16)
    causal = torch.tril(torch.ones(6, 6, dtype=torch.bool)).view(1, 1, 6, 6)
    perturbed = hidden_state.clone()
    perturbed[0, -1] += 1.0
    with torch.no_grad():
        base = block(hidden_state, encoder_output, self_attention_mask=causal)
        changed = block(perturbed, encoder_output, self_attention_mask=causal)
    # Future target positions must not influence earlier outputs.
    assert torch.allclose(base[0, :-1], changed[0, :-1])
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_encoder_decoder_model_shape_and_gradient():
    model = _make_model()
    src_ids = torch.randint(0, 32, (2, 7))
    tgt_ids = torch.randint(0, 32, (2, 5))
    logits = model(src_ids, tgt_ids)
    assert logits.shape == (2, 5, 32)
    logits.sum().backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_encoder_decoder_ties_embeddings_by_default():
    model = _make_model()
    assert model.lm_head.weight is model.embed_tokens.weight
    untied = _make_model(tie_word_embeddings=False)
    assert untied.lm_head.weight is not untied.embed_tokens.weight


def test_encoder_decoder_tgt_future_not_visible():
    model = _make_model()
    src_ids = torch.randint(0, 32, (1, 7))
    tgt_ids = torch.randint(0, 32, (1, 6))
    perturbed = tgt_ids.clone()
    perturbed[0, -1] = (perturbed[0, -1] + 1) % 32
    with torch.no_grad():
        base = model(src_ids, tgt_ids)
        changed = model(src_ids, perturbed)
    # Perturbing the last target token must not change earlier logits.
    assert torch.allclose(base[0, :-1], changed[0, :-1])
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_encoder_decoder_cross_attention_depends_on_src():
    model = _make_model()
    src_ids = torch.randint(0, 32, (1, 7))
    tgt_ids = torch.randint(0, 32, (1, 6))
    perturbed_src = src_ids.clone()
    perturbed_src[0, 3] = (perturbed_src[0, 3] + 1) % 32
    with torch.no_grad():
        base = model(src_ids, tgt_ids)
        changed = model(perturbed_src, tgt_ids)
    # Every decoder position attends to the encoder output, so all logits move.
    assert not torch.allclose(base, changed)


def test_encoder_decoder_with_padding_masks():
    model = _make_model()
    src_ids = torch.randint(0, 32, (2, 7))
    tgt_ids = torch.randint(0, 32, (2, 5))
    src_mask = torch.ones(2, 7, dtype=torch.bool)
    src_mask[0, -2:] = False
    tgt_mask = torch.ones(2, 5, dtype=torch.bool)
    tgt_mask[1, -1] = False
    logits = model(src_ids, tgt_ids, src_mask=src_mask, tgt_mask=tgt_mask)
    assert logits.shape == (2, 5, 32)
    assert torch.isfinite(logits).all()
