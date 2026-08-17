"""Tests for the inference-time modules.

Covers paged attention (cache append/gather/reset/clone and the paged kernel
vs. dense attention), the on-disk KV store, speculative and EAGLE decoding,
the multi-token prediction head and the block-sparse indexer.
"""

import pytest
import torch

from llminfra import (
    BlockSparseAttention,
    BlockSparseIndexer,
    EagleSpeculator,
    MultiTokenPredictionHead,
    OnDiskKVStore,
    PagedAttentionCache,
    SpeculativeDecoder,
    paged_attention,
)

HIDDEN = 32
HEADS = 4
SEQ = 7
BATCH = 2


def test_paged_cache_append_and_gather_matches_dense():
    cache = PagedAttentionCache(
        num_blocks=4,
        block_size=2,
        num_heads=2,
        head_dim=4,
    )
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, key, value)

    gathered_key, gathered_value = cache.get(0)
    torch.testing.assert_close(gathered_key, key)
    torch.testing.assert_close(gathered_value, value)


def test_paged_attention_matches_dense_attention():
    cache = PagedAttentionCache(
        num_blocks=4,
        block_size=2,
        num_heads=2,
        head_dim=4,
    )
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, key, value)
    query = torch.randn(3, 2, 4)

    actual = paged_attention(
        query,
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        num_tokens=5,
        block_size=2,
        causal=True,
    )

    scores = torch.einsum("qhd,khd->qk", query, key) / (4**0.5)
    offset = 5 - 3
    mask = torch.arange(5) <= (torch.arange(3)[:, None] + offset)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qk,khd->qhd", weights, value)
    torch.testing.assert_close(actual, expected)


def test_paged_cache_reset_frees_blocks():
    cache = PagedAttentionCache(
        num_blocks=4,
        block_size=2,
        num_heads=1,
        head_dim=2,
    )
    cache.append(0, torch.randn(3, 1, 2), torch.randn(3, 1, 2))
    assert len(cache.block_tables[0]) == 2
    cache.reset(0)
    assert 0 not in cache.block_tables
    assert len(cache.allocator.free_blocks) == 4


def test_paged_cache_clone_sequence():
    cache = PagedAttentionCache(
        num_blocks=8,
        block_size=2,
        num_heads=2,
        head_dim=4,
    )
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, key, value)
    cache.clone_sequence(0, 1)
    cloned_key, cloned_value = cache.get(1)
    torch.testing.assert_close(cloned_key, key)
    torch.testing.assert_close(cloned_value, value)


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


def test_eagle_speculator_deterministic():
    vocab = 16

    def draft_head(hidden_states):
        logits = torch.zeros(hidden_states.size(0), hidden_states.size(1), vocab)
        logits[..., 2] = 1.0
        return logits

    def target_model(input_ids):
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab)
        logits[..., 2] = 1.0
        return logits

    speculator = EagleSpeculator(draft_head, target_model, num_speculative_tokens=3)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    hidden_states = torch.randn(2, 4, HIDDEN)
    output = speculator(input_ids, hidden_states)
    assert output.size(1) == 7
    assert (output[:, 4:] == 2).all()


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


def _constant_model(vocab_size: int = 32):
    def model(ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ids.size(0), ids.size(1), vocab_size)

    return model


def test_speculative_decoder_validates_arguments():
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        SpeculativeDecoder(model, model, num_speculative_tokens=0)
    with pytest.raises(ValueError, match=">= 0"):
        SpeculativeDecoder(model, model, temperature=-0.5)


def test_speculative_decoder_rejects_short_input():
    model = _constant_model()
    decoder = SpeculativeDecoder(model, model, num_speculative_tokens=4)
    with pytest.raises(ValueError, match="at least num_speculative_tokens"):
        decoder(torch.zeros(1, 2, dtype=torch.long))


def test_eagle_speculator_validates_arguments():
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        EagleSpeculator(model, model, num_speculative_tokens=0)
