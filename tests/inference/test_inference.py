"""Tests for KV-cache, paged-attention, and sparse-indexing infrastructure."""

import pytest
import torch

from llminfra import (
    BlockSparseAttention,
    BlockSparseIndexer,
    OnDiskKVStore,
    PagedAttentionCache,
    paged_attention,
)


def test_paged_cache_append_and_gather_matches_dense() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    cache.append(0, key, value)
    actual_key, actual_value = cache.get(0)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)


def test_paged_attention_matches_dense_attention() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=2, head_dim=4)
    key = torch.randn(5, 2, 4)
    value = torch.randn(5, 2, 4)
    query = torch.randn(3, 2, 4)
    cache.append(0, key, value)

    actual = paged_attention(
        query,
        cache.key_cache,
        cache.value_cache,
        cache.block_tables[0],
        num_tokens=5,
        block_size=2,
        causal=True,
    )
    scores = torch.einsum("qhd,khd->qk", query, key) / 2.0
    mask = torch.arange(5) <= (torch.arange(3)[:, None] + 2)
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    expected = torch.einsum("qk,khd->qhd", weights, value)
    torch.testing.assert_close(actual, expected)


def test_paged_cache_clone_uses_copy_on_write() -> None:
    cache = PagedAttentionCache(num_blocks=8, block_size=4, num_heads=1, head_dim=2)
    prefix_key = torch.randn(3, 1, 2)
    prefix_value = torch.randn(3, 1, 2)
    cache.append(0, prefix_key, prefix_value)
    cache.clone_sequence(0, 1)
    shared_block = cache.block_tables[0][-1]

    new_key = torch.randn(1, 1, 2)
    new_value = torch.randn(1, 1, 2)
    cache.append(1, new_key, new_value)

    assert cache.block_tables[0][-1] == shared_block
    assert cache.block_tables[1][-1] != shared_block
    torch.testing.assert_close(cache.get(0)[0], prefix_key)
    torch.testing.assert_close(cache.get(1)[0], torch.cat((prefix_key, new_key)))


def test_paged_cache_reset_preserves_cloned_blocks() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=4, num_heads=1, head_dim=2)
    key = torch.randn(2, 1, 2)
    value = torch.randn(2, 1, 2)
    cache.append(0, key, value)
    cache.clone_sequence(0, 1)
    shared_block = cache.block_tables[0][0]
    cache.reset(0)
    assert shared_block in cache.allocator.allocated
    torch.testing.assert_close(cache.get(1)[0], key)


def test_paged_cache_reset_releases_unshared_blocks() -> None:
    cache = PagedAttentionCache(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
    cache.append(0, torch.randn(3, 1, 2), torch.randn(3, 1, 2))
    cache.reset(0)
    assert 0 not in cache.block_tables
    assert len(cache.allocator.free_blocks) == 4


def test_on_disk_kv_store_round_trip(tmp_path) -> None:
    store = OnDiskKVStore(tmp_path / "kv")
    key = torch.randn(4, 2, 8)
    value = torch.randn(4, 2, 8)
    store.save(1, key, value)
    actual_key, actual_value = store.load(1)
    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)
    store.delete(1)
    with pytest.raises(FileNotFoundError):
        store.load(1)


def test_sparse_indexer_is_causal_and_integrates_with_attention() -> None:
    hidden_size = 32
    num_heads = 4
    indexer = BlockSparseIndexer(
        hidden_size,
        num_heads,
        block_size=2,
        top_k=4,
        max_seq_len=16,
        causal=True,
    )
    attention = BlockSparseAttention(
        hidden_size,
        num_heads,
        block_size=2,
        num_kv_groups=2,
        top_k=4,
    )
    hidden_state = torch.randn(2, 7, hidden_size)
    block_indices = indexer(hidden_state)
    for query_block in range(block_indices.size(2)):
        assert (block_indices[:, :, query_block] <= query_block).all()
    output = attention(hidden_state, block_indices=block_indices)
    assert output.shape == hidden_state.shape
