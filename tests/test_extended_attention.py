"""Tests for the extended attention-family implementations."""

import pytest
import torch

from attentionfactory import (
    BlockSparseAttention,
    GroupQueryAttention,
    LinearAttention,
    PagedAttentionCache,
    SlidingWindowAttention,
    paged_attention,
)

HIDDEN = 64
HEADS = 4
BATCH = 2
SEQ = 7


def make_input(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(BATCH, SEQ, HIDDEN, generator=generator)


def make_causal_mask():
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    return causal.expand(BATCH, 1, SEQ, SEQ)


@pytest.fixture()
def swa():
    return SlidingWindowAttention(
        HIDDEN,
        HEADS,
        window_size=2,
        num_kv_groups=2,
        dropout=0.0,
    )


def test_sliding_window_output_shape_and_gradient(swa):
    x = make_input().requires_grad_(True)
    out = swa(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sliding_window_weights_respect_window(swa):
    _, weights = swa(make_input(), return_attention_weights=True)
    assert weights.shape == (BATCH, HEADS, SEQ, SEQ)
    for i in range(SEQ):
        for j in range(SEQ):
            allowed = 0 <= i - j <= 2
            if not allowed:
                assert weights[0, 0, i, j].item() == 0.0


def test_sliding_window_large_window_equals_gqa():
    swa_full = SlidingWindowAttention(
        HIDDEN,
        HEADS,
        window_size=SEQ,
        num_kv_groups=2,
        dropout=0.0,
        causal=False,
    )
    gqa = GroupQueryAttention(HIDDEN, HEADS, num_kv_groups=2, dropout=0.0)
    swa_full.load_state_dict(gqa.state_dict())
    swa_full.eval()
    gqa.eval()

    torch.testing.assert_close(swa_full(make_input()), gqa(make_input()))


def test_sliding_window_rejects_bad_constructor_args():
    with pytest.raises(ValueError, match="window_size"):
        SlidingWindowAttention(HIDDEN, HEADS, window_size=0)
    with pytest.raises(ValueError, match="divisible"):
        SlidingWindowAttention(HIDDEN, HEADS, window_size=4, num_kv_groups=3)


@pytest.fixture()
def bsa():
    return BlockSparseAttention(
        HIDDEN,
        HEADS,
        block_size=2,
        num_kv_groups=2,
        top_k=1,
        dropout=0.0,
    )


def test_block_sparse_shape_and_gradient(bsa):
    x = make_input().requires_grad_(True)
    out = bsa(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_block_sparse_selects_only_local_block(bsa):
    _, weights = bsa(make_input(), return_attention_weights=True)
    # block_size=2 and top_k=1 selects the query's own block.
    for i in range(SEQ):
        for j in range(SEQ):
            if i // 2 != j // 2:
                assert weights[0, 0, i, j].item() == 0.0


def test_block_sparse_with_all_blocks_equals_causal_gqa():
    num_blocks = (SEQ + 1) // 2
    indices = torch.arange(num_blocks).expand(HEADS, num_blocks, num_blocks)
    sparse = BlockSparseAttention(
        HIDDEN,
        HEADS,
        block_size=2,
        num_kv_groups=2,
        top_k=num_blocks,
        dropout=0.0,
    )
    gqa = GroupQueryAttention(HIDDEN, HEADS, num_kv_groups=2, dropout=0.0)
    sparse.load_state_dict(gqa.state_dict())
    sparse.eval()
    gqa.eval()

    x = make_input()
    torch.testing.assert_close(
        sparse(x, block_indices=indices),
        gqa(x, attention_mask=make_causal_mask()),
    )


def test_block_sparse_rejects_out_of_range_block(bsa):
    indices = torch.full((HEADS, (SEQ + 1) // 2, 1), 99, dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range"):
        bsa(make_input(), block_indices=indices)


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
    x = make_input().requires_grad_(True)
    out = lin_attn(x)
    assert out.shape == (BATCH, SEQ, HIDDEN)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_linear_attention_is_deterministic(lin_attn):
    lin_attn.eval()
    x = make_input()
    torch.testing.assert_close(lin_attn(x), lin_attn(x))


def test_linear_attention_does_not_return_weights(lin_attn):
    with pytest.raises(ValueError, match="does not materialize"):
        lin_attn(make_input(), return_attention_weights=True)


def test_linear_attention_masked_row_is_finite(lin_attn):
    mask = torch.zeros(BATCH, 1, SEQ, dtype=torch.bool)
    out = lin_attn(make_input(), attention_mask=mask)
    assert torch.isfinite(out).all()


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
