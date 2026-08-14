"""Tests for the PyTorch-style flashattention API (functional + nn.Module)."""

import pytest
import torch

from attentionfactory import FlashAttention, flash_attention
from attentionfactory.flashattention import (
    ATTENTION_FN_REGISTRY,
    get_version_module,
    list_versions,
)
from attentionfactory.flashattention.common import FlashAttentionConfig

CONFIG = FlashAttentionConfig(block_size_q=8, block_size_kv=8)
VERSIONS = list_versions()


def make_qkv(seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 3, 20, 16, generator=generator)
    k = torch.randn(2, 3, 20, 16, generator=generator)
    v = torch.randn(2, 3, 20, 24, generator=generator)  # value_dim != head_dim
    return q, k, v


@pytest.mark.parametrize("version", VERSIONS)
def test_functional_matches_version_modules(version):
    q, k, v = make_qkv()
    direct = ATTENTION_FN_REGISTRY[version](q, k, v, causal=True, config=CONFIG)
    wrapped = flash_attention(q, k, v, version=version, causal=True, config=CONFIG)
    torch.testing.assert_close(wrapped, direct)


@pytest.mark.parametrize("version", VERSIONS)
def test_attention_is_differentiable(version):
    """`attention()` must route gradients through the tiled backward pass."""
    module = get_version_module(version)
    q, k, v = (t.requires_grad_(True) for t in make_qkv())
    grad_out = torch.randn(2, 3, 20, 24, generator=torch.Generator().manual_seed(1))

    out = ATTENTION_FN_REGISTRY[version](q, k, v, causal=True, config=CONFIG)
    assert out.grad_fn is not None
    out.backward(grad_out)

    # The autograd path must reproduce the manual forward/backward pair exactly.
    fwd = module.forward(q.detach(), k.detach(), v.detach(), causal=True, config=CONFIG)
    manual = module.backward(
        q.detach(), k.detach(), v.detach(), grad_out, fwd, causal=True, config=CONFIG
    )
    torch.testing.assert_close(out.detach(), fwd.out)
    torch.testing.assert_close(q.grad, manual.dQ)
    torch.testing.assert_close(k.grad, manual.dK)
    torch.testing.assert_close(v.grad, manual.dV)


@pytest.mark.parametrize("version", VERSIONS)
def test_attention_supports_padding_mask_with_autograd(version):
    q, k, v = (t.requires_grad_(True) for t in make_qkv())
    mask = torch.rand(2, 20, generator=torch.Generator().manual_seed(2)) > 0.3
    mask[:, 0] = True

    out = flash_attention(
        q, k, v, version=version, key_padding_mask=mask, config=CONFIG
    )
    out.sum().backward()
    for t in (q, k, v):
        assert t.grad is not None and torch.isfinite(t.grad).all()


def test_module_wrapper_matches_functional():
    q, k, v = make_qkv()
    attn = FlashAttention(version="fa3", causal=True, config=CONFIG)

    torch.testing.assert_close(
        attn(q, k, v),
        flash_attention(q, k, v, version="fa3", causal=True, config=CONFIG),
    )
    assert "fa3" in attn.extra_repr()
    assert "causal=True" in attn.extra_repr()


def test_module_wrapper_is_differentiable():
    q, k, v = (t.requires_grad_(True) for t in make_qkv())
    attn = FlashAttention()  # defaults to fa2

    out = attn(q, k, v)
    assert out.shape == (2, 3, 20, 24)
    out.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_unknown_version_raises():
    q, k, v = make_qkv()
    with pytest.raises(ValueError, match="Unknown FlashAttention version"):
        flash_attention(q, k, v, version="fa9")
    with pytest.raises(ValueError, match="Unknown FlashAttention version"):
        FlashAttention(version="fa9")
