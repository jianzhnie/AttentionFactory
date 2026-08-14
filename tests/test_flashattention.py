"""Numerical correctness tests for the educational FlashAttention versions.

Every version is checked against the dense `reference_attention` for forward
outputs and against autograd through that reference for gradients, across
dtypes, causal masking, padding masks and cross-attention shapes.
"""

import math

import pytest
import torch

from attentionfactory.flashattention import fa1, fa2, fa3, fa4, flash_attention
from attentionfactory.flashattention.common import (
    FlashAttentionConfig,
    reference_attention,
)

VERSIONS = [fa1, fa2, fa3, fa4]
VERSION_IDS = ["fa1", "fa2", "fa3", "fa4"]

# Small blocks so every test exercises multi-tile online-softmax merges.
TILED_CONFIG = FlashAttentionConfig(block_size_q=16, block_size_kv=8)

TOLERANCES = {
    torch.float32: 2e-5,
    torch.float16: 1e-2,
    torch.bfloat16: 5e-2,
}


def make_inputs(batch, heads, q_len, kv_len, head_dim, value_dim, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape_q = (batch, heads, q_len, head_dim)
    shape_k = (batch, heads, kv_len, head_dim)
    shape_v = (batch, heads, kv_len, value_dim)
    q = torch.randn(shape_q, generator=generator, dtype=dtype)
    k = torch.randn(shape_k, generator=generator, dtype=dtype)
    v = torch.randn(shape_v, generator=generator, dtype=dtype)
    return q, k, v


def make_padding_mask(batch, kv_len, seed=1):
    """Random padding mask; batch row 0 is fully masked to test the edge case."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand(batch, kv_len, generator=generator) > 0.3
    mask[1:, 0] = True  # other rows keep at least one valid key
    mask[0] = False
    return mask


def autograd_reference(q, k, v, causal, mask, grad_out):
    q_ref, k_ref, v_ref = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = reference_attention(q_ref, k_ref, v_ref, causal=causal, key_padding_mask=mask)
    out.backward(grad_out)
    return out.detach(), q_ref.grad, k_ref.grad, v_ref.grad


CASES = [
    # (batch, heads, q_len, kv_len, head_dim, value_dim, causal, use_mask, dtype)
    (2, 3, 37, 37, 16, 24, False, False, torch.float32),  # self-attn, d_v != d_qk
    (2, 3, 40, 40, 16, 16, True, False, torch.float32),  # causal
    (2, 3, 37, 51, 16, 8, False, True, torch.float32),  # cross-attn + padding
    (2, 2, 33, 47, 32, 16, True, True, torch.float32),  # causal + padding
    (2, 2, 64, 64, 32, 32, True, False, torch.float16),  # fp16
    (2, 2, 20, 28, 16, 16, False, True, torch.bfloat16),  # bf16 + padding
]


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
@pytest.mark.parametrize(
    (
        "batch",
        "heads",
        "q_len",
        "kv_len",
        "head_dim",
        "value_dim",
        "causal",
        "use_mask",
        "dtype",
    ),
    CASES,
)
def test_forward_backward_match_reference(
    version, batch, heads, q_len, kv_len, head_dim, value_dim, causal, use_mask, dtype
):
    q, k, v = make_inputs(batch, heads, q_len, kv_len, head_dim, value_dim, dtype)
    mask = make_padding_mask(batch, kv_len) if use_mask else None
    generator = torch.Generator().manual_seed(2)
    grad_out = torch.randn(
        batch, heads, q_len, value_dim, dtype=dtype, generator=generator
    )
    tol = TOLERANCES[dtype]

    ref_out, ref_dq, ref_dk, ref_dv = autograd_reference(
        q, k, v, causal, mask, grad_out
    )

    fwd = version.forward(
        q, k, v, causal=causal, key_padding_mask=mask, config=TILED_CONFIG
    )
    assert (fwd.out.float() - ref_out.float()).abs().max().item() <= tol

    bwd = version.backward(
        q,
        k,
        v,
        grad_out,
        fwd,
        causal=causal,
        key_padding_mask=mask,
        config=TILED_CONFIG,
    )
    assert (bwd.dQ.float() - ref_dq.float()).abs().max().item() <= tol
    assert (bwd.dK.float() - ref_dk.float()).abs().max().item() <= tol
    assert (bwd.dV.float() - ref_dv.float()).abs().max().item() <= tol


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
def test_lse_matches_logsumexp(version):
    q, k, v = make_inputs(1, 1, 10, 12, 8, 8, torch.float32)
    scores = torch.einsum("bhid,bhjd->bhij", q / math.sqrt(8), k)
    ref_lse = torch.logsumexp(scores, dim=-1, keepdim=True)

    fwd = version.forward(q, k, v, config=TILED_CONFIG)
    assert (fwd.lse - ref_lse).abs().max().item() <= 1e-4


@pytest.mark.parametrize("version_name", VERSION_IDS)
def test_single_tile_matches_reference(version_name):
    """Block sizes larger than the sequence must degrade to plain attention."""
    q, k, v = make_inputs(1, 1, 5, 5, 8, 8, torch.float32)
    big_config = FlashAttentionConfig(block_size_q=1024, block_size_kv=1024)

    ref = reference_attention(q, k, v, causal=True)
    out = flash_attention(q, k, v, version=version_name, causal=True, config=big_config)
    assert (out - ref).abs().max().item() <= 2e-5


def test_fa3_fp8_forward_approximates_reference():
    q, k, v = make_inputs(1, 2, 32, 32, 16, 16, torch.float32)
    fp8_config = FlashAttentionConfig(block_size_q=16, block_size_kv=8, fp8=True)

    ref = reference_attention(q, k, v)
    fwd = fa3.forward(q, k, v, config=fp8_config)
    # Simulated E4M3 quantization is lossy but should stay in the ballpark.
    assert (fwd.out - ref).abs().max().item() <= 0.2


def test_fa3_fp8_backward_raises():
    q, k, v = make_inputs(1, 1, 16, 16, 8, 8, torch.float32)
    fp8_config = FlashAttentionConfig(fp8=True)
    fwd = fa3.forward(q, k, v, config=fp8_config)

    with pytest.raises(ValueError, match="FP8 backward"):
        fa3.backward(q, k, v, torch.ones_like(fwd.out), fwd, config=fp8_config)


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
def test_invalid_shapes_raise(version):
    q, k, v = make_inputs(1, 2, 8, 8, 16, 16, torch.float32)
    bad_k = torch.randn(1, 4, 8, 16)  # head mismatch

    with pytest.raises(ValueError, match="head"):
        version.forward(q, bad_k, v)

    bad_mask = torch.ones(3, 8, dtype=torch.bool)  # batch mismatch
    with pytest.raises(ValueError, match="key_padding_mask"):
        version.forward(q, k, v, key_padding_mask=bad_mask)


@pytest.mark.parametrize("version", VERSIONS, ids=VERSION_IDS)
def test_fully_masked_rows_produce_zero_output(version):
    q, k, v = make_inputs(1, 1, 6, 6, 8, 8, torch.float32)
    mask = torch.zeros(1, 6, dtype=torch.bool)

    fwd = version.forward(q, k, v, key_padding_mask=mask, config=TILED_CONFIG)
    assert torch.isfinite(fwd.out).all()
    assert (fwd.out == 0).all()
