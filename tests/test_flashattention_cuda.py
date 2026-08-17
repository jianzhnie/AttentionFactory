"""Long-sequence CUDA tests for the educational FlashAttention versions.

These exercise an 8192-long key/value sequence on a GPU to smoke out
accumulation issues that short CPU tests cannot reach. The whole module is
skipped when CUDA is unavailable.
"""

import pytest
import torch
from helpers import make_key_padding_mask, make_qkv, with_grad

from attentionfactory.flashattention import (
    ATTENTION_FN_REGISTRY,
    FlashAttentionConfig,
    get_version_module,
    reference_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA long tests require a GPU"
)

DEVICE = torch.device("cuda")
CONFIG = FlashAttentionConfig(block_size_q=128, block_size_kv=128, num_stages=2)
VERSIONS = ("fa1", "fa2", "fa3", "fa4")

# Short queries over a long key/value sequence.
Q_LEN, KV_LEN, HEADS, DIM = 64, 8192, 8, 64


def _inputs(causal: bool):
    q, k, v = make_qkv(1, HEADS, Q_LEN, KV_LEN, DIM, DIM, seed=7, device=DEVICE)
    mask = None
    if not causal:
        mask = make_key_padding_mask(1, KV_LEN, fully_masked_row=False, device=DEVICE)
    return q, k, v, mask


@pytest.mark.parametrize("version_name", VERSIONS)
@pytest.mark.parametrize("causal", [False, True])
def test_long_kv_forward_and_backward(version_name, causal):
    module = get_version_module(version_name)
    attention_fn = ATTENTION_FN_REGISTRY[version_name]
    q, k, v, mask = _inputs(causal)

    forward_result = module.forward(
        q, k, v, causal=causal, key_padding_mask=mask, config=CONFIG
    )
    reference_out = reference_attention(q, k, v, causal=causal, key_padding_mask=mask)

    assert forward_result.out.shape == q.shape
    assert torch.isfinite(forward_result.out).all()
    torch.testing.assert_close(forward_result.out, reference_out, atol=1e-5, rtol=1e-4)

    # Autograd path: gradients exist, have input shapes and stay finite.
    q_auto, k_auto, v_auto = with_grad(q, k, v)
    auto_out = attention_fn(
        q_auto, k_auto, v_auto, causal=causal, key_padding_mask=mask, config=CONFIG
    )
    auto_grads = torch.autograd.grad(auto_out.sum(), (q_auto, k_auto, v_auto))
    for tensor, grad in zip((q_auto, k_auto, v_auto), auto_grads, strict=True):
        assert grad.shape == tensor.shape
        assert torch.isfinite(grad).all()

    # Manual backward path: same checks against the saved forward result.
    manual = module.backward(
        q,
        k,
        v,
        torch.ones_like(forward_result.out),
        forward_result,
        causal=causal,
        key_padding_mask=mask,
        config=CONFIG,
    )
    for tensor, grad in zip(
        (q, k, v), (manual.grad_q, manual.grad_k, manual.grad_v), strict=True
    ):
        assert grad.shape == tensor.shape
        assert torch.isfinite(grad).all()


def test_fa3_fp8_long_forward():
    q, k, v, mask = _inputs(causal=False)
    fp8_config = FlashAttentionConfig(
        block_size_q=128, block_size_kv=128, num_stages=2, fp8=True
    )

    forward_result = get_version_module("fa3").forward(
        q, k, v, causal=False, key_padding_mask=mask, config=fp8_config
    )
    reference_out = reference_attention(q, k, v, key_padding_mask=mask)

    assert forward_result.out.shape == q.shape
    assert torch.isfinite(forward_result.out).all()
    torch.testing.assert_close(forward_result.out, reference_out, atol=2e-1, rtol=2e-1)
    assert forward_result.saved_state["fp8_enabled"]
