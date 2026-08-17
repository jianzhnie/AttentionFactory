"""Tests for the shared BaseAttention helpers in base.py."""

import pytest
import torch

from llminfra import MultiHeadAttention
from llminfra.attention.base import validate_attention_inputs


@pytest.fixture()
def module():
    return MultiHeadAttention(hidden_size=32, num_heads=4, dropout=0.0)


def test_split_combine_head_roundtrip(module):
    x = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(0))

    heads = module.split_head(x)
    assert heads.shape == (2, 4, 5, 8)

    combined = module.combine_head(heads)
    assert combined.shape == x.shape
    torch.testing.assert_close(combined, x)


def test_compute_attention_weights_are_normalized(module):
    scores = torch.randn(2, 4, 5, 5, generator=torch.Generator().manual_seed(0))
    weights = module.compute_attention_weights(scores)

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 4, 5))


def test_compute_attention_weights_respects_mask(module):
    scores = torch.randn(1, 1, 3, 3, generator=torch.Generator().manual_seed(0))
    mask = torch.tensor([[[[1, 1, 0]]]])  # (batch, 1, 1, seq) broadcast mask

    weights = module.compute_attention_weights(scores, mask)

    assert weights[..., 2].abs().max().item() == 0.0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 3))


def test_validate_attention_inputs_ok():
    batch, seq_len = validate_attention_inputs(torch.randn(2, 5, 8), None, num_heads=4)
    assert (batch, seq_len) == (2, 5)


def test_validate_attention_inputs_rejects_2d_hidden_state():
    with pytest.raises(ValueError, match="3D"):
        validate_attention_inputs(torch.randn(5, 8), None, num_heads=4)


def test_validate_attention_inputs_rejects_2d_mask():
    with pytest.raises(ValueError, match="3D or 4D"):
        validate_attention_inputs(torch.randn(2, 5, 8), torch.ones(2, 5), num_heads=4)


def test_validate_attention_inputs_rejects_mask_batch_mismatch():
    with pytest.raises(ValueError, match="batch size"):
        validate_attention_inputs(
            torch.randn(2, 5, 8), torch.ones(3, 1, 1, 5), num_heads=4
        )


def test_extra_repr_reports_configuration(module):
    assert "hidden_size=32" in module.extra_repr()
    assert "num_heads=4" in module.extra_repr()
