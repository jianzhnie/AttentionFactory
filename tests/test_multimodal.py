"""Tests for the multimodal fusion interfaces.

Covers ``VisionEncoderAdapter`` (projection shape and validation) and
``CrossAttentionFuser`` (late fusion with ``kv_len != q_len`` and dependence
on the vision input).
"""

import pytest
import torch

from llminfra.models.multimodal import CrossAttentionFuser, VisionEncoderAdapter


def test_vision_encoder_adapter_shape():
    adapter = VisionEncoderAdapter(vision_dim=24, hidden_size=16)
    vision_features = torch.randn(2, 10, 24)
    projected = adapter(vision_features)
    assert projected.shape == (2, 10, 16)


def test_vision_encoder_adapter_validates_input():
    adapter = VisionEncoderAdapter(vision_dim=24, hidden_size=16)
    with pytest.raises(ValueError, match="3D"):
        adapter(torch.randn(10, 24))
    with pytest.raises(ValueError, match="last dim"):
        adapter(torch.randn(2, 10, 8))


def test_cross_attention_fuser_shape_and_weights():
    fuser = CrossAttentionFuser(hidden_size=16, num_heads=2)
    text_state = torch.randn(2, 5, 16)
    vision_state = torch.randn(2, 12, 16)
    fused, weights = fuser(text_state, vision_state, return_attention_weights=True)
    assert fused.shape == (2, 5, 16)
    assert weights.shape == (2, 2, 5, 12)
    # Attention weights over the vision tokens are normalized per text token.
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 5))


def test_cross_attention_fuser_depends_on_vision():
    fuser = CrossAttentionFuser(hidden_size=16, num_heads=2)
    fuser.eval()
    text_state = torch.randn(1, 5, 16)
    vision_state = torch.randn(1, 12, 16)
    perturbed_vision = vision_state.clone()
    perturbed_vision[0, 4] += 1.0
    with torch.no_grad():
        base = fuser(text_state, vision_state)
        changed = fuser(text_state, perturbed_vision)
    assert not torch.allclose(base, changed)


def test_adapter_fuser_pipeline_and_gradient():
    adapter = VisionEncoderAdapter(vision_dim=24, hidden_size=16)
    fuser = CrossAttentionFuser(hidden_size=16, num_heads=2)
    vision_features = torch.randn(2, 12, 24)
    text_state = torch.randn(2, 5, 16, requires_grad=True)
    fused = fuser(text_state, adapter(vision_features))
    assert fused.shape == (2, 5, 16)
    fused.sum().backward()
    assert text_state.grad is not None
    assert torch.isfinite(text_state.grad).all()
    assert adapter.proj.weight.grad is not None
