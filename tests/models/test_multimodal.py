"""Tests for the multimodal fusion interfaces.

Covers ``VisionEncoderAdapter`` (projection shape and validation) and
``CrossAttentionFuser`` (late fusion with ``kv_len != q_len`` and dependence
on the vision input).
"""

import pytest
import torch

from llminfra.models.multimodal import (
    CrossAttentionFuser,
    MultimodalCausalLM,
    VisionEncoderAdapter,
)


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


def _make_multimodal_model(fusion_mode: str) -> MultimodalCausalLM:
    return MultimodalCausalLM(
        vocab_size=32,
        vision_dim=8,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        mrope_section=(2, 2, 4),
        fusion_mode=fusion_mode,
        max_seq_len=32,
    )


@pytest.mark.parametrize("fusion_mode", ["early", "cross_attention"])
def test_alignment_logits_ignore_padded_positions(fusion_mode):
    model = _make_multimodal_model(fusion_mode).eval()
    input_ids = torch.randint(0, 32, (1, 4))
    vision = torch.randn(1, 4, 8)
    grid = torch.tensor([[1, 2, 2]])
    text_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    vision_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    # Perturb only the padded positions; the alignment summaries must be
    # computed from valid tokens alone.
    perturbed_ids = (input_ids + 1) % 32
    perturbed_ids[0, :2] = input_ids[0, :2]
    perturbed_vision = vision.clone()
    perturbed_vision[0, 2:] += 10.0
    with torch.no_grad():
        base = model(
            input_ids,
            vision,
            grid,
            attention_mask=text_mask,
            vision_attention_mask=vision_mask,
        )
        changed = model(
            perturbed_ids,
            perturbed_vision,
            grid,
            attention_mask=text_mask,
            vision_attention_mask=vision_mask,
        )
    torch.testing.assert_close(base.alignment_logits, changed.alignment_logits)


@pytest.mark.parametrize("fusion_mode", ["early", "cross_attention"])
def test_multimodal_still_validates_image_grid_token_count(fusion_mode):
    model = _make_multimodal_model(fusion_mode)
    input_ids = torch.randint(0, 32, (1, 4))
    vision = torch.randn(1, 4, 8)
    bad_grid = torch.tensor([[1, 2, 3]])  # 6 tokens, but vision has 4
    with pytest.raises(ValueError, match="vision_features contains"):
        model(input_ids, vision, bad_grid)
