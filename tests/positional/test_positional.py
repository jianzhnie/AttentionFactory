"""Tests for the positional encoding modules.

Covers RoPE (and ``apply_rotary_pos_emb``), YaRN, dynamic NTK, partial RoPE,
position interpolation, LongRoPE, 2D position embedding, the ALiBi bias and
the ``get_positional_encoding`` factory.
"""

import pytest
import torch

from llminfra import (
    ALiBiBias,
    DynamicNTKRotaryEmbedding,
    LongRoPEScaledRotaryEmbedding,
    MultiModalRotaryPositionEmbedding,
    PartialRotaryPositionEmbedding,
    PositionInterpolation,
    RotaryPositionEmbedding,
    TwoDimensionalPositionEmbedding,
    YaRNParameters,
    YaRNScaledRotaryEmbedding,
    apply_rotary_pos_emb,
    get_positional_encoding,
)


def test_rotary_embedding_preserves_norm_and_shape():
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=16)
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_apply_rotary_pos_emb_rejects_odd_dim():
    x = torch.randn(1, 1, 3, 7)
    cos = torch.ones(1, 1, 3, 7)
    sin = torch.zeros(1, 1, 3, 7)
    with pytest.raises(ValueError, match="even"):
        apply_rotary_pos_emb(x, cos, sin)


def test_yarn_and_dynamic_ntk_are_finite():
    params = YaRNParameters(
        factor=4.0,
        original_max_position_embeddings=2048,
    )
    yarn = YaRNScaledRotaryEmbedding(8, max_seq_len=4096, params=params)
    ntk = DynamicNTKRotaryEmbedding(
        8,
        original_max_position_embeddings=2048,
        max_seq_len=8192,
    )
    x = torch.randn(1, 2, 64, 8)
    assert torch.isfinite(yarn(x)).all()
    assert torch.isfinite(ntk(x)).all()


def test_alibi_shape_and_causal_mask():
    alibi = ALiBiBias(num_heads=4, max_seq_len=8)
    x = torch.randn(1, 4, 6, 8)
    bias = alibi(x)
    assert bias.shape == (1, 4, 6, 6)
    assert torch.isinf(bias[0, 0, 0, 1])
    assert bias[0, 0, 5, 4] < 0
    assert bias[0, 0, 5, 4] < bias[0, 1, 5, 4]


def test_partial_rope_shape_and_norm_preserved():
    rope = PartialRotaryPositionEmbedding(
        dim=8, partial_rotary_factor=0.5, max_seq_len=16
    )
    x = torch.randn(2, 3, 7, 8)
    y = rope(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4)


def test_position_interpolation_is_finite():
    interpolate = PositionInterpolation(
        dim=8,
        original_max_position_embeddings=2048,
        max_seq_len=4096,
    )
    x = torch.randn(1, 1, 128, 8)
    assert torch.isfinite(interpolate(x)).all()


def test_longrope_and_2d_position_are_finite():
    factors = [1.0, 2.0, 3.0, 4.0]
    longrope = LongRoPEScaledRotaryEmbedding(
        dim=8,
        original_max_position_embeddings=4,
        max_seq_len=16,
        long_factor=factors,
        short_factor=factors,
    )
    x = torch.randn(1, 1, 8, 8)
    assert torch.isfinite(longrope(x)).all()
    two_d = TwoDimensionalPositionEmbedding(8, max_blocks=4, max_positions_per_block=4)
    y = torch.randn(1, 8, 8)
    assert torch.isfinite(two_d(y)).all()


def test_positional_encoding_factory():
    assert isinstance(get_positional_encoding("rope", dim=8), RotaryPositionEmbedding)
    assert isinstance(get_positional_encoding("alibi", dim=8, num_heads=4), ALiBiBias)
    with pytest.raises(ValueError, match="Unknown"):
        get_positional_encoding("unknown", dim=8)


def test_positional_factory_new_modes():
    assert isinstance(
        get_positional_encoding("partial_rope", dim=8, partial_rotary_factor=0.5),
        PartialRotaryPositionEmbedding,
    )
    assert isinstance(
        get_positional_encoding(
            "interpolation",
            dim=8,
            original_max_position_embeddings=2048,
        ),
        PositionInterpolation,
    )
    assert isinstance(
        get_positional_encoding("mrope", dim=8, mrope_section=(1, 1, 2)),
        MultiModalRotaryPositionEmbedding,
    )


def test_mrope_preserves_norm_and_uses_independent_axes():
    mrope = MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1, 2))
    x = torch.randn(2, 3, 5, 8)
    position_ids = torch.stack(
        [
            torch.arange(5).expand(2, -1),
            torch.zeros(2, 5, dtype=torch.long),
            torch.arange(5).flip(0).expand(2, -1),
        ]
    )
    output = mrope(x, position_ids)
    assert output.shape == x.shape
    torch.testing.assert_close(
        output.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-4
    )


def test_mrope_validates_sections_and_position_shape():
    with pytest.raises(ValueError, match="sum"):
        MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1))
    mrope = MultiModalRotaryPositionEmbedding(8, mrope_section=(1, 1, 2))
    with pytest.raises(ValueError, match="position_ids"):
        mrope(torch.randn(2, 5, 8), torch.zeros(3, 1, 5))
