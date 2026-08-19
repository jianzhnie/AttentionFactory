"""Tests for speculative decoding strategies and draft heads."""

import pytest
import torch

from llminfra import (
    DSFlashDecoder,
    DSparkScheduler,
    Eagle1Speculator,
    Eagle2Speculator,
    Eagle3Speculator,
    EagleSpeculator,
    MedusaHead,
    MultiTokenPredictionHead,
    NGramSpeculator,
    SpeculativeDecoder,
    medusa_loss,
)

HIDDEN_SIZE = 32


def _constant_model(vocab_size: int = 32):
    def model(input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)

    return model


def _fixed_token_model(token_id: int, vocab_size: int = 16):
    def model(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(input_ids.size(0), input_ids.size(1), vocab_size)
        logits[..., token_id] = 1.0
        return logits

    return model


def test_speculative_decoder_accepts_deterministic_tokens() -> None:
    model = _fixed_token_model(1)
    decoder = SpeculativeDecoder(model, model, num_speculative_tokens=3)
    output = decoder(torch.zeros(2, 4, dtype=torch.long))
    assert output.shape == (2, 7)
    assert (output[:, 4:] == 1).all()


def test_eagle_speculator_and_versioned_interfaces() -> None:
    target_model = _fixed_token_model(2)
    input_ids = torch.zeros(2, 4, dtype=torch.long)
    hidden_states = torch.randn(2, 4, HIDDEN_SIZE)

    for speculator_type in (
        EagleSpeculator,
        Eagle1Speculator,
        Eagle2Speculator,
        Eagle3Speculator,
    ):
        output = speculator_type(target_model, target_model, num_speculative_tokens=3)(
            input_ids, hidden_states
        )
        assert output.shape == (2, 7)
        assert (output[:, 4:] == 2).all()


def test_dsflash_uses_dynamic_scheduler() -> None:
    model = _fixed_token_model(1, vocab_size=8)
    input_ids = torch.zeros(1, 3, dtype=torch.long)
    decoder = DSFlashDecoder(model, model, DSparkScheduler((2,)))
    assert decoder(input_ids).size(1) >= input_ids.size(1)


def test_ngram_speculator_copies_prompt_continuation() -> None:
    prompt = torch.tensor([[1, 2, 1, 2, 0]])
    output = NGramSpeculator(
        _fixed_token_model(3, vocab_size=8),
        ngram_size=2,
        num_speculative_tokens=1,
    )(prompt)
    assert output.shape[0] == 1


def test_multi_token_prediction_returns_multiple_logits() -> None:
    head = MultiTokenPredictionHead(HIDDEN_SIZE, vocab_size=64, num_predictions=3)
    logits = head(torch.randn(2, 7, HIDDEN_SIZE))
    assert len(logits) == 3
    assert all(item.shape == (2, 7, 64) for item in logits)


def test_medusa_head_candidates_loss_and_gradient() -> None:
    head = MedusaHead(HIDDEN_SIZE, vocab_size=64, num_heads=3)
    hidden = torch.randn(2, 7, HIDDEN_SIZE, requires_grad=True)
    labels = torch.randint(0, 64, (2, 7))
    logits = head(hidden)
    assert logits.shape == (2, 7, 3, 64)
    candidate_ids, candidate_scores = head.generate_candidates(hidden, top_k=4)
    assert candidate_ids.shape == candidate_scores.shape == (2, 3, 4)
    loss = medusa_loss(head, hidden, labels, weight_decay=0.8)
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_speculative_decoder_validates_arguments() -> None:
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        SpeculativeDecoder(model, model, num_speculative_tokens=0)
    with pytest.raises(ValueError, match=">= 0"):
        SpeculativeDecoder(model, model, temperature=-0.5)


def test_speculative_decoder_accepts_short_prompt() -> None:
    model = _constant_model()
    output = SpeculativeDecoder(model, model, num_speculative_tokens=4)(
        torch.zeros(1, 2, dtype=torch.long)
    )
    assert output.size(1) >= 3


def test_residual_sampling_uses_probability_difference() -> None:
    model = _constant_model(vocab_size=4)
    decoder = SpeculativeDecoder(model, model, temperature=1.0)
    draft_logits = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    target_logits = torch.tensor([[0.0, 0.0, 20.0, 0.0]])
    samples = torch.cat(
        [decoder._sample_residual(draft_logits, target_logits) for _ in range(20)]
    )
    assert (samples == 2).all()


def test_eagle_speculator_validates_arguments() -> None:
    model = _constant_model()
    with pytest.raises(ValueError, match=">= 1"):
        EagleSpeculator(model, model, num_speculative_tokens=0)
