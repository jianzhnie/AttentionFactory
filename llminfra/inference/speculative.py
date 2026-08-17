"""Educational speculative decoding interface.

This module simulates draft-target verification without requiring trained
draft weights. It is an interface-level implementation, not a production
sampler.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class SpeculativeDecoder(nn.Module):
    """Draft-then-verify speculative decoding loop.

    Args:
        draft_model: Callable mapping ``(batch, seq)`` ids to logits.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        num_speculative_tokens: Number of draft tokens generated per block.
        temperature: Sampling temperature; 0 selects argmax. With
            ``temperature > 0`` a rejection sampler verifies
            drafts: a draft token is accepted with probability
            ``min(1, p_target / p_draft)`` and otherwise replaced by a fresh
            sample from ``norm(max(0, p_target - p_draft))``.
        append_bonus_token: When True, sample one extra "bonus" token from
            the target logits after a fully accepted draft block (the
            standard speculative decoding behavior). Defaults to False to
            preserve the original block-size-only output contract.
    """

    def __init__(
        self,
        draft_model: Callable[[torch.Tensor], torch.Tensor],
        target_model: Callable[[torch.Tensor], torch.Tensor],
        num_speculative_tokens: int = 4,
        temperature: float = 0.0,
        append_bonus_token: bool = False,
    ) -> None:
        super().__init__()
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
        self.draft_model = draft_model
        self.target_model = target_model
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.temperature = float(temperature)
        self.append_bonus_token = bool(append_bonus_token)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate one speculative block.

        Teaching simplifications: draft tokens are read off the last
        ``num_speculative_tokens`` positions of the draft logits (no
        autoregressive draft rollout), and acceptance is decided batch-wide
        (one rejecting row stops the whole batch). The probability correction
        itself follows standard speculative sampling.

        Returns:
            Input ids concatenated with accepted tokens (plus one bonus
            token when ``append_bonus_token`` is enabled and every draft
            was accepted).
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        if input_ids.size(1) < self.num_speculative_tokens:
            raise ValueError(
                "input sequence must be at least num_speculative_tokens long"
            )
        draft_logits = self.draft_model(input_ids)
        draft_window = draft_logits[:, -self.num_speculative_tokens :]
        draft_tokens = self._sample(draft_window)

        target_input = torch.cat([input_ids, draft_tokens], dim=-1)
        target_logits = self.target_model(target_input)
        accepted: list[torch.Tensor] = []
        start = input_ids.size(1)
        for step in range(self.num_speculative_tokens):
            step_logits = target_logits[:, start + step - 1 : start + step]
            if self.temperature == 0.0:
                target_next = self._sample(step_logits)
                accepted.append(target_next)
                if (draft_tokens[:, step] != target_next[:, 0]).any():
                    break
            else:
                draft_token = draft_tokens[:, step : step + 1]
                if self._accept_draft(
                    draft_window[:, step], step_logits[:, 0], draft_token
                ):
                    accepted.append(draft_token)
                else:
                    accepted.append(
                        self._sample_residual(
                            draft_window[:, step], step_logits[:, 0]
                        )
                    )
                    break
        else:
            if self.append_bonus_token:
                accepted.append(self._sample(target_logits[:, -1:]))
        return torch.cat([input_ids, *accepted], dim=-1)

    def _accept_draft(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        draft_token: torch.Tensor,
    ) -> bool:
        """Batch-wide rejection-sampling decision for one draft position.

        Accepts the draft token with probability ``min(1, p_target/p_draft)``
        per row, where both distributions are temperature-scaled softmaxes
        evaluated at the draft token. Returns True only if every batch row
        accepts.
        """
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        p_draft_token = p_draft.gather(-1, draft_token)
        p_target_token = p_target.gather(-1, draft_token)
        ratio = p_target_token / p_draft_token.clamp_min(1e-12)
        accept_prob = torch.clamp(ratio, max=1.0)
        return bool((torch.rand_like(accept_prob) < accept_prob).all())

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.size(-1))
        sampled = torch.multinomial(flat, 1)
        return sampled.reshape(probabilities.shape[:-1])

    def _sample_residual(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the correction distribution after rejecting a draft.

        Standard speculative decoding samples from the normalized positive
        difference ``max(0, p_target - p_draft)``. Numerical or identical-
        distribution edge cases can leave zero total mass; those rows safely
        fall back to the target distribution.
        """
        p_draft = torch.softmax(draft_logits / self.temperature, dim=-1)
        p_target = torch.softmax(target_logits / self.temperature, dim=-1)
        residual = (p_target - p_draft).clamp_min(0.0)
        total = residual.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            total > 1e-12,
            residual / total.clamp_min(1e-12),
            p_target,
        )
        return torch.multinomial(probabilities, 1)

    def extra_repr(self) -> str:
        return (
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"temperature={self.temperature}, "
            f"append_bonus_token={self.append_bonus_token}"
        )


class EagleSpeculator(nn.Module):
    """Eagle-style speculative decoder using hidden states for drafting.

    ``draft_head`` maps hidden states to next-token logits; ``target_model``
    maps token ids to logits for verification. This is an interface-level
    simulation, not a trained Eagle model; like `SpeculativeDecoder`, draft
    tokens are read off the last positions of the draft logits instead of an
    autoregressive rollout.

    Args:
        draft_head: Callable mapping hidden states to logits.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        num_speculative_tokens: Number of draft tokens generated per block.
        append_bonus_token: When True, take one extra "bonus" token (argmax
            of the final target logits) after a fully accepted draft block.
            Defaults to False to preserve the original output contract.
    """

    def __init__(
        self,
        draft_head: Callable[[torch.Tensor], torch.Tensor],
        target_model: Callable[[torch.Tensor], torch.Tensor],
        num_speculative_tokens: int = 4,
        append_bonus_token: bool = False,
    ) -> None:
        super().__init__()
        if num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")
        self.draft_head = draft_head
        self.target_model = target_model
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.append_bonus_token = bool(append_bonus_token)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Draft from hidden states and verify with the target model."""
        if input_ids.size(1) < self.num_speculative_tokens:
            raise ValueError(
                "input sequence must be at least num_speculative_tokens long"
            )
        draft_logits = self.draft_head(hidden_states)
        draft_tokens = torch.argmax(
            draft_logits[:, -self.num_speculative_tokens :], dim=-1
        )
        target_input = torch.cat([input_ids, draft_tokens], dim=-1)
        target_logits = self.target_model(target_input)
        accepted: list[torch.Tensor] = []
        start = input_ids.size(1)
        for step in range(self.num_speculative_tokens):
            target_next = torch.argmax(
                target_logits[:, start + step - 1 : start + step], dim=-1
            )
            accepted.append(target_next)
            if (draft_tokens[:, step] != target_next[:, 0]).any():
                break
        else:
            if self.append_bonus_token:
                accepted.append(torch.argmax(target_logits[:, -1:], dim=-1))
        return torch.cat([input_ids, *accepted], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"append_bonus_token={self.append_bonus_token}"
        )
