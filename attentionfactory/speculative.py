"""Educational speculative decoding interface.

This module simulates draft-target verification without requiring trained
draft weights. It is an interface-level implementation, not a production
sampler.
"""

from __future__ import annotations

import torch
from torch import nn


class SpeculativeDecoder(nn.Module):
    """Draft-then-verify speculative decoding loop.

    Args:
        draft_model: Callable mapping ``(batch, seq)`` ids to logits.
        target_model: Callable mapping ``(batch, seq)`` ids to logits.
        num_speculative_tokens: Number of draft tokens generated per block.
        temperature: Sampling temperature; 0 selects argmax.
    """

    def __init__(
        self,
        draft_model,
        target_model,
        num_speculative_tokens: int = 4,
        temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.temperature = float(temperature)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate one verified speculative block.

        Returns:
            Input ids concatenated with accepted tokens.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq)")
        draft_logits = self.draft_model(input_ids)
        draft_tokens = self._sample(draft_logits[:, -self.num_speculative_tokens :])

        target_input = torch.cat([input_ids, draft_tokens], dim=-1)
        target_logits = self.target_model(target_input)
        accepted: list[torch.Tensor] = []
        start = input_ids.size(1)
        for step in range(self.num_speculative_tokens):
            target_next = self._sample(
                target_logits[:, start + step - 1 : start + step]
            )
            if (
                self.temperature == 0.0
                and (draft_tokens[:, step] != target_next[:, 0]).any()
            ):
                accepted.append(target_next)
                break
            accepted.append(target_next)
        return torch.cat([input_ids, *accepted], dim=-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.size(-1))
        sampled = torch.multinomial(flat, 1)
        return sampled.reshape(probabilities.shape[:-1])

    def extra_repr(self) -> str:
        return (
            f"num_speculative_tokens={self.num_speculative_tokens}, "
            f"temperature={self.temperature}"
        )
