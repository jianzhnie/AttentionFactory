"""Multi-Token Prediction (MTP) head used by DeepSeek/Nemotron-style models."""

from __future__ import annotations

import torch
from torch import nn


class MultiTokenPredictionHead(nn.Module):
    """Predict the next ``num_predictions`` tokens from one hidden state.

    Args:
        hidden_size: Hidden state dimension.
        vocab_size: Vocabulary size.
        num_predictions: Number of future tokens predicted per step.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_predictions: int = 2,
    ) -> None:
        super().__init__()
        if num_predictions < 1:
            raise ValueError("num_predictions must be >= 1")
        self.hidden_size = int(hidden_size)
        self.vocab_size = int(vocab_size)
        self.num_predictions = int(num_predictions)
        self.heads = nn.ModuleList(
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_predictions)
        )

    def forward(self, hidden_state: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of logit tensors, one per prediction head."""
        return [head(hidden_state) for head in self.heads]

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, vocab_size={self.vocab_size}, "
            f"num_predictions={self.num_predictions}"
        )
