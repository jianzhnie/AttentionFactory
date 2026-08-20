"""Reusable output heads for classification, reward, and embedding models."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

PoolingMode = Literal["first", "last", "mean"]


def pool_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    pooling: PoolingMode = "last",
) -> torch.Tensor:
    """Pool token states while respecting a padding mask.

    Args:
        hidden_state: Tensor shaped ``(batch, seq_len, hidden_size)``.
        attention_mask: Optional boolean/integer mask shaped ``(batch, seq)``;
            nonzero positions are valid.
        pooling: ``"first"``, ``"last"``, or masked ``"mean"`` pooling.

    Returns:
        Tensor shaped ``(batch, hidden_size)``.
    """
    if hidden_state.dim() != 3:
        raise ValueError("hidden_state must have shape (batch, seq, hidden)")
    batch_size, seq_len, _ = hidden_state.shape
    if seq_len < 1:
        raise ValueError("hidden_state sequence length must be >= 1")
    if pooling not in {"first", "last", "mean"}:
        raise ValueError("pooling must be 'first', 'last', or 'mean'")

    if attention_mask is None:
        mask = torch.ones(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=hidden_state.device,
        )
    else:
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError("attention_mask must have shape (batch, seq_len)")
        mask = attention_mask.to(device=hidden_state.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("every batch row must contain at least one valid token")

    if pooling == "mean":
        weights = mask.to(hidden_state.dtype).unsqueeze(-1)
        return (hidden_state * weights).sum(dim=1) / weights.sum(dim=1)

    positions = torch.arange(seq_len, device=hidden_state.device)[None]
    if pooling == "first":
        selected = torch.where(mask, positions, seq_len).min(dim=1).values
    else:
        selected = torch.where(mask, positions, -1).max(dim=1).values
    batch = torch.arange(batch_size, device=hidden_state.device)
    return hidden_state[batch, selected]


class SequenceClassificationHead(nn.Module):
    """Pool a sequence and predict one of ``num_labels`` classes."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        pooling: PoolingMode = "first",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_labels < 1:
            raise ValueError("hidden_size and num_labels must be >= 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, num_labels, bias=bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits shaped ``(batch, num_labels)``."""
        pooled = pool_hidden_state(hidden_state, attention_mask, self.pooling)
        logits: torch.Tensor = self.projection(self.dropout(pooled))
        return logits


class TokenClassificationHead(nn.Module):
    """Predict a label independently for every sequence position."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_labels < 1:
            raise ValueError("hidden_size and num_labels must be >= 1")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.hidden_size = int(hidden_size)
        self.num_labels = int(num_labels)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, num_labels, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return logits shaped ``(batch, seq_len, num_labels)``."""
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_size:
            raise ValueError(
                "hidden_state must have shape (batch, seq_len, hidden_size)"
            )
        logits: torch.Tensor = self.projection(self.dropout(hidden_state))
        return logits


class RewardModelHead(nn.Module):
    """Produce one scalar reward for each input sequence."""

    def __init__(
        self,
        hidden_size: int,
        pooling: PoolingMode = "last",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be >= 1")
        self.hidden_size = int(hidden_size)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return rewards shaped ``(batch,)``."""
        pooled = pool_hidden_state(hidden_state, attention_mask, self.pooling)
        reward: torch.Tensor = self.projection(self.dropout(pooled)).squeeze(-1)
        return reward


class EmbeddingHead(nn.Module):
    """Pool, project, and optionally L2-normalize sequence embeddings."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int | None = None,
        pooling: PoolingMode = "mean",
        normalize: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        output_size = hidden_size if output_size is None else output_size
        if hidden_size < 1 or output_size < 1:
            raise ValueError("hidden_size and output_size must be >= 1")
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.pooling = pooling
        self.normalize = bool(normalize)
        self.projection = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sentence embeddings shaped ``(batch, output_size)``."""
        pooled = pool_hidden_state(hidden_state, attention_mask, self.pooling)
        output = self.projection(pooled)
        return F.normalize(output, dim=-1) if self.normalize else output


__all__ = [
    "EmbeddingHead",
    "PoolingMode",
    "RewardModelHead",
    "SequenceClassificationHead",
    "TokenClassificationHead",
    "pool_hidden_state",
]
