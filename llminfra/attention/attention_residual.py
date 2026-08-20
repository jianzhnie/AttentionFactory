"""Attention Residual (AttnRes) teaching module.

Kimi K3 uses Attention Residuals to strengthen the path from attention
outputs across layers. This module provides a simple learned per-dimension
residual gate that can be plugged into a transformer block.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionResidual(nn.Module):
    """Learned residual connection specialized for attention outputs.

    Args:
        hidden_size: Feature dimension of both inputs.
        init_scale: Initial value of the per-dimension residual gate.
    """

    def __init__(self, hidden_size: int, init_scale: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.init_scale = float(init_scale)
        self.weight = nn.Parameter(torch.full((hidden_size,), init_scale))

    def forward(
        self, hidden_state: torch.Tensor, attention_output: torch.Tensor
    ) -> torch.Tensor:
        """Return ``hidden_state + weight * attention_output``."""
        return hidden_state + self.weight * attention_output

    def extra_repr(self) -> str:
        """Return a string representation of the module's extra information."""
        return f"hidden_size={self.hidden_size}, init_scale={self.init_scale}"
