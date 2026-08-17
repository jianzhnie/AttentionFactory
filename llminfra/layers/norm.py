"""Normalization layers used by mainstream transformer architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization used by Llama/Qwen/Mistral-style models.

    Args:
        hidden_size: Feature dimension.
        eps: Small value added to the variance for numerical stability.
        elementwise_affine: Whether to use a learnable per-dimension weight.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
