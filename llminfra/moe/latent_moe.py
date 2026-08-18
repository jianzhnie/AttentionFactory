"""Educational LatentMoE used by Nemotron-3-style hybrid models."""

from __future__ import annotations

import torch
from torch import nn

from .mixture_of_experts import MixtureOfExperts


class LatentMoE(nn.Module):
    """MoE that routes and computes experts in a low-dimensional latent space.

    Args:
        hidden_size: Input/output feature dimension.
        latent_size: Latent dimension used for routing and expert computation.
        num_experts: Number of routed experts.
        intermediate_size: Expert FFN intermediate dimension.
        top_k: Number of experts selected per token.
        residual: If True, add the input to the projected output.
    """

    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        num_experts: int,
        intermediate_size: int,
        top_k: int = 2,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.num_experts = int(num_experts)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.residual = bool(residual)
        self.down_proj = nn.Linear(hidden_size, latent_size)
        self.up_proj = nn.Linear(latent_size, hidden_size)
        self.moe = MixtureOfExperts(
            hidden_size=latent_size,
            num_experts=num_experts,
            intermediate_size=intermediate_size,
            top_k=top_k,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to latent space, route experts, then project back."""
        latent = self.down_proj(x)
        routed = self.moe(latent)
        output = self.up_proj(routed)
        if self.residual:
            output = output + x
        return output

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, latent_size={self.latent_size}, "
            f"num_experts={self.num_experts}, top_k={self.top_k}, "
            f"residual={self.residual}"
        )
