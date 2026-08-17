"""FlashMLA-style inference interface simulation.

FlashMLA is a production CUDA kernel for Multi-head Latent Attention. This
module only simulates the cache-oriented interface: it stores latent KV
states and exposes prefill/decode-style methods without implementing GPU
kernel scheduling.
"""

from __future__ import annotations

import torch
from torch import nn

from .mla import MultiHeadLatentAttention


class FlashMLA(nn.Module):
    """Thin interface wrapper around ``MultiHeadLatentAttention``."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_latent_size: int,
        kv_latent_size: int,
        max_batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.mla = MultiHeadLatentAttention(
            hidden_size,
            num_heads,
            q_latent_size=q_latent_size,
            kv_latent_size=kv_latent_size,
        )
        self.max_batch_size = int(max_batch_size)
        self.latent_cache: list[torch.Tensor] = []

    def prefill(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run the attention layer and store latent KV states."""
        output = self.mla(hidden_state)
        with torch.no_grad():
            latent = self.mla.kv_down_proj(hidden_state)
            self.latent_cache.append(latent.detach())
        return output

    def decode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run one decode step using the same latent interface."""
        return self.mla(hidden_state)

    def reset_cache(self) -> None:
        """Clear the simulated latent cache."""
        self.latent_cache.clear()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Default forward behavior equals ``prefill``."""
        return self.prefill(hidden_state)
