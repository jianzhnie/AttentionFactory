"""FlashMLA-style inference interface simulation.

FlashMLA is a production CUDA kernel for Multi-head Latent Attention. This
module only simulates the cache-oriented interface: it stores latent KV
states and exposes prefill/decode-style methods without implementing GPU
kernel scheduling.
"""

from __future__ import annotations

import torch
from torch import nn

from .multi_head_latent_attention import MultiHeadLatentAttention


class FlashMLA(nn.Module):
    """Thin interface wrapper around ``MultiHeadLatentAttention``.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        q_latent_size: Latent dimension of the query branch.
        kv_latent_size: Latent dimension of the key/value branch.
        max_batch_size: Stored for interface parity with the real kernel's
            cache pre-allocation; not enforced.

    Note:
        ``prefill`` *records* latent KV states in ``latent_cache`` for
        inspection, but ``decode`` does not attend to the cache — this
        simulates the cache-oriented interface, not the cached computation.

    """

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
        output: torch.Tensor = self.mla(hidden_state)
        with torch.no_grad():
            latent = self.mla.kv_down_proj(hidden_state)
            self.latent_cache.append(latent.detach())
        return output

    def decode(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run one decode step using the same latent interface."""
        output: torch.Tensor = self.mla(hidden_state)
        return output

    def reset_cache(self) -> None:
        """Clear the simulated latent cache."""
        self.latent_cache.clear()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Run the default forward path, equivalent to ``prefill``."""
        return self.prefill(hidden_state)
