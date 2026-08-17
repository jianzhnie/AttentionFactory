"""Attention with Linear Biases (ALiBi) positional bias."""

from __future__ import annotations

import torch

from .base import BasePositionalEncoding


class ALiBiBias(BasePositionalEncoding):
    """Attention with Linear Biases (ALiBi).

    ``forward`` returns a bias tensor of shape
    ``(1, num_heads, seq_len, seq_len)`` that can be added to attention
    scores. When ``causal`` is True, future positions receive ``-inf``.
    """

    def __init__(
        self,
        num_heads: int,
        max_seq_len: int,
        causal: bool = True,
        slope_base: float = 2.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        self.num_heads = int(num_heads)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.slope_base = float(slope_base)
        slopes = [
            slope_base ** (-8 * (head + 1) / self.num_heads)
            for head in range(self.num_heads)
        ]
        self.register_buffer("slopes", torch.tensor(slopes), persistent=False)

    def forward(self, x: torch.Tensor | int | None = None) -> torch.Tensor:
        """Return ALiBi bias for the sequence length of ``x`` if provided."""
        if isinstance(x, int):
            seq_len = x
        else:
            seq_len = self.max_seq_len if x is None else x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        # Derive the device from the registered buffer so the module works
        # after .to("cuda") / .to("mps") moves.
        device = self.slopes.device
        q_pos = torch.arange(seq_len, device=device).view(-1, 1)
        k_pos = torch.arange(seq_len, device=device).view(1, -1)
        distance = q_pos - k_pos
        bias = -self.slopes[:, None, None] * distance.abs().unsqueeze(0)
        if self.causal:
            future = q_pos < k_pos
            bias = bias.masked_fill(future[None], float("-inf"))
        return bias.unsqueeze(0)
