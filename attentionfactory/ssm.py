"""Simplified state-space model layer for hybrid Mamba-style blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class Mamba2Layer(nn.Module):
    """Teaching-level diagonal SSM layer.

    This is not a faithful Mamba-2 CUDA implementation. It preserves the
    fixed-size recurrent state structure used by Mamba-2 hybrid models.
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d_state = int(d_state)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.b_proj = nn.Linear(hidden_size, d_state)
        self.c_proj = nn.Linear(hidden_size, d_state)
        self.dt_proj = nn.Linear(hidden_size, 1)
        self.out_proj = nn.Linear(1, hidden_size)
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), d_state)
        self.A = nn.Parameter(-torch.exp(log_dt))
        self.D = nn.Parameter(torch.ones(1))

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the recurrent SSM over the sequence dimension."""
        batch_size, seq_len, _ = x.size()
        u = self.in_proj(x)
        b = self.b_proj(x)
        c = self.c_proj(x)
        dt = torch.sigmoid(self.dt_proj(x))[..., 0]
        a_bar = torch.exp(self.A[None, None, :] * dt[..., None])

        if state is None:
            state = torch.zeros(
                batch_size, self.d_state, device=x.device, dtype=x.dtype
            )
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            state = a_bar[:, step] * state + b[:, step] * u[:, step].sum(
                dim=-1, keepdim=True
            )
            y = torch.einsum("bd,bd->b", c[:, step], state)
            y = y + self.D * u[:, step].sum(dim=-1)
            outputs.append(y)
        output = torch.stack(outputs, dim=1).unsqueeze(-1)
        return self.out_proj(output)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, d_state={self.d_state}"
