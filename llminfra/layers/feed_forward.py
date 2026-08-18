"""Feed-forward network modules used by transformer and MoE blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network used by Llama/Qwen/DeepSeek-style models.

    The module computes ``down(silu(gate(x)) * up(x))``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def _init_weights(self) -> None:
        for module in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


class FeedForward(nn.Module):
    """Simple two-layer feed-forward network with configurable activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        if activation not in {"gelu", "relu", "silu"}:
            raise ValueError(f"Unknown activation: {activation}")
        self.activation_name = activation
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self._activation(self.w1(x)))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "gelu":
            return F.gelu(x)
        if self.activation_name == "relu":
            return F.relu(x)
        return F.silu(x)

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        if self.w1.bias is not None:
            nn.init.zeros_(self.w1.bias)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)
