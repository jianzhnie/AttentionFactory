"""Hybrid linear/full attention used by long-context architectures.

Models such as Qwen3-Next and Kimi Linear interleave a majority of linear
attention layers with a small number of full attention layers. This module
provides a simple routing wrapper around ``LinearAttention`` and
``GroupQueryAttention`` so a transformer can select the layer type by index.
"""

from __future__ import annotations

import torch

from .base_attention import BaseAttention, validate_attention_inputs
from .grouped_query_attention import GroupQueryAttention
from .linear_attention import LinearAttention


class HybridAttention(BaseAttention):
    """Interleave linear attention and full grouped-query attention.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        linear_interval: Number of linear layers per hybrid block.
        full_interval: Number of full attention layers per hybrid block.
        linear_feature_dim: Feature dimension used by linear attention.
        num_kv_groups: KV groups used by the full attention branch.
        dropout: Dropout probability for full attention weights.
        bias: Whether linear projections use biases.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        linear_interval: int = 3,
        full_interval: int = 1,
        linear_feature_dim: int | None = None,
        num_kv_groups: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        if linear_interval < 1 or full_interval < 1:
            raise ValueError("linear_interval and full_interval must be >= 1")
        self.linear_interval = int(linear_interval)
        self.full_interval = int(full_interval)
        self.linear_attention = LinearAttention(
            hidden_size,
            num_heads,
            feature_dim=linear_feature_dim,
            dropout=dropout,
            bias=bias,
        )
        self.full_attention = GroupQueryAttention(
            hidden_size,
            num_heads,
            num_kv_groups=num_heads if num_kv_groups is None else num_kv_groups,
            dropout=dropout,
            bias=bias,
        )

    def is_linear_layer(self, layer_index: int) -> bool:
        """Return whether ``layer_index`` should use linear attention."""
        period = self.linear_interval + self.full_interval
        return layer_index % period < self.linear_interval

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        layer_index: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Route ``layer_index`` to linear or full attention."""
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        if self.is_linear_layer(layer_index):
            return self.linear_attention(
                hidden_state,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )
        return self.full_attention(
            hidden_state,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )

    def extra_repr(self) -> str:
        return (
            f"{super().extra_repr()}, "
            f"linear_interval={self.linear_interval}, "
            f"full_interval={self.full_interval}"
        )
