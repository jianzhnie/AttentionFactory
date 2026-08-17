"""Hybrid SSM/attention block that alternates sublayers by a pattern.

This is a teaching-level generalization of the hybrid layouts used by
Zamba (Mamba blocks interleaved with shared attention) and Qwen3-Next
(linear/SSM layers with occasional full attention): a ``pattern`` such as
``"ssm:ssm:attn"`` decides which sublayer type runs at each position.
Unlike the real architectures, the sublayers are composed sequentially
without residual connections or normalization, to keep the routing logic
front and center.
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention.mha import MultiHeadAttention
from .ssm import Mamba2Layer

VALID_TOKENS = ("ssm", "attn")


class HybridSSMBlock(nn.Module):
    """Alternate Mamba2 SSM sublayers and attention sublayers by a pattern.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads, used when ``attention`` is
            ``None`` and a fresh ``MultiHeadAttention`` is created per
            ``"attn"`` token.
        pattern: Sublayer layout, either a ``":"``-separated string such as
            ``"ssm:ssm:attn"`` or a list of tokens. Each token is ``"ssm"``
            (a fresh ``Mamba2Layer``) or ``"attn"`` (an attention module).
        attention: Optional attention module used for every ``"attn"``
            token (weights are shared across positions). Defaults to one
            fresh ``MultiHeadAttention`` per ``"attn"`` token.
        d_state: SSM state dimension for each ``Mamba2Layer``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        pattern: str | list[str] = "ssm:attn",
        attention: nn.Module | None = None,
        d_state: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        tokens = pattern.split(":") if isinstance(pattern, str) else list(pattern)
        if not tokens:
            raise ValueError("pattern must contain at least one token")
        invalid = [token for token in tokens if token not in VALID_TOKENS]
        if invalid:
            raise ValueError(
                f"unknown pattern tokens {invalid}; expected {list(VALID_TOKENS)}"
            )
        self.pattern = tokens

        layers: list[nn.Module] = []
        for token in tokens:
            if token == "ssm":
                layers.append(Mamba2Layer(hidden_size, d_state=d_state))
            elif attention is None:
                layers.append(MultiHeadAttention(hidden_size, num_heads))
            else:
                layers.append(attention)
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run each sublayer in pattern order.

        SSM sublayers return ``(output, state)``; only ``output`` is passed
        on and the state is dropped (no streaming across calls here).
        ``attention_mask`` is forwarded to every attention sublayer.

        Args:
            hidden_state: Input of shape ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional mask for the attention sublayers.

        Returns:
            Final hidden states, same shape as the input.
        """
        for layer in self.layers:
            if isinstance(layer, Mamba2Layer):
                hidden_state, _ = layer(hidden_state)
            else:
                hidden_state = layer(hidden_state, attention_mask=attention_mask)
        return hidden_state

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"pattern={':'.join(self.pattern)}"
        )
