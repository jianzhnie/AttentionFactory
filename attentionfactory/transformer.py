"""Transformer building blocks that compose attention, norm and FFN modules."""

from __future__ import annotations

import torch
from torch import nn

from .ffn import SwiGLUFFN
from .hybrid_attention import HybridAttention
from .mha import MultiHeadAttention
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional pluggable attention and FFN.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        intermediate_size: FFN intermediate dimension.
        attention: Optional attention module. Defaults to ``MultiHeadAttention``.
        ffn: Optional FFN module. Defaults to ``SwiGLUFFN``.
        norm_eps: RMSNorm epsilon.
        pre_norm: Whether to apply pre-norm residual style.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        attention: nn.Module | None = None,
        ffn: nn.Module | None = None,
        norm_eps: float = 1e-5,
        pre_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.pre_norm = bool(pre_norm)
        self.attention = attention or MultiHeadAttention(hidden_size, num_heads)
        self.ffn = ffn or SwiGLUFFN(hidden_size, intermediate_size)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
        layer_index: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one transformer block.

        ``layer_index`` is forwarded to ``HybridAttention`` when used, so the
        caller can reproduce Qwen3-Next/Kimi-style 3:1 linear/full layouts.
        """
        normed = self.norm1(hidden_state)
        if isinstance(self.attention, HybridAttention):
            result = self.attention(
                normed,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
                layer_index=layer_index,
            )
        else:
            result = self.attention(
                normed,
                attention_mask=attention_mask,
                return_attention_weights=return_attention_weights,
            )

        if return_attention_weights:
            attention_output, attention_weights = result
        else:
            attention_output = result

        if self.pre_norm:
            hidden_state = hidden_state + attention_output
            hidden_state = hidden_state + self.ffn(self.norm2(hidden_state))
        else:
            hidden_state = self.norm1(hidden_state + attention_output)
            hidden_state = self.norm2(hidden_state + self.ffn(hidden_state))

        if return_attention_weights:
            return hidden_state, attention_weights
        return hidden_state

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"intermediate_size={self.intermediate_size}, pre_norm={self.pre_norm}"
        )
