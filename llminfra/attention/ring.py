"""Educational Ring Attention implementation.

Ring Attention splits the sequence into chunks and computes exact attention
by carrying a running online-softmax state across chunks. This module does
not simulate multi-device communication; it preserves the blockwise exact
attention structure used by long-context training.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseAttention, validate_attention_inputs


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    num_chunks: int = 2,
) -> torch.Tensor:
    """Exact blockwise attention over sequence chunks.

    Args:
        q: Shape ``(batch, heads, q_len, head_dim)``.
        k: Shape ``(batch, heads, kv_len, head_dim)``.
        v: Shape ``(batch, heads, kv_len, value_dim)``.
        causal: Apply causal masking.
        num_chunks: Number of sequence chunks.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D tensors")
    if q.size(0) != k.size(0) or k.size(0) != v.size(0):
        raise ValueError("batch sizes must match")
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("head counts must match")
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")

    batch, heads, q_len, head_dim = q.size()
    kv_len = k.size(2)
    scale = 1.0 / math.sqrt(head_dim)
    q_chunks = torch.chunk(q, num_chunks, dim=2)
    k_chunks = torch.chunk(k, num_chunks, dim=2)
    v_chunks = torch.chunk(v, num_chunks, dim=2)
    outputs: list[torch.Tensor] = []

    q_start = 0
    for q_chunk in q_chunks:
        # The accumulator has V's feature dimension, which may differ from
        # Q/K's head dimension.
        out = q_chunk.new_zeros(batch, heads, q_chunk.size(2), v.size(-1))
        row_max = torch.full(
            (batch, heads, q_chunk.size(2), 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        normalizer = torch.zeros(
            batch, heads, q_chunk.size(2), 1, device=q.device, dtype=q.dtype
        )
        k_start = 0
        for k_chunk, v_chunk in zip(k_chunks, v_chunks, strict=True):
            scores = torch.einsum("bhid,bhjd->bhij", q_chunk, k_chunk) * scale
            if causal:
                q_pos = torch.arange(
                    q_start, q_start + q_chunk.size(2), device=q.device
                ).view(-1, 1)
                k_pos = torch.arange(
                    k_start, k_start + k_chunk.size(2), device=q.device
                ).view(1, -1)
                scores = scores.masked_fill(
                    (q_pos + (kv_len - q_len)) < k_pos, float("-inf")
                )
            block_max = scores.max(dim=-1, keepdim=True).values
            finite_block_max = torch.where(
                torch.isfinite(block_max),
                block_max,
                torch.zeros_like(block_max),
            )
            new_max = torch.maximum(row_max, finite_block_max)
            probabilities = torch.where(
                torch.isfinite(block_max),
                torch.exp(scores - new_max),
                torch.zeros_like(scores),
            )
            old_scale = torch.exp(row_max - new_max)
            normalizer = old_scale * normalizer + probabilities.sum(
                dim=-1, keepdim=True
            )
            out = old_scale * out + probabilities @ v_chunk
            row_max = new_max
            k_start += k_chunk.size(2)

        safe_normalizer = normalizer.clamp_min(torch.finfo(normalizer.dtype).eps)
        outputs.append(out / safe_normalizer)
        q_start += q_chunk.size(2)
    return torch.cat(outputs, dim=2)


class RingAttention(BaseAttention):
    """Multi-head attention with exact chunked online-softmax forward.

    Args:
        hidden_size: Dimensionality of input and output features.
        num_heads: Number of attention heads.
        num_chunks: Number of sequence chunks processed by the online softmax.
        dropout: Dropout applied to the output when training (this module
            never materializes attention weights, so there is nothing to
            drop out on the probabilities).
        bias: Whether linear projections use biases.

    The forward pass is always causal and does not support
    ``attention_mask``; use another attention module if you need padding or
    custom masks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_chunks: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(hidden_size, num_heads, dropout, bias)
        self.num_chunks = int(num_chunks)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_projections(self.q_proj, self.k_proj, self.v_proj, self.o_proj)

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run chunked exact attention.

        Attention weights are not materialized in Ring Attention, so
        ``return_attention_weights=True`` is not supported.

        Raises:
            ValueError: If ``return_attention_weights`` is True or an
                ``attention_mask`` is passed (chunked causal attention does
                not support custom masks).
        """
        if return_attention_weights:
            raise ValueError("RingAttention does not materialize attention weights")
        if attention_mask is not None:
            raise ValueError("RingAttention does not support attention_mask")
        validate_attention_inputs(hidden_state, attention_mask, self.num_heads)
        q = self.split_head(self.q_proj(hidden_state))
        k = self.split_head(self.k_proj(hidden_state))
        v = self.split_head(self.v_proj(hidden_state))
        output = ring_attention(
            q,
            k,
            v,
            causal=True,
            num_chunks=self.num_chunks,
        )
        output = self.o_proj(self.combine_head(output))
        if self.training and self.dropout_prob > 0:
            output = self.dropout(output)
        return output

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, num_chunks={self.num_chunks}"
