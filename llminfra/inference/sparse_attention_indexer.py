"""Educational block-sparse indexer for top-k KV block selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BlockSparseIndexer(nn.Module):
    """Learned global KV block indexer used with block-sparse attention.

    The indexer scores each KV block and returns a
    ``(batch, heads, num_q_blocks, top_k)`` tensor that can be passed to
    ``BlockSparseAttention``. This is a simplified teaching version of the
    learned indexers used by MSA/DSA.

    Note:
        When fewer than ``top_k`` candidate blocks exist (early query blocks
        under ``causal=True``), the selection is padded by repeating the last
        valid block index. Consumers that build a boolean mask from the
        indices (such as `BlockSparseAttention`) treat duplicates
        idempotently, so the padding does not double-count attention. Block
        scores are means over zero-padded tails, so the final partial block
        is scored slightly low.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        top_k: int,
        max_seq_len: int,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if block_size < 1 or top_k < 1:
            raise ValueError("block_size and top_k must be >= 1")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.max_seq_len = int(max_seq_len)
        self.causal = bool(causal)
        self.score_proj = nn.Linear(hidden_size, num_heads)
        nn.init.xavier_uniform_(self.score_proj.weight)
        if self.score_proj.bias is not None:
            nn.init.zeros_(self.score_proj.bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Return selected KV block indices for each query block."""
        batch_size, seq_len, _ = hidden_state.size()
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        if seq_len == 0:
            return hidden_state.new_zeros(
                (batch_size, self.num_heads, 0, self.top_k), dtype=torch.long
            )
        padded_len = (
            (seq_len + self.block_size - 1) // self.block_size
        ) * self.block_size
        if padded_len != seq_len:
            hidden_state = F.pad(hidden_state, (0, 0, 0, padded_len - seq_len))

        block_count = padded_len // self.block_size
        block_vectors = hidden_state.view(
            batch_size, block_count, self.block_size, self.hidden_size
        ).mean(dim=2)
        scores = self.score_proj(block_vectors).transpose(1, 2)

        rows: list[torch.Tensor] = []
        for query_block in range(block_count):
            candidate_scores = (
                scores[:, :, : query_block + 1] if self.causal else scores
            )
            count = min(self.top_k, candidate_scores.size(-1))
            selected = torch.topk(candidate_scores, count, dim=-1).indices
            if count < self.top_k:
                last = candidate_scores.size(-1) - 1
                selected = F.pad(selected, (0, self.top_k - count), value=last)
            rows.append(selected)
        return torch.stack(rows, dim=2)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"block_size={self.block_size}, top_k={self.top_k}, "
            f"causal={self.causal}"
        )
