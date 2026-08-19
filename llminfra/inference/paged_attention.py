"""Educational PagedAttention cache manager and dense gather implementation.

PagedAttention is primarily a *system-level* KV cache optimization used by
vLLM: the cache is split into fixed-size physical blocks, and a sequence is
described by a block table instead of a contiguous memory allocation. This
file provides a small PyTorch-only simulation of that interface, including
reference counts and copy-on-write for shared prefix blocks. It does not
implement CUDA memory management or asynchronous tiered offload.
"""

from __future__ import annotations

import torch


class PagedKVBlockAllocator:
    """Simple allocator for fixed-size physical KV blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.num_blocks = int(num_blocks)
        self.free_blocks = list(range(self.num_blocks))
        self.allocated: set[int] = set()
        self.ref_counts = [0] * self.num_blocks

    def allocate(self) -> int:
        """Allocate one physical block and return its id."""
        if not self.free_blocks:
            raise RuntimeError("Paged KV cache is full")
        block_id = self.free_blocks.pop()
        self.allocated.add(block_id)
        self.ref_counts[block_id] = 1
        return block_id

    def retain(self, block_id: int) -> None:
        """Add one logical owner for an allocated physical block."""
        if block_id not in self.allocated:
            raise ValueError(f"Block {block_id} is not allocated")
        self.ref_counts[block_id] += 1

    def free(self, block_id: int) -> None:
        """Return one physical block to the free list."""
        if block_id not in self.allocated:
            raise ValueError(f"Block {block_id} is not allocated")
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.allocated.remove(block_id)
            self.free_blocks.append(block_id)

    def reference_count(self, block_id: int) -> int:
        """Return the number of logical sequences owning ``block_id``."""
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Block id out of range: {block_id}")
        return self.ref_counts[block_id]


class PagedAttentionCache:
    """Fixed-block KV cache with per-sequence block tables.

    ``append`` accepts one or more new key/value rows for a sequence. The
    implementation simulates the mapping from logical token slots to physical
    blocks; it is not a CUDA paged memory allocator.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if block_size < 1 or num_heads < 1 or head_dim < 1:
            raise ValueError("block_size, num_heads and head_dim must be >= 1")
        self.allocator = PagedKVBlockAllocator(num_blocks)
        self.block_size = int(block_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.key_cache = torch.zeros(
            num_blocks,
            block_size,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        self.value_cache = torch.zeros_like(self.key_cache)
        self.block_tables: dict[int, list[int]] = {}
        self.num_tokens: dict[int, int] = {}

    def append(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append ``key``/``value`` rows to ``seq_id``.

        Args:
            seq_id: Logical sequence id.
            key: Shape ``(seq_len, num_heads, head_dim)``.
            value: Same shape as ``key``.
        """
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError("key and value must have shape (seq, heads, head_dim)")
        if key.size(1) != self.num_heads or key.size(2) != self.head_dim:
            raise ValueError("key/value head configuration does not match cache")
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")

        block_table = self.block_tables.setdefault(seq_id, [])
        num_tokens = self.num_tokens.setdefault(seq_id, 0)

        for token_idx in range(key.size(0)):
            slot_in_block = num_tokens % self.block_size
            if slot_in_block == 0:
                block_table.append(self.allocator.allocate())
            elif self.allocator.reference_count(block_table[-1]) > 1:
                # A cloned sequence shares its prefix blocks. Appending into
                # a partial shared tail first creates a private copy.
                shared_block = block_table[-1]
                private_block = self.allocator.allocate()
                self.key_cache[private_block].copy_(self.key_cache[shared_block])
                self.value_cache[private_block].copy_(self.value_cache[shared_block])
                block_table[-1] = private_block
                self.allocator.free(shared_block)
            physical_block = block_table[-1]
            self.key_cache[physical_block, slot_in_block] = key[token_idx]
            self.value_cache[physical_block, slot_in_block] = value[token_idx]
            num_tokens += 1
            self.num_tokens[seq_id] = num_tokens

    def get(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the cached K/V rows for ``seq_id`` as dense tensors."""
        if seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence id: {seq_id}")
        block_table = self.block_tables[seq_id]
        num_tokens = self.num_tokens[seq_id]
        if num_tokens == 0:
            empty_shape = (0, self.num_heads, self.head_dim)
            return (
                self.key_cache.new_zeros(empty_shape),
                self.value_cache.new_zeros(empty_shape),
            )
        rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        remaining = num_tokens
        for block_id in block_table:
            count = min(remaining, self.block_size)
            rows.append(
                (
                    self.key_cache[block_id, :count],
                    self.value_cache[block_id, :count],
                )
            )
            remaining -= count
        keys = torch.cat([row[0] for row in rows], dim=0)
        values = torch.cat([row[1] for row in rows], dim=0)
        return keys, values

    def clone_sequence(self, source_seq_id: int, new_seq_id: int) -> None:
        """Clone a sequence by sharing blocks until either clone is mutated."""
        if new_seq_id in self.block_tables:
            raise ValueError(f"Sequence already exists: {new_seq_id}")
        if source_seq_id not in self.block_tables:
            raise KeyError(f"Unknown sequence id: {source_seq_id}")
        blocks = list(self.block_tables[source_seq_id])
        for block_id in blocks:
            self.allocator.retain(block_id)
        self.block_tables[new_seq_id] = blocks
        self.num_tokens[new_seq_id] = self.num_tokens[source_seq_id]

    def reset(self, seq_id: int) -> None:
        """Free all physical blocks owned by ``seq_id`` and clear its table."""
        for block_id in self.block_tables.get(seq_id, []):
            self.allocator.free(block_id)
        self.block_tables.pop(seq_id, None)
        self.num_tokens.pop(seq_id, None)


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: list[int],
    num_tokens: int,
    block_size: int,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Dense-query attention over a paged KV cache.

    Args:
        query: Shape ``(q_len, num_heads, head_dim)``.
        key_cache: Shape ``(num_blocks, block_size, num_heads, head_dim)``.
        value_cache: Same shape as ``key_cache``.
        block_table: Physical block ids owned by this sequence.
        num_tokens: Number of valid tokens in the sequence cache.
        block_size: Number of tokens per physical block.
        causal: If True, each query may attend only to preceding cached tokens.

    Returns:
        Output tensor of shape ``(q_len, num_heads, head_dim)``.
    """
    if query.dim() != 3:
        raise ValueError("query must have shape (q_len, heads, head_dim)")
    if num_tokens == 0 or not block_table:
        return value_cache.new_zeros(
            (query.size(0), value_cache.size(2), value_cache.size(3))
        )

    rows: list[torch.Tensor] = []
    remaining = num_tokens
    for block_id in block_table:
        count = min(remaining, block_size)
        rows.append(key_cache[block_id, :count])
        remaining -= count
    key = torch.cat(rows, dim=0)

    rows = []
    remaining = num_tokens
    for block_id in block_table:
        count = min(remaining, block_size)
        rows.append(value_cache[block_id, :count])
        remaining -= count
    value = torch.cat(rows, dim=0)

    scale = 1.0 / (query.size(-1) ** 0.5)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if causal:
        q_pos = torch.arange(query.size(0), device=query.device)
        k_pos = torch.arange(num_tokens, device=query.device)
        offset = num_tokens - query.size(0)
        scores = scores.masked_fill(
            k_pos[None, None, :] > (q_pos[None, :, None] + offset), float("-inf")
        )
    weights = torch.softmax(scores, dim=-1).to(value.dtype)
    return torch.einsum("hqk,khd->qhd", weights, value)
