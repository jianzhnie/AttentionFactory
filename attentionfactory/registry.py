"""Registry helpers for constructing attention and positional modules."""

from __future__ import annotations

from .alibi_attention import AlibiAttention
from .block_sparse_attention import BlockSparseAttention
from .compressed_sparse_attention import CompressedSparseAttention
from .gated_delta_net import GatedDeltaNet
from .gqa import GroupQueryAttention
from .hybrid_attention import HybridAttention
from .lightning_attention import LightningAttention
from .linear_attention import LinearAttention
from .mha import MultiHeadAttention
from .mla import MultiHeadLatentAttention
from .mqa import MultiQueryAttention
from .positional import (
    BasePositionalEncoding,
    get_positional_encoding,
)
from .ring_attention import RingAttention
from .sliding_window_attention import SlidingWindowAttention

ATTENTION_REGISTRY = {
    "alibi": AlibiAttention,
    "mha": MultiHeadAttention,
    "mqa": MultiQueryAttention,
    "gqa": GroupQueryAttention,
    "mla": MultiHeadLatentAttention,
    "swa": SlidingWindowAttention,
    "block_sparse": BlockSparseAttention,
    "compressed_sparse": CompressedSparseAttention,
    "ring": RingAttention,
    "linear": LinearAttention,
    "lightning": LightningAttention,
    "gated_delta": GatedDeltaNet,
    "hybrid": HybridAttention,
}


def build_attention(
    name: str,
    hidden_size: int,
    num_heads: int,
    **kwargs: object,
):
    """Build an attention module by registry name.

    Required extra keyword arguments depend on the architecture, for example
    ``window_size`` for ``swa``, ``block_size`` for ``block_sparse``, or
    ``q_latent_size`` / ``kv_latent_size`` for ``mla``.
    """
    try:
        factory = ATTENTION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown attention: {name}; available: {list_attentions()}"
        ) from exc
    return factory(hidden_size=hidden_size, num_heads=num_heads, **kwargs)


def list_attentions() -> list[str]:
    """Return the available attention registry names."""
    return list(ATTENTION_REGISTRY)


def build_positional_encoding(
    name: str,
    *,
    dim: int,
    num_heads: int | None = None,
    max_seq_len: int = 4096,
    **kwargs: object,
) -> BasePositionalEncoding:
    """Build a positional encoding module by name."""
    return get_positional_encoding(
        name,
        dim=dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        **kwargs,
    )
