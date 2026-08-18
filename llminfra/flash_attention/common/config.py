"""Configuration shared by the educational FlashAttention versions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlashAttentionConfig:
    """Knobs shared by the simplified educational implementations.

    Attributes:
        block_size_q: Number of query rows per tile.
        block_size_kv: Number of key/value rows per tile.
        num_stages: Logical pipeline depth for FA3's ping-pong tile buffers.
        fp8: Simulate FA3's block-quantized FP8 path (forward only, matching
            the official FA3 release).
        keep_debug_state: Record scheduler/pipeline metadata in the results.
    """

    block_size_q: int = 1024
    block_size_kv: int = 1024
    num_stages: int = 2
    fp8: bool = False
    keep_debug_state: bool = True
