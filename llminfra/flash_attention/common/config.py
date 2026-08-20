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
            the official FA3 release). The simulation reads each tile's amax
            on the host, so on CUDA it forces one host-device sync per tile.
        keep_debug_state: Record scheduler/pipeline metadata in the results.
            This is a teaching/debugging aid: the per-tile trace entries read
            GPU tensors on the host, so on CUDA each tile costs a host-device
            sync. Performance-sensitive runs should set this to ``False``.

    """

    block_size_q: int = 1024
    block_size_kv: int = 1024
    num_stages: int = 2
    fp8: bool = False
    keep_debug_state: bool = True
