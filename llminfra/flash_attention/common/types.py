"""Result containers returned by the educational FlashAttention versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ForwardResult:
    """Output of a tiled forward pass.

    Attributes:
        out: Attention output, shape ``(batch, heads, q_len, v_dim)``.
        lse: Per-row log-sum-exp of the masked scores, shape
            ``(batch, heads, q_len, 1)``. Saved so backward can recompute
            probabilities instead of storing the attention matrix. Fully
            masked rows carry ``0`` by convention.
        normalizers: Final per-row softmax denominators (float32).
        row_max: Final per-row running maxima (float32).
        saved_state: Optional debug metadata (loop order, pipeline traces).
    """

    out: torch.Tensor
    lse: torch.Tensor
    normalizers: torch.Tensor | None = None
    row_max: torch.Tensor | None = None
    saved_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackwardResult:
    """Gradients of a tiled backward pass with respect to q, k and v."""

    grad_q: torch.Tensor
    grad_k: torch.Tensor
    grad_v: torch.Tensor
    debug_state: dict[str, Any] = field(default_factory=dict)
