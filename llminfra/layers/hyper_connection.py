"""Reference implementation of manifold-constrained hyper-connections.

The implementation keeps the residual-stream mixing explicit and entirely in
PyTorch.  It is intended for architecture experiments and checkpoint-shape
prototyping; production mHC implementations should replace the Sinkhorn loop
with a fused kernel and preserve stream state across pipeline stages.
"""

from __future__ import annotations

import torch
from torch import nn


class ManifoldConstrainedHyperConnection(nn.Module):
    """Mix ``hc_mult`` residual streams with a doubly-stochastic matrix.

    ``hidden`` and ``branch`` use shape ``(..., hidden_size)``.  The module
    expands the residual into ``hc_mult`` streams, injects the branch output,
    applies a learned non-negative mixing matrix constrained by Sinkhorn
    normalization, and reduces the mixed streams with a learned softmax
    weighting (``stream_weight``, initialized near-uniform so it starts as
    the mean).  The near-mean reduction keeps this reference layer
    compatible with ordinary ``TransformerBlock`` residuals.

    Each stream carries its own learned ``stream_scale``/``stream_bias`` so
    the streams are genuine copies with broken symmetry; without them every
    stream would be identical and the mixing matrix could not affect the
    output.  Likewise the reduction must stay non-uniform: a doubly
    stochastic matrix times an exact mean is invariant to the matrix.

    Args:
        hidden_size: Feature dimension of the residual stream.
        hc_mult: Number of parallel residual streams.
        sinkhorn_iters: Number of normalization iterations for the mixing
            matrix.  More iterations improve the doubly-stochastic constraint.
        init_scale: Initial branch contribution before stream mixing.

    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if hc_mult <= 0:
            raise ValueError("hc_mult must be positive")
        if sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")
        self.hidden_size = int(hidden_size)
        self.hc_mult = int(hc_mult)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.logits = nn.Parameter(torch.zeros(hc_mult, hc_mult))
        self.branch_scale = nn.Parameter(torch.full((hidden_size,), float(init_scale)))
        self.stream_scale = nn.Parameter(torch.ones(hc_mult, hidden_size))
        self.stream_bias = nn.Parameter(torch.randn(hc_mult, hidden_size) * 0.02)
        self.stream_weight = nn.Parameter(torch.randn(hc_mult) * 0.02)

    def mixing_matrix(self) -> torch.Tensor:
        """Return the current approximately doubly-stochastic mixing matrix."""
        # Subtract the max before exponentiating so large logits cannot
        # overflow to inf (which would turn the normalization into NaN).
        matrix = (self.logits - self.logits.amax()).exp()
        for _ in range(self.sinkhorn_iters):
            row_sum = matrix.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(matrix.dtype).eps
            )
            matrix = matrix / row_sum
            column_sum = matrix.sum(dim=-2, keepdim=True).clamp_min(
                torch.finfo(matrix.dtype).eps
            )
            matrix = matrix / column_sum
        return matrix

    def forward(self, hidden: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        """Apply constrained stream mixing and return a normal residual shape."""
        if hidden.shape != branch.shape:
            raise ValueError("hidden and branch must have identical shapes")
        if hidden.ndim < 1 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(f"expected last dimension {self.hidden_size}")
        streams = hidden.unsqueeze(-2) * self.stream_scale + self.stream_bias
        streams = streams + self.branch_scale * branch.unsqueeze(-2)
        mixed = torch.einsum("ij,...jh->...ih", self.mixing_matrix(), streams)
        weights = torch.softmax(self.stream_weight, dim=0)
        return torch.einsum("i,...ih->...h", weights, mixed)


__all__ = ["ManifoldConstrainedHyperConnection"]
