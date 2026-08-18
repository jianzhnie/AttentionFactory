"""Dense attention baseline used as the correctness reference.

This module materializes the full score and probability matrices, exactly
what the tiled online-softmax versions avoid. It exists so tests can check
the educational kernels against a straightforward, obviously-correct formula.
"""

from __future__ import annotations

import torch

from .masking import build_full_mask, normalize_key_padding_mask
from .ops import scaled_scores


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense softmax attention over the full score matrix.

    Softmax statistics are computed in float32. Fully masked rows produce
    zero outputs (and never NaNs), matching the tiled implementations.
    """
    key_padding_mask = normalize_key_padding_mask(
        key_padding_mask,
        batch_size=q.shape[0],
        kv_len=k.shape[2],
        device=q.device,
    )

    scores = scaled_scores(q, k)
    full_mask = build_full_mask(
        batch_size=q.shape[0],
        q_len=q.shape[2],
        kv_len=k.shape[2],
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q.device,
    )
    if full_mask is not None:
        scores = scores.masked_fill(~full_mask, float("-inf"))

    row_max = scores.max(dim=-1, keepdim=True).values
    # Fully masked rows have max -inf; substitute 0 so exp() yields 0 instead
    # of NaN. Masked positions are exp(-inf - m) = 0 either way.
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)

    safe_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    probabilities = torch.where(
        row_sum > 0, exp_scores / safe_sum, torch.zeros_like(exp_scores)
    )
    return probabilities.to(v.dtype) @ v
