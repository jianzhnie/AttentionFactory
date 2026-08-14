"""Numerical primitives shared by all educational FlashAttention versions.

Everything here operates either on a single (query tile, key/value tile) pair
or on the per-row online-softmax state carried between tiles. Scores and
softmax statistics are computed in float32 regardless of the input dtype;
fully masked rows are defined to contribute zeros and never produce NaNs.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .masking import build_block_mask
from .types import ForwardResult


def scaled_scores(q_block: torch.Tensor, k_block: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product scores ``Q K^T / sqrt(d)``, computed in float32.

    Inputs are upcast to float32 *before* the matmul so the score tile stays
    accurate for 16-bit inputs, mirroring the fp32-accumulation behavior of
    real GPU kernels.
    """
    scale = 1.0 / math.sqrt(q_block.shape[-1])
    q_fp32 = q_block.to(torch.float32) * scale
    return torch.einsum("bhid,bhjd->bhij", q_fp32, k_block.to(torch.float32))


def block_scores_and_mask(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_slice: slice,
    k_slice: slice,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Score tile for one (query tile, key tile) pair plus its validity mask."""
    scores = scaled_scores(q[:, :, q_slice, :], k[:, :, k_slice, :])
    mask = build_block_mask(
        batch_size=q.shape[0],
        q_slice=q_slice,
        k_slice=k_slice,
        q_len=q.shape[2],
        kv_len=k.shape[2],
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q.device,
    )
    return scores, mask


def compute_local_statistics(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    v_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unnormalized softmax statistics of a single score tile.

    Returns ``(block_max, block_sum, weighted_values)``: the row-wise maximum,
    the row-wise sum of ``exp(scores - block_max)``, and the unnormalized
    ``exp(scores - block_max) @ V`` contribution. Fully masked rows yield a
    ``block_max`` of 0 and zero contributions, so later merges stay finite.
    """
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    block_max = scores.max(dim=-1, keepdim=True).values
    # A fully masked row has max -inf; substitute 0 so exp() yields 0 for that
    # row instead of NaN. Masked positions are exp(-inf - m) = 0 either way.
    block_max = torch.where(
        torch.isfinite(block_max), block_max, torch.zeros_like(block_max)
    )
    exp_scores = torch.exp(scores - block_max)

    block_sum = exp_scores.sum(dim=-1, keepdim=True)
    weighted_values = torch.einsum(
        "bhij,bhjd->bhid", exp_scores.to(v_block.dtype), v_block
    )
    return block_max, block_sum, weighted_values


def merge_state_normalized(
    out_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold one tile's contribution into a running *normalized* output tile.

    FA1-style merge: the output tile is renormalized at every step, so it
    always holds the exact attention output over the keys seen so far.
    """
    new_row_max = torch.maximum(row_max_block, block_max)
    old_scale = torch.exp(row_max_block - new_row_max)
    new_scale = torch.exp(block_max - new_row_max)
    new_normalizer = old_scale * normalizer_block + new_scale * block_sum

    safe_normalizer = torch.where(
        new_normalizer > 0, new_normalizer, torch.ones_like(new_normalizer)
    )
    out_block = (old_scale * normalizer_block / safe_normalizer).to(
        out_block.dtype
    ) * out_block + (new_scale / safe_normalizer).to(
        weighted_values.dtype
    ) * weighted_values
    out_block = torch.where(new_normalizer > 0, out_block, torch.zeros_like(out_block))
    return out_block, new_normalizer, new_row_max


def merge_state_unnormalized(
    out_acc_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold one tile's contribution into the running *unnormalized* state.

    FA2-style merge: the accumulator stays scaled by ``exp(-row_max)`` and is
    only divided by the normalizer in `finalize_unnormalized`, once all tiles
    have been merged.
    """
    new_row_max = torch.maximum(row_max_block, block_max)
    old_scale = torch.exp(row_max_block - new_row_max)
    new_scale = torch.exp(block_max - new_row_max)
    out_acc_block = (
        old_scale.to(out_acc_block.dtype) * out_acc_block
        + new_scale.to(weighted_values.dtype) * weighted_values
    )
    normalizer_block = old_scale * normalizer_block + new_scale * block_sum
    return out_acc_block, normalizer_block, new_row_max


def finalize_unnormalized(
    out_acc: torch.Tensor,
    normalizers: torch.Tensor,
) -> torch.Tensor:
    """Apply the deferred softmax division to an unnormalized accumulator."""
    safe_normalizers = torch.where(
        normalizers > 0, normalizers, torch.ones_like(normalizers)
    )
    out = out_acc / safe_normalizers.to(out_acc.dtype)
    return torch.where(normalizers > 0, out, torch.zeros_like(out))


def lse_from_state(normalizers: torch.Tensor, row_max: torch.Tensor) -> torch.Tensor:
    """Log-sum-exp of the masked scores from the final online-softmax state.

    Fully masked rows (normalizer 0) get an LSE of 0 by convention.
    """
    safe_normalizers = torch.where(
        normalizers > 0, normalizers, torch.ones_like(normalizers)
    )
    lse = row_max + torch.log(safe_normalizers)
    return torch.where(normalizers > 0, lse, torch.zeros_like(lse))


def assemble_forward_result(
    out_blocks: list[torch.Tensor],
    normalizer_blocks: list[torch.Tensor],
    row_max_blocks: list[torch.Tensor],
    *,
    normalized: bool,
    out_dtype: torch.dtype,
    saved_state: dict[str, Any],
) -> ForwardResult:
    """Concatenate per-tile state and build the `ForwardResult`.

    With ``normalized=False`` the deferred softmax division (FA2-style) is
    applied here. Accumulators are float32; the output is cast back to
    ``out_dtype`` so it matches the input precision.
    """
    out_acc = torch.cat(out_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)
    out = out_acc if normalized else finalize_unnormalized(out_acc, normalizers)
    return ForwardResult(
        out=out.to(out_dtype),
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state=saved_state,
    )


def probabilities_from_lse(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> torch.Tensor:
    """Rebuild a tile's softmax probabilities from saved log-sum-exp values."""
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))
    return torch.exp(scores - lse_block)


def backward_step(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    out_block: torch.Tensor,
    grad_out_block: torch.Tensor,
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact attention gradients for one (query tile, key/value tile) pair.

    Probabilities are recomputed from the saved log-sum-exp instead of a
    stored attention matrix, then the standard attention derivatives apply:

    .. code-block:: text

        P  = exp(S - LSE)        recomputed probabilities
        dV = P^T @ dO
        dP = dO @ V^T
        D  = rowsum(dO * O)      saved-output trick, avoids storing P
        dS = P * (dP - D)
        dQ = dS @ K / sqrt(d)
        dK = dS^T @ Q / sqrt(d)
    """
    probabilities = probabilities_from_lse(scores, valid_mask, lse_block).to(
        grad_out_block.dtype
    )
    dV = torch.einsum("bhij,bhid->bhjd", probabilities, grad_out_block)
    dP = torch.einsum("bhid,bhjd->bhij", grad_out_block, v_block)
    row_dot = torch.sum(grad_out_block * out_block, dim=-1, keepdim=True)
    dS = probabilities * (dP - row_dot)
    scale = 1.0 / math.sqrt(q_block.shape[-1])
    dQ = scale * torch.einsum("bhij,bhjd->bhid", dS, k_block)
    dK = scale * torch.einsum("bhij,bhid->bhjd", dS, q_block)
    return dQ, dK, dV
