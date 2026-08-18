"""Shared input builders and reference utilities for the test suite.

All builders take an explicit seed and use a local ``torch.Generator`` so
tests are deterministic without touching the global RNG state.
"""

from __future__ import annotations

import torch

from llminfra.flash_attention.common import reference_attention


def make_qkv(
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    value_dim: int,
    *,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random q/k/v in ``(batch, heads, seq, dim)`` layout.

    ``value_dim`` may differ from ``head_dim`` to cover the MLA-style case.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, heads, q_len, head_dim, generator=generator, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, head_dim, generator=generator, dtype=dtype)
    v = torch.randn(batch, heads, kv_len, value_dim, generator=generator, dtype=dtype)
    return q, k, v


def make_hidden_state(
    batch: int,
    seq: int,
    hidden: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Random input for the nn.Module attention variants, ``(batch, seq, hidden)``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, hidden, generator=generator)


def make_key_padding_mask(
    batch: int,
    kv_len: int,
    *,
    seed: int = 1,
    fully_masked_row: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Random 2D key padding mask (``True`` = valid) for the tiled versions.

    With ``fully_masked_row=True`` batch row 0 is entirely masked to exercise
    the empty-row edge case; every other row keeps at least one valid key.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    mask = torch.rand(batch, kv_len, generator=generator, device=device) > 0.3
    if fully_masked_row:
        mask[1:, 0] = True
        mask[0] = False
    else:
        mask[:, 0] = True
    return mask


def make_causal_mask(batch: int, seq: int) -> torch.Tensor:
    """Broadcastable ``(batch, 1, seq, seq)`` causal mask for the nn modules."""
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    return causal.expand(batch, 1, seq, seq)


def make_padding_mask(batch: int, seq: int) -> torch.Tensor:
    """Broadcastable ``(batch, 1, 1, seq)`` padding mask for the nn modules.

    Row 0 has its last two key positions masked; other rows attend everywhere.
    """
    mask = torch.ones(batch, 1, 1, seq, dtype=torch.bool)
    mask[0, 0, 0, -2:] = False
    return mask


def with_grad(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Detached clones of the inputs with ``requires_grad`` enabled."""
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


def reference_with_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference output plus its autograd gradients w.r.t. q/k/v."""
    q_ref, k_ref, v_ref = with_grad(q, k, v)
    out = reference_attention(
        q_ref, k_ref, v_ref, causal=causal, key_padding_mask=key_padding_mask
    )
    out.backward(grad_out)
    return out.detach(), q_ref.grad, k_ref.grad, v_ref.grad
