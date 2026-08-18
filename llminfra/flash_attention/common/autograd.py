"""Autograd bridge for the tiled attention implementations.

Wraps a version's plain ``forward``/``backward`` function pair in a
`torch.autograd.Function`, so the educational kernels can be called like any
differentiable PyTorch operation: the output tensor records the graph and
``loss.backward()`` routes through the tiled backward pass.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .config import FlashAttentionConfig
from .types import BackwardResult, ForwardResult

ForwardImpl = Callable[..., ForwardResult]
BackwardImpl = Callable[..., BackwardResult]


class TiledAttentionFunction(torch.autograd.Function):
    """Wire a tiled (forward, backward) pair into PyTorch autograd.

    Like the real FlashAttention kernels, only the inputs, the output and the
    per-row log-sum-exp are saved; attention probabilities are recomputed
    during the backward pass instead of being stored.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        key_padding_mask: torch.Tensor | None,
        config: FlashAttentionConfig | None,
        forward_impl: ForwardImpl,
        backward_impl: BackwardImpl,
    ) -> torch.Tensor:
        config = config or FlashAttentionConfig()
        result = forward_impl(
            q, k, v, causal=causal, key_padding_mask=key_padding_mask, config=config
        )
        ctx.save_for_backward(q, k, v, result.out, result.lse)
        ctx.causal = causal
        ctx.key_padding_mask = key_padding_mask
        ctx.config = config
        ctx.backward_impl = backward_impl
        return result.out

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
    ]:
        q, k, v, out, lse = ctx.saved_tensors
        # Rebuild the minimal forward result the tiled backward needs.
        forward_result = ForwardResult(out=out, lse=lse)
        grads = ctx.backward_impl(
            q,
            k,
            v,
            grad_out.contiguous(),
            forward_result,
            causal=ctx.causal,
            key_padding_mask=ctx.key_padding_mask,
            config=ctx.config,
        )
        # Gradient slots for causal/key_padding_mask/config and the two
        # implementation callables are unused.
        return grads.grad_q, grads.grad_k, grads.grad_v, None, None, None, None, None
