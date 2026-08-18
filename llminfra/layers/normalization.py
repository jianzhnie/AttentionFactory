"""Normalization layers used by mainstream transformer architectures."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root mean square normalization used by Llama/Qwen/Mistral-style models.

    Args:
        hidden_size: Feature dimension.
        eps: Small value added to the variance for numerical stability.
        elementwise_affine: Whether to use a learnable per-dimension weight.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``."""
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight
        return normalized

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class LayerNorm(nn.Module):
    """Teaching-grade layer normalization over the last dimension.

    Normalizes to zero mean and unit variance, then applies a learnable
    per-dimension weight and (optionally) bias. Implemented manually with
    explicit mean/variance computation for readability.

    Args:
        hidden_size: Feature dimension.
        eps: Small value added to the variance for numerical stability.
        bias: Whether to use a learnable per-dimension bias.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x``."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.weight
        if self.bias is not None:
            normalized = normalized + self.bias
        return normalized

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class DeepNorm(nn.Module):
    """DeepNorm residual scaling from the GLM-130B paper.

    Implements ``norm(alpha * residual + sublayer_output)`` as introduced in
    "GLM-130B: An Open Bilingual Pre-trained Model" (Zeng et al., 2022),
    based on DeepNorm (Wang et al., 2022). Scaling the residual branch by
    ``alpha > 1`` bounds the residual growth and allows stable training of
    very deep (100+ layer) models. GLM-130B uses
    ``alpha = (2 * num_layers) ** 0.5`` with a LayerNorm inside.

    Args:
        hidden_size: Feature dimension.
        alpha: Residual scaling factor. Defaults to 1.0 (plain post-norm
            residual, no scaling).
        eps: Epsilon of the internal :class:`LayerNorm`.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.alpha = float(alpha)
        self.norm = LayerNorm(hidden_size, eps=eps)

    def forward(
        self, residual: torch.Tensor, sublayer_output: torch.Tensor
    ) -> torch.Tensor:
        """Combine and normalize a residual branch with a sublayer output.

        Args:
            residual: The residual (skip-connection) tensor.
            sublayer_output: Output of the attention/FFN sublayer.

        Returns:
            ``norm(alpha * residual + sublayer_output)``.
        """
        return self.norm(residual * self.alpha + sublayer_output)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, alpha={self.alpha}"


class LayerScale(nn.Module):
    """Per-channel learnable scaling of a sublayer output.

    Introduced in "Going deeper with Image Transformers" (CaiT, Touvron et
    al., 2021) and also used by CogView and several ViT-style LLM blocks.
    With the default ``init_value=1.0`` the module is initially an identity;
    small init values (e.g. 1e-5, as in CaiT) let very deep models start
    close to the residual branch.

    Args:
        hidden_size: Feature dimension.
        init_value: Initial value of every per-channel scale.
    """

    def __init__(self, hidden_size: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.init_value = float(init_value)
        self.weight = nn.Parameter(torch.full((hidden_size,), self.init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the last dimension of ``x`` per channel."""
        return x * self.weight

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, init_value={self.init_value}"
