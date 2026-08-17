"""Teaching-grade multimodal fusion interfaces for LLMInfra.

This module provides the minimal interfaces needed to plug vision features
into a text model: a ``VisionEncoderAdapter`` that projects pre-computed
patch features into the model's hidden size, and a ``CrossAttentionFuser``
that performs late fusion by letting text tokens attend to vision tokens.

Simplifications (documented for teaching purposes):

- ``VisionEncoderAdapter`` is a single linear projection, not a real ViT.
  It is intentionally *not* compatible with pretrained ViT checkpoints; it
  only defines the tensor interface ``(batch, num_patches, vision_dim) ->
  (batch, num_patches, hidden_size)``.
- ``CrossAttentionFuser`` reuses ``CrossAttention`` from
  ``llminfra.encoder_decoder`` because ``MultiHeadAttention`` only supports
  self-attention (query length == key/value length), while fusion requires
  text queries to attend over a different number of vision key/value tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder_decoder import CrossAttention


class VisionEncoderAdapter(nn.Module):
    """Project pre-computed vision patch features into the model hidden size.

    This is an interface skeleton: it assumes a vision encoder has already
    produced patch-level features and only learns the linear projection into
    the language model's representation space. It does not load or reproduce
    real ViT weights.

    Args:
        vision_dim: Feature dimension produced by the (external) vision
            encoder.
        hidden_size: Model hidden dimension to project into.
        bias: Whether to use a bias in the projection.
    """

    def __init__(self, vision_dim: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        if vision_dim < 1 or hidden_size < 1:
            raise ValueError("vision_dim and hidden_size must be >= 1")
        self.vision_dim = int(vision_dim)
        self.hidden_size = int(hidden_size)
        self.proj = nn.Linear(vision_dim, hidden_size, bias=bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features into the model hidden size.

        Args:
            vision_features: Tensor of shape
                ``(batch, num_patches, vision_dim)``.

        Returns:
            Tensor of shape ``(batch, num_patches, hidden_size)``.
        """
        if vision_features.dim() != 3:
            raise ValueError("vision_features must be 3D")
        if vision_features.size(-1) != self.vision_dim:
            raise ValueError(
                f"vision_features last dim must be {self.vision_dim}, "
                f"got {vision_features.size(-1)}"
            )
        return self.proj(vision_features)

    def extra_repr(self) -> str:
        return f"vision_dim={self.vision_dim}, hidden_size={self.hidden_size}"


class CrossAttentionFuser(CrossAttention):
    """Late-fusion module: text tokens attend to vision tokens.

    Queries are projected from the text hidden state while keys and values
    are projected from (adapter-projected) vision features, so the number of
    vision tokens may differ from the number of text tokens. The attention
    math is inherited from ``CrossAttention`` (``nn.Linear`` projections plus
    ``BaseAttention.split_head``/``combine_head``/
    ``compute_attention_weights``).

    Args:
        hidden_size: Dimensionality of text and (projected) vision features.
        num_heads: Number of attention heads. Must divide ``hidden_size``.
        dropout: Dropout probability for attention weights.
        bias: Whether to use bias in the linear projections.
    """

    def forward(
        self,
        text_state: torch.Tensor,
        vision_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Fuse vision information into the text hidden state.

        Args:
            text_state: Tensor of shape ``(batch, text_len, hidden_size)``.
            vision_state: Tensor of shape ``(batch, num_patches,
                hidden_size)``, typically the output of
                ``VisionEncoderAdapter``.
            attention_mask: Optional mask broadcastable against the
                ``(batch, num_heads, text_len, num_patches)`` score tensor,
                e.g. a ``(batch, 1, 1, num_patches)`` mask over valid
                patches. 1/True marks visible patches, 0/False masks them.
            return_attention_weights: Also return the attention weights.

        Returns:
            Fused tensor of shape ``(batch, text_len, hidden_size)``, and
            optionally the weights of shape ``(batch, num_heads, text_len,
            num_patches)``.
        """
        return super().forward(
            text_state,
            vision_state,
            attention_mask=attention_mask,
            return_attention_weights=return_attention_weights,
        )
