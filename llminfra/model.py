"""Composable causal language model built from LLMInfra modules."""

from __future__ import annotations

import torch
from torch import nn

from .layers.ffn import SwiGLUFFN
from .layers.norm import RMSNorm
from .layers.transformer import TransformerBlock
from .moe.moe import DeepSeekMoE
from .positional import ALiBiBias
from .registry import build_attention, build_positional_encoding


class CausalLMModel(nn.Module):
    """Teaching-level causal language model.

    This class composes embedding, positional encoding, transformer blocks,
    RMSNorm and an output head. It is intended for architecture experiments,
    not for reproducing a specific production model.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int = 4096,
        attention_name: str = "gqa",
        attention_kwargs: dict | None = None,
        positional: str = "rope",
        positional_kwargs: dict | None = None,
        use_moe: bool = False,
        num_experts: int = 8,
        expert_top_k: int = 2,
        num_shared_experts: int = 1,
        norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("vocab_size, hidden_size and num_layers must be >= 1")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        if positional == "alibi":
            attention_name = "alibi"
            positional = "none"
        self.attention_name = attention_name
        self.use_moe = bool(use_moe)
        self.tie_word_embeddings = bool(tie_word_embeddings)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        attention_kwargs = dict(attention_kwargs or {})
        if attention_name == "alibi" and "max_seq_len" not in attention_kwargs:
            attention_kwargs["max_seq_len"] = max_seq_len
        if attention_name == "gqa" and "num_kv_groups" not in attention_kwargs:
            attention_kwargs["num_kv_groups"] = max(1, num_heads // 2)

        self.positional = (
            None
            if positional in {None, "none"}
            else build_positional_encoding(
                positional,
                dim=hidden_size,
                max_seq_len=max_seq_len,
                **dict(positional_kwargs or {}),
            )
        )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden_size,
                num_heads,
                intermediate_size,
                attention=build_attention(
                    attention_name,
                    hidden_size,
                    num_heads,
                    **attention_kwargs,
                ),
                ffn=(
                    DeepSeekMoE(
                        hidden_size,
                        num_routed_experts=num_experts,
                        num_shared_experts=num_shared_experts,
                        intermediate_size=intermediate_size,
                        top_k=expert_top_k,
                    )
                    if use_moe
                    else SwiGLUFFN(hidden_size, intermediate_size)
                ),
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        )

        self.norm = RMSNorm(hidden_size, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the model over ``input_ids``.

        Args:
            input_ids: Long tensor of shape ``(batch, seq_len)``.
            attention_mask: Optional padding mask of shape
                ``(batch, seq_len)``, ``(batch, 1, seq_len)`` or
                ``(batch, 1, seq_len, seq_len)``.
            return_attention_weights: Return weights from the final block when
                the attention module supports them.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        batch_size, seq_len = input_ids.size()
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        hidden_state = self.embed_tokens(input_ids)
        if self.positional is not None and not isinstance(self.positional, ALiBiBias):
            hidden_state = self.positional(hidden_state)

        combined_mask = self._build_mask(
            attention_mask, batch_size, seq_len, input_ids.device
        )
        last_weights: torch.Tensor | None = None
        for layer_index, block in enumerate(self.blocks):
            wants_weights = (
                return_attention_weights and layer_index == self.num_layers - 1
            )
            result = block(
                hidden_state,
                attention_mask=combined_mask,
                return_attention_weights=wants_weights,
                layer_index=layer_index,
            )
            if wants_weights:
                hidden_state, last_weights = result
            else:
                hidden_state = result

        hidden_state = self.norm(hidden_state)
        logits = self.lm_head(hidden_state)
        if return_attention_weights:
            if last_weights is None:
                raise ValueError(
                    "The configured attention module does not return weights"
                )
            return logits, last_weights
        return logits

    @staticmethod
    def _build_mask(
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Combine a user padding mask with a causal mask."""
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        ).expand(batch_size, 1, seq_len, seq_len)
        if attention_mask is None:
            return causal
        if attention_mask.dim() == 2:
            padding = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            padding = attention_mask.unsqueeze(1)
        elif attention_mask.dim() == 4:
            padding = attention_mask
        else:
            raise ValueError("attention_mask must be 2D, 3D or 4D")
        if padding.size(0) != batch_size:
            raise ValueError("attention_mask batch size must match input_ids")
        return causal & padding.bool()

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, num_heads={self.num_heads}, "
            f"attention_name={self.attention_name}, use_moe={self.use_moe}"
        )
