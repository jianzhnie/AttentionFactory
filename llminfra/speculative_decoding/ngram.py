"""N-Gram speculative decoding."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class NGramSpeculator(nn.Module):
    """Draft tokens by copying a continuation observed in the prompt."""

    def __init__(
        self,
        target_model: Callable[[torch.Tensor], torch.Tensor],
        ngram_size: int = 2,
        num_speculative_tokens: int = 4,
        append_bonus_token: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if ngram_size < 1 or num_speculative_tokens < 1:
            raise ValueError("ngram_size and num_speculative_tokens must be >= 1")
        self.target_model = target_model
        self.ngram_size = int(ngram_size)
        self.num_speculative_tokens = int(num_speculative_tokens)
        self.append_bonus_token = bool(append_bonus_token)
        self.pad_token_id = int(pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate and verify one N-Gram speculative block."""
        if input_ids.dim() != 2 or input_ids.size(1) < self.ngram_size:
            raise ValueError("input_ids must be 2-D and contain ngram_size tokens")
        rows = []
        for row in input_ids:
            sequence = row.tolist()
            key = sequence[-self.ngram_size :]
            match = next(
                (
                    i
                    for i in range(len(sequence) - self.ngram_size - 1, -1, -1)
                    if sequence[i : i + self.ngram_size] == key
                ),
                None,
            )
            if match is None:
                continuation = [self.pad_token_id]
            else:
                continuation = sequence[match + self.ngram_size :]
            continuation = continuation[: self.num_speculative_tokens]
            draft = row.new_tensor(
                continuation
                + [continuation[-1]] * (self.num_speculative_tokens - len(continuation))
            )
            logits = self.target_model(torch.cat((row, draft)).unsqueeze(0))
            verified = torch.argmax(logits[:, row.numel() - 1 : -1], dim=-1)[0]
            accepted = []
            for index, token in enumerate(verified):
                accepted.append(token.view(1))
                if token != draft[index]:
                    break
            if self.append_bonus_token and len(accepted) == self.num_speculative_tokens:
                accepted.append(torch.argmax(logits[:, -1], dim=-1))
            rows.append(torch.cat((row, *accepted)))
        output = input_ids.new_full(
            (len(rows), max(row.numel() for row in rows)), self.pad_token_id
        )
        for index, row in enumerate(rows):
            output[index, : row.numel()] = row
        return output


__all__ = ["NGramSpeculator"]
