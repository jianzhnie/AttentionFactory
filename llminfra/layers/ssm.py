"""Simplified state-space model layer for hybrid Mamba-style blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class Mamba2Layer(nn.Module):
    """Teaching-level diagonal SSM layer.

    This is not a faithful Mamba-2 CUDA implementation. The main teaching
    simplification is the per-channel structure: a real Mamba-2 keeps a
    separate state of shape ``(batch, channels, d_state)`` per channel,
    whereas this layer collapses the channel dimension to a scalar at each
    step (``u`` is summed over channels) and shares a single state of shape
    ``(batch, d_state)`` across all channels. It preserves the fixed-size
    recurrent state structure used by Mamba-2 hybrid models, and the state
    can be threaded through calls for streaming-style decoding.

    Two scan strategies are offered (``scan`` argument of ``forward``):

    - ``"recurrent"`` (default): plain step-by-step recurrence.
    - ``"chunked"``: the sequence is split into chunks; each chunk is
      scanned with an intra-chunk recurrence started from a zero state,
      while the carried-in state contributes through the chunk's cumulative
      decay product. Combining the two terms at chunk boundaries is the
      associative ( Blelloch-style / Mamba-2 chunked ) scan idea, kept here
      in a deliberately simple pure-PyTorch form.
    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.d_state = int(d_state)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.b_proj = nn.Linear(hidden_size, d_state)
        self.c_proj = nn.Linear(hidden_size, d_state)
        self.dt_proj = nn.Linear(hidden_size, 1)
        self.out_proj = nn.Linear(1, hidden_size)
        log_dt = torch.linspace(math.log(dt_min), math.log(dt_max), d_state)
        self.A = nn.Parameter(-torch.exp(log_dt))
        self.D = nn.Parameter(torch.ones(1))

    def _recurrent_scan(
        self,
        u: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step recurrence over the sequence.

        Args:
            u: Projected inputs, ``(batch, seq_len, hidden_size)``.
            b: Input-to-state vectors, ``(batch, seq_len, d_state)``.
            c: State-to-output vectors, ``(batch, seq_len, d_state)``.
            dt: Positive per-step discretization factors, ``(batch, seq_len)``.
            state: Initial state, ``(batch, d_state)``.

        Returns:
            ``(y, final_state)`` where ``y`` is the scalar-per-step sequence
            of shape ``(batch, seq_len)`` (before ``out_proj``) and
            ``final_state`` has shape ``(batch, d_state)``.
        """
        a_bar = torch.exp(self.A[None, None, :] * dt[..., None])
        outputs: list[torch.Tensor] = []
        for step in range(u.size(1)):
            state = a_bar[:, step] * state + b[:, step] * u[:, step].sum(
                dim=-1, keepdim=True
            )
            y = torch.einsum("bd,bd->b", c[:, step], state)
            y = y + self.D * u[:, step].sum(dim=-1)
            outputs.append(y)
        return torch.stack(outputs, dim=1), state

    def _chunked_scan(
        self,
        u: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked scan using the associativity of the linear recurrence.

        The recurrence ``state_t = a_t * state_{t-1} + b_t * u_t`` is
        associative, so within each chunk the output can be decomposed as
        ``c_t . (decay_t * state_in + local_state_t)`` where ``decay_t`` is
        the cumulative decay product from the chunk start to step ``t`` and
        ``local_state_t`` is the intra-chunk recurrence started from zero.
        At the chunk boundary the two terms are combined into the state
        carried into the next chunk. This teaching version still loops over
        steps inside each chunk; a production kernel would parallelize the
        intra-chunk part.

        Args:
            u: Projected inputs, ``(batch, seq_len, hidden_size)``.
            b: Input-to-state vectors, ``(batch, seq_len, d_state)``.
            c: State-to-output vectors, ``(batch, seq_len, d_state)``.
            dt: Positive per-step discretization factors, ``(batch, seq_len)``.
            state: Initial state, ``(batch, d_state)``.
            chunk_size: Number of time steps per chunk.

        Returns:
            Same contract as :meth:`_recurrent_scan`.
        """
        a_bar = torch.exp(self.A[None, None, :] * dt[..., None])
        seq_len = u.size(1)
        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            decay = torch.ones_like(state)
            local_state = torch.zeros_like(state)
            for step in range(start, end):
                decay = decay * a_bar[:, step]
                local_state = a_bar[:, step] * local_state + b[:, step] * u[
                    :, step
                ].sum(dim=-1, keepdim=True)
                y = torch.einsum("bd,bd->b", c[:, step], decay * state + local_state)
                y = y + self.D * u[:, step].sum(dim=-1)
                outputs.append(y)
            state = decay * state + local_state
        return torch.stack(outputs, dim=1), state

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        scan: str = "recurrent",
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SSM over the sequence dimension.

        Args:
            x: Input hidden states, ``(batch, seq_len, hidden_size)``.
            state: Optional incoming state, ``(batch, d_state)``.
            scan: ``"recurrent"`` (default, step-by-step) or ``"chunked"``
                (chunk-parallel decomposition; both compute the same math).
            chunk_size: Time steps per chunk when ``scan="chunked"``.

        Returns:
            ``(output, final_state)`` with ``output`` of shape
            ``(batch, seq_len, hidden_size)`` and ``final_state`` of shape
            ``(batch, d_state)``. Pass ``final_state`` back as ``state`` to
            continue the sequence in streaming/chunked decoding.
        """
        if scan not in ("recurrent", "chunked"):
            raise ValueError(f"scan must be 'recurrent' or 'chunked', got {scan!r}")
        batch_size, seq_len, _ = x.size()
        if state is None:
            state = torch.zeros(
                batch_size, self.d_state, device=x.device, dtype=x.dtype
            )
        if seq_len == 0:
            return x.new_zeros(batch_size, 0, self.hidden_size), state

        u = self.in_proj(x)
        b = self.b_proj(x)
        c = self.c_proj(x)
        dt = torch.sigmoid(self.dt_proj(x))[..., 0]

        if scan == "chunked":
            y, state = self._chunked_scan(u, b, c, dt, state, chunk_size)
        else:
            y, state = self._recurrent_scan(u, b, c, dt, state)
        return self.out_proj(y.unsqueeze(-1)), state

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, d_state={self.d_state}"
