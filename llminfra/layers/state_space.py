"""Selective state-space layers for Mamba-style hybrid models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class Mamba2State:
    """Streaming state carried between :class:`Mamba2Layer` calls.

    Attributes:
        ssm: Per-channel selective state shaped
            ``(batch, d_inner, d_state)``.
        convolution: Previous unprojected convolution inputs shaped
            ``(batch, d_inner, conv_kernel - 1)``.

    """

    ssm: torch.Tensor
    convolution: torch.Tensor


class Mamba2Layer(nn.Module):
    """Pure-PyTorch selective diagonal SSM with Mamba-2-style structure.

    The layer keeps an independent ``d_state`` state vector for every inner
    channel, uses input-dependent B/C/dt parameters, applies a causal
    depthwise convolution, and supports recurrent or chunked scans. The two
    scan modes evaluate the same recurrence and support exact streaming by
    passing :class:`Mamba2State` between calls.

    This is a numerically explicit reference implementation. It omits SSD
    tensor-core tiling, head/group parameter sharing, fused scan kernels, and
    the initialization details required to reproduce an official checkpoint.

    Args:
        hidden_size: Input and output feature dimension.
        d_state: State dimension maintained per inner channel.
        d_inner: Expanded channel dimension. Defaults to ``hidden_size``.
        conv_kernel: Causal depthwise convolution width.
        dt_min: Minimum selective discretization step.
        dt_max: Maximum selective discretization step.
        bias: Enable biases on projections and the causal convolution.

    """

    def __init__(
        self,
        hidden_size: int,
        d_state: int = 16,
        d_inner: int | None = None,
        conv_kernel: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or d_state < 1:
            raise ValueError("hidden_size and d_state must be >= 1")
        if d_inner is not None and d_inner < 1:
            raise ValueError("d_inner must be >= 1")
        if conv_kernel < 1:
            raise ValueError("conv_kernel must be >= 1")
        if not 0 < dt_min <= dt_max:
            raise ValueError("dt bounds must satisfy 0 < dt_min <= dt_max")

        self.hidden_size = int(hidden_size)
        self.d_state = int(d_state)
        self.d_inner = int(d_inner or hidden_size)
        self.conv_kernel = int(conv_kernel)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

        self.in_proj = nn.Linear(hidden_size, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.conv_kernel,
            groups=self.d_inner,
            bias=bias,
        )
        self.b_proj = nn.Linear(hidden_size, d_state, bias=False)
        self.c_proj = nn.Linear(hidden_size, d_state, bias=False)
        self.dt_proj = nn.Linear(hidden_size, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=bias)

        rates = torch.linspace(dt_min, dt_max, d_state)
        self.A_log = nn.Parameter(rates.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        nn.init.constant_(
            self.dt_proj.bias,
            math.log(math.expm1((dt_min + dt_max) / 2)),
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Mamba2State:
        """Allocate a zero streaming state for a batch."""
        return Mamba2State(
            ssm=torch.zeros(
                batch_size,
                self.d_inner,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
            convolution=torch.zeros(
                batch_size,
                self.d_inner,
                self.conv_kernel - 1,
                device=device,
                dtype=dtype,
            ),
        )

    def _validate_state(
        self,
        state: Mamba2State,
        batch_size: int,
    ) -> None:
        expected_ssm = (batch_size, self.d_inner, self.d_state)
        expected_conv = (batch_size, self.d_inner, self.conv_kernel - 1)
        if tuple(state.ssm.shape) != expected_ssm:
            raise ValueError(
                f"state.ssm must have shape {expected_ssm}, "
                f"got {tuple(state.ssm.shape)}"
            )
        if tuple(state.convolution.shape) != expected_conv:
            raise ValueError(
                f"state.convolution must have shape {expected_conv}, "
                f"got {tuple(state.convolution.shape)}"
            )

    def _causal_convolution(
        self,
        u: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply depthwise causal convolution and update its streaming state."""
        channel_first = u.transpose(1, 2)
        history = torch.cat([state, channel_first], dim=-1)
        convolved = F.conv1d(
            history,
            self.conv1d.weight,
            self.conv1d.bias,
            groups=self.d_inner,
        )
        next_state = history[..., -(self.conv_kernel - 1) :]
        if self.conv_kernel == 1:
            next_state = history[..., :0]
        return F.silu(convolved.transpose(1, 2)), next_state

    def _discretize(
        self,
        u: torch.Tensor,
        b: torch.Tensor,
        dt: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize one time step of diagonal A and input-dependent B.

        Applies the zero-order hold for a single step so the scans never
        materialize ``(batch, seq, d_inner, d_state)`` tensors: ``u``/``dt``
        are shaped ``(batch, d_inner)``, ``b`` is ``(batch, d_state)`` and
        ``a`` is the full ``(d_inner, d_state)`` diagonal matrix. Both
        returned tensors are shaped ``(batch, d_inner, d_state)``.
        """
        a_bar = torch.exp(dt[:, :, None] * a)
        b_bar = dt[:, :, None] * b[:, None, :] * u[:, :, None]
        return a_bar, b_bar

    def _recurrent_scan(
        self,
        u: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the selective recurrence one time step at a time."""
        a = -torch.exp(self.A_log).to(dtype=u.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(u.size(1)):
            a_bar, b_bar = self._discretize(u[:, step], b[:, step], dt[:, step], a)
            state = a_bar * state + b_bar
            y = torch.einsum("bin,bn->bi", state, c[:, step])
            outputs.append(y + self.D * u[:, step])
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
        """Evaluate the same recurrence with associative chunk boundaries."""
        a = -torch.exp(self.A_log).to(dtype=u.dtype)
        outputs: list[torch.Tensor] = []
        for start in range(0, u.size(1), chunk_size):
            end = min(start + chunk_size, u.size(1))
            decay = torch.ones_like(state)
            local_state = torch.zeros_like(state)
            for step in range(start, end):
                a_bar, b_bar = self._discretize(u[:, step], b[:, step], dt[:, step], a)
                decay = decay * a_bar
                local_state = a_bar * local_state + b_bar
                step_state = decay * state + local_state
                y = torch.einsum("bin,bn->bi", step_state, c[:, step])
                outputs.append(y + self.D * u[:, step])
            state = decay * state + local_state
        return torch.stack(outputs, dim=1), state

    def forward(
        self,
        x: torch.Tensor,
        state: Mamba2State | None = None,
        scan: str = "recurrent",
        chunk_size: int = 16,
    ) -> tuple[torch.Tensor, Mamba2State]:
        """Run the selective SSM and return output plus streaming state.

        Args:
            x: Hidden states shaped ``(batch, seq_len, hidden_size)``.
            state: Optional state returned by a previous call.
            scan: ``"recurrent"`` or mathematically equivalent ``"chunked"``.
            chunk_size: Positive chunk width for ``scan="chunked"``.

        """
        if x.dim() != 3 or x.size(-1) != self.hidden_size:
            raise ValueError(f"x must have shape (batch, seq, {self.hidden_size})")
        if scan not in {"recurrent", "chunked"}:
            raise ValueError(f"scan must be 'recurrent' or 'chunked', got {scan!r}")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.initial_state(
                batch_size,
                device=x.device,
                dtype=x.dtype,
            )
        self._validate_state(state, batch_size)
        if seq_len == 0:
            return x.new_zeros(batch_size, 0, self.hidden_size), state

        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u, convolution_state = self._causal_convolution(u, state.convolution)
        b = self.b_proj(x)
        c = self.c_proj(x)
        dt = F.softplus(self.dt_proj(x)).clamp(self.dt_min, self.dt_max)

        if scan == "chunked":
            y, ssm_state = self._chunked_scan(u, b, c, dt, state.ssm, chunk_size)
        else:
            y, ssm_state = self._recurrent_scan(u, b, c, dt, state.ssm)
        output = self.out_proj(y * F.silu(gate))
        return output, Mamba2State(ssm_state, convolution_state)

    def extra_repr(self) -> str:
        """Show hidden/inner/state dims and conv kernel in ``repr(self)``."""
        return (
            f"hidden_size={self.hidden_size}, d_inner={self.d_inner}, "
            f"d_state={self.d_state}, conv_kernel={self.conv_kernel}"
        )
