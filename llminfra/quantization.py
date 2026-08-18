"""Quantization-aware training utilities for Transformer components.

The implementations in this module are portable PyTorch references. They
simulate low-precision numerics with straight-through estimators (STE), so
they are useful for architecture experiments and QAT unit tests. They do not
replace vendor FP8/INT8 kernels used by production inference engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.func import functional_call

QuantizationMode = Literal["int4", "int8", "fp8_e4m3"]


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for fake quantization.

    Args:
        mode: Target numerical format.
        per_channel: Compute one scale per channel instead of one scale for
            the entire tensor. This is most useful for weight tensors.
        channel_axis: Axis retained when ``per_channel=True``.
        quantize_weights: Fake-quantize floating-point parameters.
        quantize_inputs: Fake-quantize tensor inputs.
        quantize_outputs: Fake-quantize tensor outputs.
        eps: Lower bound for scale values.
    """

    mode: QuantizationMode = "int8"
    per_channel: bool = False
    channel_axis: int = 0
    quantize_weights: bool = True
    quantize_inputs: bool = True
    quantize_outputs: bool = True
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.mode not in {"int4", "int8", "fp8_e4m3"}:
            raise ValueError(f"Unsupported quantization mode: {self.mode!r}")
        if self.eps <= 0:
            raise ValueError("eps must be > 0")


class FakeQuantizer(nn.Module):
    """Apply differentiable fake quantization with an STE backward pass."""

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a fake-quantized tensor while preserving identity gradients."""
        if not tensor.is_floating_point() or tensor.numel() == 0:
            return tensor
        with torch.no_grad():
            if self.config.mode == "fp8_e4m3":
                quantized = self._fake_fp8_e4m3(tensor)
            else:
                bits = 4 if self.config.mode == "int4" else 8
                quantized = self._fake_symmetric_integer(tensor, bits)
        return tensor + (quantized - tensor).detach()

    def _reduction_dims(self, tensor: torch.Tensor) -> tuple[int, ...]:
        if not self.config.per_channel:
            return tuple(range(tensor.dim()))
        axis = self.config.channel_axis % tensor.dim()
        return tuple(dim for dim in range(tensor.dim()) if dim != axis)

    def _fake_symmetric_integer(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        qmax = 2 ** (bits - 1) - 1
        reduce_dims = self._reduction_dims(tensor)
        if reduce_dims:
            max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
        else:
            max_abs = tensor.detach().abs()
        scale = (max_abs / qmax).clamp_min(self.config.eps)
        return (tensor / scale).round().clamp(-qmax, qmax) * scale

    @staticmethod
    def _fake_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
        """Approximate finite E4M3 values without requiring FP8 hardware.

        E4M3 has three explicit mantissa bits and a maximum finite magnitude
        of 448. This reference rounds normal values to a power-of-two step
        selected by their exponent. Subnormal edge behavior is approximated.
        """
        max_finite = 448.0
        clamped = tensor.clamp(-max_finite, max_finite)
        magnitude = clamped.abs()
        safe = magnitude.clamp_min(torch.finfo(clamped.dtype).tiny)
        exponent = torch.floor(torch.log2(safe)).clamp(-6, 8)
        step = torch.pow(torch.full_like(exponent, 2.0), exponent - 3)
        rounded = (clamped / step).round() * step
        return torch.where(magnitude == 0, torch.zeros_like(rounded), rounded)

    def extra_repr(self) -> str:
        return (
            f"mode={self.config.mode!r}, "
            f"per_channel={self.config.per_channel}, "
            f"channel_axis={self.config.channel_axis}"
        )


class QATWrapper(nn.Module):
    """Wrap an arbitrary module with input, weight and output fake quantization.

    Parameters are supplied through :func:`torch.func.functional_call`, so the
    wrapped module is never mutated in-place and gradients still reach its
    original parameters. Nested tensor tuples/lists/dicts are supported for
    modules such as attention layers that optionally return weights.
    """

    def __init__(self, module: nn.Module, config: QuantizationConfig) -> None:
        super().__init__()
        self.module = module
        self.config = config
        self.activation_quantizer = FakeQuantizer(config)
        weight_config = QuantizationConfig(
            mode=config.mode,
            per_channel=True,
            channel_axis=config.channel_axis,
            quantize_weights=config.quantize_weights,
            quantize_inputs=config.quantize_inputs,
            quantize_outputs=config.quantize_outputs,
            eps=config.eps,
        )
        self.weight_quantizer = FakeQuantizer(weight_config)

    def _map_tensors(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return self.activation_quantizer(value)
        if isinstance(value, tuple):
            return tuple(self._map_tensors(item) for item in value)
        if isinstance(value, list):
            return [self._map_tensors(item) for item in value]
        if isinstance(value, dict):
            return {key: self._map_tensors(item) for key, item in value.items()}
        return value

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped module with the configured fake quantization."""
        call_args = self._map_tensors(args) if self.config.quantize_inputs else args
        call_kwargs = (
            self._map_tensors(kwargs) if self.config.quantize_inputs else kwargs
        )

        if self.config.quantize_weights:
            state: dict[str, torch.Tensor] = {}
            for name, parameter in self.module.named_parameters():
                state[name] = self.weight_quantizer(parameter)
            state.update(dict(self.module.named_buffers()))
            output = functional_call(self.module, state, call_args, call_kwargs)
        else:
            output = self.module(*call_args, **call_kwargs)

        if self.config.quantize_outputs:
            output = self._map_tensors(output)
        return output

    def extra_repr(self) -> str:
        return f"mode={self.config.mode!r}, module={type(self.module).__name__}"


def build_quantized(
    module: nn.Module,
    config: QuantizationConfig | None = None,
    *,
    mode: QuantizationMode = "int8",
    per_channel: bool = False,
    channel_axis: int = 0,
) -> QATWrapper:
    """Build a QAT wrapper around ``module``.

    Passing an explicit ``config`` takes precedence over the convenience
    keyword arguments.
    """
    resolved = config or QuantizationConfig(
        mode=mode,
        per_channel=per_channel,
        channel_axis=channel_axis,
    )
    return QATWrapper(module, resolved)
