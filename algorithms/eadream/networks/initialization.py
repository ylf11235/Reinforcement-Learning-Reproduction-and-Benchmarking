"""Parameter initializers used by the pinned EADream implementation."""

from __future__ import annotations

import math
from collections.abc import Callable

from torch import nn


_TRUNCATED_NORMAL_CORRECTION = 0.87962566103423978


def _fan_average(module: nn.Module) -> float | None:
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
        fan_out = module.out_features
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        kernel_area = math.prod(module.kernel_size)
        fan_in = kernel_area * module.in_channels
        fan_out = kernel_area * module.out_channels
    else:
        return None
    return (fan_in + fan_out) / 2.0


def _zero_bias(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


def source_weight_init(module: nn.Module) -> nn.Module:
    """Apply the source fan-average truncated-normal initializer in place."""
    fan_average = _fan_average(module)
    if fan_average is not None:
        std = math.sqrt(1.0 / fan_average) / _TRUNCATED_NORMAL_CORRECTION
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        _zero_bias(module)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        _zero_bias(module)
    return module


def source_uniform_init(given_scale: float) -> Callable[[nn.Module], nn.Module]:
    """Return the source fan-average uniform initializer for an output scale."""
    scale = float(given_scale)
    if scale < 0:
        raise ValueError("given_scale must be non-negative")

    def initialize(module: nn.Module) -> nn.Module:
        fan_average = _fan_average(module)
        if fan_average is not None:
            limit = math.sqrt(3.0 * scale / fan_average)
            nn.init.uniform_(module.weight, -limit, limit)
            _zero_bias(module)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            _zero_bias(module)
        return module

    return initialize


__all__ = ["source_uniform_init", "source_weight_init"]
