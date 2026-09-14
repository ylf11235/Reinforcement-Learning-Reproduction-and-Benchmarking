"""Scalar transforms, returns, and percentile normalization for EADream."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Apply the signed logarithmic transform used by reward/value heads."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Invert :func:`symlog`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def lambda_return(
    reward: Tensor,
    value: Tensor,
    discount: Tensor,
    bootstrap: Tensor,
    lambda_: float,
) -> Tensor:
    """Compute source-order TD(lambda) returns along the leading time axis."""
    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + discount * next_values * (1.0 - lambda_)
    result: list[Tensor] = []
    aggregate = bootstrap
    for index in range(len(inputs) - 1, -1, -1):
        aggregate = inputs[index] + discount[index] * lambda_ * aggregate
        result.append(aggregate)
    return torch.stack(list(reversed(result)), dim=0)


class RewardEMA:
    """Track source 5th/95th return percentiles for diagnostic normalization."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        alpha: float = 0.01,
    ) -> None:
        self.alpha = float(alpha)
        self.values = torch.zeros(2, device=device, dtype=torch.float32)

    def update(self, x: Tensor) -> tuple[Tensor, Tensor]:
        quantiles = torch.tensor((0.05, 0.95), device=x.device, dtype=x.dtype)
        current = torch.quantile(x.detach(), quantiles)
        if self.values.device != current.device or self.values.dtype != current.dtype:
            self.values = self.values.to(device=current.device, dtype=current.dtype)
        self.values = self.alpha * current + (1.0 - self.alpha) * self.values
        scale = torch.clamp(self.values[1] - self.values[0], min=1.0)
        return self.values[0].detach(), scale.detach()

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.update(x)

    def state_dict(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "values": self.values.detach().clone()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        alpha = float(state["alpha"])
        values = state["values"]
        if not isinstance(values, Tensor) or values.shape != (2,):
            raise ValueError("RewardEMA values must be a two-element tensor")
        self.alpha = alpha
        self.values = values.detach().clone().to(device=self.values.device)
