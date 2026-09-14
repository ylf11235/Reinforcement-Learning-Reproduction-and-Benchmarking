"""Reward, continuation, value, and actor heads for native EADream."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from eadream.engine.distributions import OneHotDist, SymlogTwoHotDist

from .initialization import source_uniform_init, source_weight_init


class _HeadBase(nn.Module):
    def __init__(self, input_dim: int, units: int, layers: int) -> None:
        super().__init__()
        if input_dim <= 0 or units <= 0 or layers < 0:
            raise ValueError("head dimensions must be positive and layers non-negative")
        modules: list[nn.Module] = []
        self.hidden_linears: list[nn.Linear] = []
        self.hidden_norms: list[nn.LayerNorm] = []
        current = input_dim
        for _ in range(layers):
            linear = nn.Linear(current, units, bias=False)
            norm = nn.LayerNorm(units, eps=1e-3)
            self.hidden_linears.append(linear)
            self.hidden_norms.append(norm)
            modules.extend((linear, norm, nn.SiLU()))
            current = units
        self.trunk = nn.Sequential(*modules)
        self.trunk.apply(source_weight_init)
        self.trunk_output_dim = current


class ScalarHead(_HeadBase):
    """Distribution head used for reward/value or binary continuation."""

    def __init__(
        self,
        input_dim: int,
        *,
        units: int = 512,
        layers: int = 2,
        distribution: str = "symlog_disc",
        bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
        outscale: float | None = None,
    ) -> None:
        super().__init__(input_dim, units, layers)
        if distribution not in ("symlog_disc", "binary"):
            raise ValueError("distribution must be 'symlog_disc' or 'binary'")
        self.distribution = distribution
        self.bins = int(bins)
        self.low = float(low)
        self.high = float(high)
        output_dim = self.bins if distribution == "symlog_disc" else 1
        if outscale is None:
            outscale = 0.0 if distribution == "symlog_disc" else 1.0
        self.output = nn.Linear(self.trunk_output_dim, output_dim)
        self.output.apply(source_uniform_init(outscale))

    def forward(self, features: Tensor):
        logits = self.output(self.trunk(features))
        if self.distribution == "symlog_disc":
            return SymlogTwoHotDist(
                logits, low=self.low, high=self.high, bins=self.bins
            )
        return torch.distributions.Independent(
            torch.distributions.Bernoulli(logits=logits), 1
        )


class ActorHead(_HeadBase):
    """Minimal-action-set categorical policy head."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        units: int = 512,
        layers: int = 2,
        unimix: float = 0.01,
        outscale: float = 1.0,
    ) -> None:
        super().__init__(input_dim, units, layers)
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.action_dim = int(action_dim)
        self.unimix = float(unimix)
        self.output = nn.Linear(self.trunk_output_dim, self.action_dim)
        self.output.apply(source_uniform_init(outscale))

    def forward(self, features: Tensor) -> OneHotDist:
        return OneHotDist(
            logits=self.output(self.trunk(features)), unimix_ratio=self.unimix
        )


__all__ = ["ActorHead", "ScalarHead"]
