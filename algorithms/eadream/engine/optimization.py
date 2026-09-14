"""Finite-safe FP32 AdamW construction for EADream."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class OptimizerBundle:
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    clip_norm: float
    parameters: tuple[nn.Parameter, ...] = field(default_factory=tuple, repr=False)
    declared_epsilon: float | None = None
    declared_weight_decay: float | None = None
    effective_betas: tuple[float, float] = field(
        default=(0.9, 0.999), init=False
    )
    effective_epsilon: float = field(default=1e-8, init=False)
    effective_weight_decay: float = field(default=0.01, init=False)

    def __post_init__(self) -> None:
        optimizer_parameters = tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        if not self.parameters:
            self.parameters = optimizer_parameters
        if not optimizer_parameters:
            raise ValueError("optimizer bundle requires at least one parameter")
        if len(self.parameters) != len(optimizer_parameters) or any(
            actual is not owned
            for actual, owned in zip(
                self.parameters, optimizer_parameters, strict=True
            )
        ):
            raise ValueError("bundle parameters must exactly match optimizer parameters")

    def step(self, loss: Tensor, parameters: Iterable[nn.Parameter]) -> float:
        supplied = tuple(parameters)
        if len(supplied) != len(self.parameters) or any(
            actual is not owned
            for actual, owned in zip(supplied, self.parameters, strict=True)
        ):
            raise ValueError("step parameters must exactly match the bundle's owned parameters")
        if not bool(torch.isfinite(loss).all()):
            raise FloatingPointError("non-finite optimization loss")
        self.optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
            for parameter in self.parameters:
                gradient = parameter.grad
                if gradient is None:
                    continue
                values = gradient.coalesce().values() if gradient.is_sparse else gradient
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("non-finite gradient")
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite gradient norm")
            self.optimizer.step()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        return float(norm.detach().cpu())


def make_adamw(
    parameters: Iterable[nn.Parameter],
    *,
    learning_rate: float,
    clip_norm: float,
    declared_epsilon: float | None = None,
    declared_weight_decay: float | None = None,
) -> OptimizerBundle:
    """Construct AdamW with executable defaults, not ignored config values."""
    owned_parameters = tuple(parameters)
    if not owned_parameters:
        raise ValueError("AdamW requires at least one parameter")
    optimizer = torch.optim.AdamW(
        owned_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return OptimizerBundle(
        optimizer=optimizer,
        scaler=scaler,
        clip_norm=clip_norm,
        parameters=owned_parameters,
        declared_epsilon=declared_epsilon,
        declared_weight_decay=declared_weight_decay,
    )
