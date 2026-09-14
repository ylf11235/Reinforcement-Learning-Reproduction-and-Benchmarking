"""Adaptive source-compatible loss harmonization."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Harmonizer(nn.Module):
    """Learn a positive loss divisor with the source regularizer."""

    def __init__(self) -> None:
        super().__init__()
        self.harmony_s = nn.Parameter(torch.zeros(()))

    def coefficient(self) -> Tensor:
        return torch.exp(self.harmony_s)

    def forward(self, x: Tensor) -> Tensor:
        harmony = self.coefficient()
        return x / harmony + torch.log(harmony + 1.0)
