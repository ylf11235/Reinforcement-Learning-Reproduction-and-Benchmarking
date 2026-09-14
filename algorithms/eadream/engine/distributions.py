"""Source-compatible distributions and pixel objectives for EADream."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import OneHotCategorical
from torch.nn import functional as F

from .math import symlog, symexp


class OneHotDist(OneHotCategorical):
    """Categorical distribution with source unimix and straight-through values."""

    def __init__(
        self,
        logits: Tensor | None = None,
        probs: Tensor | None = None,
        *,
        unimix_ratio: float = 0.01,
    ) -> None:
        if (logits is None) == (probs is None):
            raise ValueError("exactly one of logits or probs must be provided")
        categorical_probs = (
            torch.softmax(logits, dim=-1) if logits is not None else probs
        )
        assert categorical_probs is not None
        categories = categorical_probs.shape[-1]
        categorical_probs = (
            (1.0 - unimix_ratio) * categorical_probs
            + unimix_ratio / categories
        )
        super().__init__(probs=categorical_probs)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        sample = super().sample(sample_shape)
        probs = self.probs
        return sample + (probs - probs.detach())

    def mode(self) -> Tensor:
        mode = F.one_hot(self.logits.argmax(dim=-1), self.logits.shape[-1]).to(
            self.logits.dtype
        )
        return mode.detach() + self.logits - self.logits.detach()


class SymlogTwoHotDist:
    """A 255-bin symlog categorical with linearly interpolated targets."""

    def __init__(
        self,
        logits: Tensor,
        *,
        low: float = -20.0,
        high: float = 20.0,
        bins: int = 255,
    ) -> None:
        if logits.shape[-1] != bins:
            raise ValueError(f"last logits dimension must be {bins}")
        self.logits = logits
        self.probs = torch.softmax(logits, dim=-1)
        self.buckets = torch.linspace(
            low, high, bins, device=logits.device, dtype=logits.dtype
        )
        self.low = float(low)
        self.high = float(high)
        self.bins = int(bins)

    def log_prob(self, value: Tensor) -> Tensor:
        transformed = symlog(value).clamp(self.low, self.high)
        position = (transformed - self.low) / (self.high - self.low) * (
            self.bins - 1
        )
        below = torch.floor(position).to(torch.long)
        above = torch.ceil(position).to(torch.long)
        above_weight = position - below.to(position.dtype)
        below_weight = 1.0 - above_weight
        target = F.one_hot(below, self.bins).to(self.logits.dtype) * below_weight[
            ..., None
        ]
        target = target + F.one_hot(above, self.bins).to(
            self.logits.dtype
        ) * above_weight[..., None]
        return (target * torch.log_softmax(self.logits, dim=-1)).sum(dim=-1)

    def mean(self) -> Tensor:
        return symexp((self.probs * self.buckets).sum(dim=-1))

    def mode(self) -> Tensor:
        return self.mean()

    def entropy(self) -> Tensor:
        log_probs = torch.log_softmax(self.logits, dim=-1)
        return -(self.probs * log_probs).sum(dim=-1)


class FocalBinaryDist:
    """Independent focal-weighted binary probabilities over image events."""

    def __init__(
        self,
        prediction: Tensor,
        *,
        alpha: float = 0.15,
        gamma: float = 4.0,
    ) -> None:
        self.prediction = prediction
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def loss(self, target: Tensor) -> Tensor:
        bce = F.binary_cross_entropy(self.prediction, target, reduction="none")
        p_t = self.prediction * target + (1.0 - self.prediction) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return focal.sum(dim=tuple(range(2, focal.ndim)))

    def log_prob(self, target: Tensor) -> Tensor:
        return -self.loss(target)

    def mean(self) -> Tensor:
        return self.prediction

    def mode(self) -> Tensor:
        return (self.prediction >= 0.5).to(self.prediction.dtype)


class EventAwareImageLoss:
    """GES-gated event-attended image squared error."""

    def __init__(self, *, attention_weight: float, event_pred_ratio: float) -> None:
        self.attention_weight = float(attention_weight)
        self.event_pred_ratio = float(event_pred_ratio)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        event: Tensor,
    ) -> Tensor:
        event_pixels = event.sum(dim=tuple(range(2, event.ndim)))
        pixel_count = math.prod(event.shape[2:])
        base = (prediction - target).square()
        attended = (
            (1.0 - self.attention_weight) * base
            + self.attention_weight * event * base
        )
        base = base.sum(dim=tuple(range(2, base.ndim)))
        attended = attended.sum(dim=tuple(range(2, attended.ndim)))
        return torch.where(
            event_pixels < self.event_pred_ratio * pixel_count,
            attended,
            base,
        )
