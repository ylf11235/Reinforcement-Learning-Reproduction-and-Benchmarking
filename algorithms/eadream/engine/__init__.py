"""Core math, distributions, schedules, and optimization for EADream."""

from .distributions import (
    EventAwareImageLoss,
    FocalBinaryDist,
    OneHotDist,
    SymlogTwoHotDist,
)
from .math import RewardEMA, lambda_return, symlog, symexp
from .optimization import OptimizerBundle, make_adamw
from .schedules import Every, Once, Until

__all__ = [
    "EventAwareImageLoss",
    "Every",
    "FocalBinaryDist",
    "Once",
    "OneHotDist",
    "OptimizerBundle",
    "RewardEMA",
    "SymlogTwoHotDist",
    "Until",
    "lambda_return",
    "make_adamw",
    "symlog",
    "symexp",
]
