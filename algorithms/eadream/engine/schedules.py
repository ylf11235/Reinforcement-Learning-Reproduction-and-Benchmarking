"""Checkpointable source-cadence schedules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Every:
    """Return elapsed interval counts while preserving fractional remainder."""

    def __init__(self, every: float | None) -> None:
        self._every = every
        self._last: float | int | None = None

    def __call__(self, step: int | float) -> int:
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count

    def state_dict(self) -> dict[str, float | int | None]:
        return {"_last": self._last}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._last = state["_last"]


class Once:
    """Return true exactly once, including across checkpoint restoration."""

    def __init__(self) -> None:
        self._once = True

    def __call__(self) -> bool:
        if self._once:
            self._once = False
            return True
        return False

    def state_dict(self) -> dict[str, bool]:
        return {"_once": self._once}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._once = bool(state["_once"])


class Until:
    """Remain enabled strictly before an optional end step."""

    def __init__(self, until: int | float | None) -> None:
        self._until = until

    def __call__(self, step: int | float) -> bool:
        if not self._until:
            return True
        return step < self._until
