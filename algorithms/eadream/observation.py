"""Strict structured observation contract shared by policy and collection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


_KEYS = frozenset({"image", "event", "is_first", "is_terminal"})


def validate_observation(observation: Mapping[str, Any], *, copy: bool = True) -> dict[str, Any]:
    if type(observation) is not dict:
        raise TypeError("observation must be a plain dict")
    if set(observation) != _KEYS:
        raise ValueError("observation keys must be exactly image, event, is_first, is_terminal")
    image = observation["image"]
    event = observation["event"]
    if type(image) is not np.ndarray or image.dtype != np.uint8:
        raise TypeError("observation.image must be a uint8 NumPy array")
    if image.shape != (64, 64, 3) or not image.flags.c_contiguous:
        raise ValueError("observation.image must be C-contiguous with shape (64, 64, 3)")
    if type(event) is not np.ndarray or event.dtype != np.uint8:
        raise TypeError("observation.event must be a uint8 NumPy array")
    if event.shape != (64, 64) or not event.flags.c_contiguous:
        raise ValueError("observation.event must be C-contiguous with shape (64, 64)")
    if type(observation["is_first"]) is not bool or type(observation["is_terminal"]) is not bool:
        raise TypeError("observation flags must be plain bool values")
    if copy:
        image = image.copy()
        event = event.copy()
    return {"image": image, "event": event, "is_first": observation["is_first"], "is_terminal": observation["is_terminal"]}


__all__ = ["validate_observation"]
