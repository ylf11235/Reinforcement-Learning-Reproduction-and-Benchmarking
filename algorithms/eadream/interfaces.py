"""Transport-safe records shared by EADream collection boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np


class PolicyObservationRequired(TypedDict):
    image: np.ndarray
    is_first: bool
    is_terminal: bool


class PolicyObservation(PolicyObservationRequired, total=False):
    event: np.ndarray


@dataclass(frozen=True)
class RecurrentStatePayload:
    logit: np.ndarray
    stoch: np.ndarray
    deter: np.ndarray
    previous_action: np.ndarray

    def __post_init__(self) -> None:
        validate_recurrent_payload(self)


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    collector_id: str
    environment_id: str
    episode_id: str
    episode_step: int
    observation: PolicyObservation
    state: RecurrentStatePayload | None
    episode_start: bool
    deterministic_actor: bool
    policy_version: int

    def __post_init__(self) -> None:
        _validate_routing_fields(self)
        _validate_observation(self.observation)
        if self.state is not None:
            if type(self.state) is not RecurrentStatePayload:
                raise TypeError("state must be a RecurrentStatePayload or None")
            validate_recurrent_payload(self.state)
        _require_bool("episode_start", self.episode_start)
        _require_bool("deterministic_actor", self.deterministic_actor)


@dataclass(frozen=True)
class PolicyResult:
    request_id: str
    collector_id: str
    environment_id: str
    episode_id: str
    episode_step: int
    action: np.ndarray
    state: RecurrentStatePayload
    diagnostics: dict[str, float]
    policy_version: int

    def __post_init__(self) -> None:
        _validate_routing_fields(self)
        if type(self.state) is not RecurrentStatePayload:
            raise TypeError("state must be a RecurrentStatePayload")
        validate_recurrent_payload(self.state)
        _validate_action(self.action, self.state.previous_action.shape[1])
        _validate_diagnostics(self.diagnostics)


class Transition(TypedDict):
    image: np.ndarray
    event: np.ndarray
    action: np.ndarray
    reward: np.float32
    discount: np.float32
    is_first: np.bool_
    is_terminal: np.bool_
    episode_id: str
    episode_step: np.int64
    collector_id: str
    environment_id: str
    policy_version: np.int64


@dataclass
class TrainingCounters:
    agent_interactions: int = 0
    ale_frames: int = 0
    reset_noop_frames: int = 0
    forced_start_frames: int = 0
    evaluation_interactions: int = 0
    gradient_updates: int = 0


def _require_nonempty_string(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_bool(name: str, value: object) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")


def _validate_routing_fields(value: PolicyRequest | PolicyResult) -> None:
    for name in ("request_id", "collector_id", "environment_id", "episode_id"):
        _require_nonempty_string(name, getattr(value, name))
    _require_nonnegative_integer("episode_step", value.episode_step)
    _require_nonnegative_integer("policy_version", value.policy_version)


def _validate_array(
    name: str,
    value: object,
    *,
    dtype: np.dtype | type,
    rank: int,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != dtype:
        raise TypeError(f"{name} must be a {np.dtype(dtype).name} NumPy array")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if any(size <= 0 for size in value.shape):
        raise ValueError(f"{name} dimensions must be positive")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return value


def _validate_observation(observation: object) -> None:
    if type(observation) is not dict:
        raise TypeError("observation must be a plain dict")
    required = {"image", "is_first", "is_terminal"}
    allowed = required | {"event"}
    keys = set(observation)
    if keys < required or not keys <= allowed:
        raise ValueError(
            "observation keys must be image, is_first, is_terminal, and optional event"
        )
    _validate_array(
        "observation.image",
        observation["image"],
        dtype=np.uint8,
        rank=3,
        shape=(64, 64, 3),
    )
    if "event" in observation:
        _validate_array(
            "observation.event",
            observation["event"],
            dtype=np.uint8,
            rank=2,
            shape=(64, 64),
        )
    _require_bool("observation.is_first", observation["is_first"])
    _require_bool("observation.is_terminal", observation["is_terminal"])


def _validate_action(action: object, action_count: int) -> None:
    value = _validate_array("action", action, dtype=np.float32, rank=1)
    if value.shape != (action_count,):
        raise ValueError(
            f"action must have shape ({action_count},), got {value.shape}"
        )
    if np.count_nonzero(value) != 1 or np.count_nonzero(value == 1.0) != 1:
        raise ValueError("action must be one-hot")


def _validate_diagnostics(diagnostics: object) -> None:
    if type(diagnostics) is not dict:
        raise TypeError("diagnostics must be a plain dict")
    for name, value in diagnostics.items():
        if type(name) is not str or not name.strip():
            raise TypeError("diagnostics keys must be non-empty strings")
        if type(value) is not float or not np.isfinite(value):
            raise TypeError("diagnostics values must be finite floats")


def validate_recurrent_payload(
    payload: RecurrentStatePayload,
) -> RecurrentStatePayload:
    if type(payload) is not RecurrentStatePayload:
        raise TypeError("payload must be a RecurrentStatePayload")
    arrays = {
        "logit": (payload.logit, 3),
        "stoch": (payload.stoch, 3),
        "deter": (payload.deter, 2),
        "previous_action": (payload.previous_action, 2),
    }
    for name, (value, rank) in arrays.items():
        if type(value) is not np.ndarray or value.dtype != np.float32:
            raise TypeError(f"{name} must be a float32 NumPy array")
        if value.ndim != rank or not value.flags.c_contiguous:
            raise ValueError(f"{name} has invalid rank or is not C-contiguous")
        if any(size <= 0 for size in value.shape):
            raise ValueError(f"{name} dimensions must be positive")
    if payload.logit.shape != payload.stoch.shape:
        raise ValueError("logit and stoch shapes differ")
    batch = payload.logit.shape[0]
    if (
        payload.deter.shape[0] != batch
        or payload.previous_action.shape[0] != batch
    ):
        raise ValueError("recurrent payload batch dimensions differ")
    return payload
