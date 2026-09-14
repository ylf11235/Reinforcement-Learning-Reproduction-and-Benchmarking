"""Validated construction of immutable EADream replay episodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np

from eadream.interfaces import Transition


REQUIRED_FIELDS = (
    "image",
    "event",
    "action",
    "reward",
    "discount",
    "is_first",
    "is_terminal",
    "episode_id",
    "episode_step",
    "collector_id",
    "environment_id",
    "policy_version",
)

FinishReason = Literal["terminated", "truncated", "interrupted"]


def _require_array(
    episode: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype | type | None = None,
    rank: int,
    trailing_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    value = episode[name]
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a plain NumPy array")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype).name}")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if trailing_shape is not None and value.shape[1:] != trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}, got {value.shape[1:]}"
        )
    return value


def validate_episode(
    episode: Mapping[str, np.ndarray], *, action_dim: int | None = None
) -> Mapping[str, np.ndarray]:
    """Validate the complete array and alignment contract of one episode."""
    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    if set(episode) != set(REQUIRED_FIELDS):
        missing = set(REQUIRED_FIELDS) - set(episode)
        extra = set(episode) - set(REQUIRED_FIELDS)
        raise ValueError(
            f"episode fields or order differ; missing={sorted(missing)}, extra={sorted(extra)}"
        )

    image = _require_array(
        episode, "image", dtype=np.uint8, rank=4, trailing_shape=(64, 64, 3)
    )
    length = image.shape[0]
    if length < 1:
        raise ValueError("episode must contain at least one item")
    _require_array(
        episode, "event", dtype=np.uint8, rank=3, trailing_shape=(64, 64)
    )
    action = _require_array(episode, "action", dtype=np.float32, rank=2)
    if action_dim is not None:
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        if action.shape[1] != action_dim:
            raise ValueError(
                f"action must have trailing shape ({action_dim},), got {action.shape[1:]}"
            )
    elif action.shape[1] < 1:
        raise ValueError("action dimension must be positive")

    for name in ("reward", "discount"):
        value = _require_array(episode, name, dtype=np.float32, rank=1)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if ((episode["discount"] < 0.0) | (episode["discount"] > 1.0)).any():
        raise ValueError("discount values must be in [0, 1]")
    for name in ("is_first", "is_terminal"):
        _require_array(episode, name, dtype=np.bool_, rank=1)
    for name in ("episode_step", "policy_version"):
        _require_array(episode, name, dtype=np.int64, rank=1)
    for name in ("episode_id", "collector_id", "environment_id"):
        value = _require_array(episode, name, rank=1)
        if value.dtype.kind != "U":
            raise TypeError(f"{name} must use a fixed-width Unicode dtype")
        if any(not str(item).strip() for item in value.tolist()):
            raise ValueError(f"{name} must contain non-empty strings")

    for name in REQUIRED_FIELDS:
        if episode[name].shape[0] != length:
            raise ValueError(
                f"all episode fields must have equal time length; {name} has "
                f"{episode[name].shape[0]}, expected {length}"
            )
    if not bool(episode["is_first"][0]) or episode["is_first"][1:].any():
        raise ValueError("an episode must have is_first only at index zero")
    if episode["is_terminal"][:-1].any():
        raise ValueError("is_terminal may be true only on the final item")
    if not np.array_equal(episode["episode_step"], np.arange(length, dtype=np.int64)):
        raise ValueError("episode_step must be contiguous from zero")
    for name in ("episode_id", "collector_id", "environment_id"):
        if np.any(episode[name] != episode[name][0]):
            raise ValueError(f"{name} must be constant within an episode")
    if (episode["policy_version"] < 0).any():
        raise ValueError("policy_version must be non-negative")
    return episode


def validate_replay_episode(
    episode: Mapping[str, np.ndarray], *, completed: bool
) -> Mapping[str, np.ndarray]:
    """Validate semantics required of builder-produced persistent replay."""
    validate_episode(episode)
    if episode["action"][0].any() or float(episode["reward"][0]) != 0.0:
        raise ValueError("initial action and reward must be zero sentinels")

    actions = episode["action"][1:]
    if len(actions):
        nonzero = np.count_nonzero(actions, axis=1)
        one = np.count_nonzero(actions == 1.0, axis=1)
        if np.any(nonzero != 1) or np.any(one != 1):
            raise ValueError("every action after the sentinel must be one-hot")

    discounts = episode["discount"]
    if np.any(discounts[:-1] != 1.0):
        raise ValueError("only a completed final boundary may change discount")
    terminal = bool(episode["is_terminal"][-1])
    final_discount = float(discounts[-1])
    if completed:
        if len(discounts) == 1 and terminal:
            raise ValueError("a sentinel-only episode cannot be terminal")
        valid_boundary = (
            terminal and final_discount in (0.0, 1.0)
        ) or (not terminal and final_discount == 1.0)
        if not valid_boundary:
            raise ValueError("completed episode has an impossible finish boundary")
    elif terminal or np.any(discounts != 1.0):
        raise ValueError("live episode must have a nonterminal unit-discount tail")
    return episode


def _plain_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _copy_observation_array(
    observation: Mapping[str, object],
    name: str,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    value = observation[name]
    if type(value) is not np.ndarray or value.dtype != np.uint8:
        raise TypeError(f"observation.{name} must be a uint8 NumPy array")
    if value.shape != shape:
        raise ValueError(f"observation.{name} must have shape {shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"observation.{name} must be C-contiguous")
    return value.copy()


class EpisodeBuilder:
    """Build one aligned episode while retaining no worker or IPC dependency."""

    def __init__(
        self,
        episode_id: str,
        action_dim: int,
        collector_id: str = "collector-0",
        environment_id: str = "env-0",
    ) -> None:
        self.episode_id = _plain_string("episode_id", episode_id)
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        self.action_dim = action_dim
        self.collector_id = _plain_string("collector_id", collector_id)
        self.environment_id = _plain_string("environment_id", environment_id)
        self._columns: dict[str, list[object]] = {
            name: [] for name in REQUIRED_FIELDS
        }
        self._started = False
        self._finished = False
        self._finish_reason: FinishReason | None = None

    @property
    def is_finished(self) -> bool:
        return self._finished

    def start(
        self, observation: Mapping[str, object], *, policy_version: int
    ) -> None:
        if self._started:
            raise RuntimeError("episode has already started")
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if set(observation) != {"image", "event", "is_first", "is_terminal"}:
            raise ValueError("observation must contain image, event, is_first, is_terminal")
        if not isinstance(observation["is_first"], (bool, np.bool_)):
            raise TypeError("observation.is_first must be boolean")
        if not isinstance(observation["is_terminal"], (bool, np.bool_)):
            raise TypeError("observation.is_terminal must be boolean")
        if not bool(observation["is_first"]):
            raise ValueError("initial observation must have is_first=true")
        if bool(observation["is_terminal"]):
            raise ValueError("initial observation must not be terminal")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")

        row = {
            "image": _copy_observation_array(
                observation, "image", shape=(64, 64, 3)
            ),
            "event": _copy_observation_array(observation, "event", shape=(64, 64)),
            "action": np.zeros(self.action_dim, dtype=np.float32),
            "reward": np.float32(0.0),
            "discount": np.float32(1.0),
            "is_first": np.bool_(True),
            "is_terminal": np.bool_(False),
            "episode_id": self.episode_id,
            "episode_step": np.int64(0),
            "collector_id": self.collector_id,
            "environment_id": self.environment_id,
            "policy_version": np.int64(policy_version),
        }
        self._append_row(row)
        self._started = True

    def append(self, transition: Transition) -> None:
        if not self._started:
            raise RuntimeError("episode has not started")
        if self._finished:
            raise RuntimeError("episode is already finished")
        if bool(self._columns["is_terminal"][-1]):
            raise RuntimeError("episode already has a terminal transition")
        if type(transition) is not dict or set(transition) != set(REQUIRED_FIELDS):
            raise ValueError("transition must be a plain dict with all replay fields")

        expected_step = len(self._columns["reward"])
        for name in ("episode_id", "collector_id", "environment_id"):
            if type(transition[name]) is not str:
                raise TypeError(f"transition {name} must be a string")
        if transition["episode_id"] != self.episode_id:
            raise ValueError("transition episode_id does not match builder")
        if transition["collector_id"] != self.collector_id:
            raise ValueError("transition collector_id does not match builder")
        if transition["environment_id"] != self.environment_id:
            raise ValueError("transition environment_id does not match builder")
        if type(transition["episode_step"]) is not np.int64:
            raise TypeError("transition episode_step must be np.int64")
        if int(transition["episode_step"]) != expected_step:
            raise ValueError(f"transition episode_step must be {expected_step}")
        if type(transition["policy_version"]) is not np.int64:
            raise TypeError("transition policy_version must be np.int64")
        if int(transition["policy_version"]) < 0:
            raise ValueError("transition policy_version must be non-negative")
        if type(transition["reward"]) is not np.float32:
            raise TypeError("transition reward must be np.float32")
        if type(transition["discount"]) is not np.float32:
            raise TypeError("transition discount must be np.float32")
        if not np.isfinite(transition["reward"]):
            raise ValueError("transition reward must be finite")
        if not 0.0 <= float(transition["discount"]) <= 1.0:
            raise ValueError("transition discount must be in [0, 1]")
        for name in ("is_first", "is_terminal"):
            if type(transition[name]) is not np.bool_:
                raise TypeError(f"transition {name} must be np.bool_")
        if bool(transition["is_first"]):
            raise ValueError("appended transition must not have is_first=true")

        image = transition["image"]
        event = transition["event"]
        action = transition["action"]
        if type(image) is not np.ndarray or image.dtype != np.uint8:
            raise TypeError("transition image must be a uint8 NumPy array")
        if image.shape != (64, 64, 3) or not image.flags.c_contiguous:
            raise ValueError("transition image must be C-contiguous with shape (64, 64, 3)")
        if type(event) is not np.ndarray or event.dtype != np.uint8:
            raise TypeError("transition event must be a uint8 NumPy array")
        if event.shape != (64, 64) or not event.flags.c_contiguous:
            raise ValueError("transition event must be C-contiguous with shape (64, 64)")
        if type(action) is not np.ndarray or action.dtype != np.float32:
            raise TypeError("transition action must be a float32 NumPy array")
        if action.shape != (self.action_dim,) or not action.flags.c_contiguous:
            raise ValueError(
                f"transition action must be C-contiguous with shape ({self.action_dim},)"
            )
        if np.count_nonzero(action) != 1 or np.count_nonzero(action == 1.0) != 1:
            raise ValueError("transition action must be one-hot")

        row = {name: transition[name] for name in REQUIRED_FIELDS}
        row["image"] = image.copy()
        row["event"] = event.copy()
        row["action"] = action.copy()
        self._append_row(row)

    def finish(self, reason: FinishReason) -> dict[str, np.ndarray]:
        if not self._started:
            raise RuntimeError("episode has not started")
        if self._finished:
            raise RuntimeError("episode is already finished")
        if reason not in ("terminated", "truncated", "interrupted"):
            raise ValueError("invalid episode finish reason")
        episode = self.snapshot()
        terminal = bool(episode["is_terminal"][-1])
        discount = float(episode["discount"][-1])
        if reason == "terminated" and (not terminal or discount != 0.0):
            raise ValueError("terminated episode must end terminal with zero discount")
        if reason == "truncated" and (not terminal or discount != 1.0):
            raise ValueError("truncated episode must end terminal with unit discount")
        if reason == "interrupted" and (terminal or discount != 1.0):
            raise ValueError(
                "interrupted episode must preserve a nonterminal unit-discount tail"
            )
        self._finished = True
        self._finish_reason = reason
        return episode

    def snapshot(self) -> dict[str, np.ndarray]:
        if not self._started:
            raise RuntimeError("episode has not started")
        episode: dict[str, np.ndarray] = {}
        for key, values in self._columns.items():
            array = np.ascontiguousarray(np.stack(values))
            array.setflags(write=False)
            episode[key] = array
        validate_episode(episode, action_dim=self.action_dim)
        return episode

    def state_dict(self) -> dict[str, object]:
        episode = None
        if self._started:
            episode = {key: value.copy() for key, value in self.snapshot().items()}
        return {
            "format_version": 1,
            "episode_id": self.episode_id,
            "action_dim": self.action_dim,
            "collector_id": self.collector_id,
            "environment_id": self.environment_id,
            "started": self._started,
            "finished": self._finished,
            "finish_reason": self._finish_reason,
            "episode": episode,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            not isinstance(state, Mapping)
            or type(state.get("format_version")) is not int
            or state.get("format_version") != 1
        ):
            raise ValueError("unsupported episode builder state")
        for name in ("episode_id", "collector_id", "environment_id"):
            if type(state.get(name)) is not str:
                raise TypeError(f"episode builder state {name} must be a string")
        if type(state.get("action_dim")) is not int:
            raise TypeError("episode builder state action_dim must be an integer")
        expected = (
            self.episode_id,
            self.action_dim,
            self.collector_id,
            self.environment_id,
        )
        actual = (
            state.get("episode_id"),
            state.get("action_dim"),
            state.get("collector_id"),
            state.get("environment_id"),
        )
        if actual != expected:
            raise ValueError("episode builder state identity does not match")
        started = state.get("started")
        finished = state.get("finished")
        reason = state.get("finish_reason")
        if type(started) is not bool or type(finished) is not bool:
            raise TypeError("episode builder state flags must be bool")
        if reason is not None and reason not in (
            "terminated",
            "truncated",
            "interrupted",
        ):
            raise ValueError("episode builder state has invalid finish reason")
        if finished != (reason is not None):
            raise ValueError("episode builder finished flag and reason disagree")

        raw_episode = state.get("episode")
        if started:
            if not isinstance(raw_episode, Mapping):
                raise TypeError("started builder state must contain episode arrays")
            validate_episode(raw_episode, action_dim=self.action_dim)
            copied = {key: raw_episode[key].copy() for key in REQUIRED_FIELDS}
            if str(copied["episode_id"][0]) != self.episode_id:
                raise ValueError("episode builder state episode_id does not match")
            if str(copied["collector_id"][0]) != self.collector_id:
                raise ValueError("episode builder state collector_id does not match")
            if str(copied["environment_id"][0]) != self.environment_id:
                raise ValueError("episode builder state environment_id does not match")
            if finished:
                terminal = bool(copied["is_terminal"][-1])
                discount = float(copied["discount"][-1])
                if reason == "terminated" and (not terminal or discount != 0.0):
                    raise ValueError("invalid terminated builder state")
                if reason == "truncated" and (not terminal or discount != 1.0):
                    raise ValueError("invalid truncated builder state")
                if reason == "interrupted" and (terminal or discount != 1.0):
                    raise ValueError("invalid interrupted builder state")
            columns = {
                key: [value.copy() for value in copied[key]] for key in REQUIRED_FIELDS
            }
        else:
            if raw_episode is not None or finished:
                raise ValueError("unstarted builder state cannot contain episode data")
            columns = {name: [] for name in REQUIRED_FIELDS}

        self._columns = columns
        self._started = started
        self._finished = finished
        self._finish_reason = cast(FinishReason | None, reason)

    def _append_row(self, row: Mapping[str, object]) -> None:
        for name in REQUIRED_FIELDS:
            self._columns[name].append(row[name])


__all__ = [
    "EpisodeBuilder",
    "FinishReason",
    "REQUIRED_FIELDS",
    "validate_episode",
    "validate_replay_episode",
]
