"""Single-environment EADream collection driver."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
import hashlib
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from eadream.engine.agent import AgentState
from eadream.engine.distributions import OneHotDist
from eadream.engine.schedules import Every, Once
from eadream.interfaces import (
    PolicyRequest,
    PolicyResult,
    TrainingCounters,
)
from eadream.replay.episode import EpisodeBuilder, validate_replay_episode
from eadream.replay.sampler import SequenceSampler
from eadream.envs.wrappers import EventObservationWrapper, StartGuard
from eadream.observation import validate_observation


class InMemoryReplay:
    """Small replay owner used by the formal in-process driver and test runs."""

    def __init__(self) -> None:
        self.completed: list[dict[str, np.ndarray]] = []
        self.live: EpisodeBuilder | None = None
        self.closed = False

    def set_live(self, episode: EpisodeBuilder | None) -> None:
        if episode is not None and episode.is_finished:
            raise ValueError("live episode must be unfinished")
        self.live = episode

    def append_completed(self, episode: Mapping[str, np.ndarray]) -> dict[str, Any]:
        validate_replay_episode(episode, completed=True)
        episode_id = str(episode["episode_id"][0])
        if any(str(item["episode_id"][0]) == episode_id for item in self.completed):
            raise ValueError(f"completed episode {episode_id!r} already exists")
        if self.live is not None:
            if not self.live.is_finished or self.live.episode_id != episode_id:
                raise ValueError("completed episode does not match live replay episode")
        copied = {key: np.ascontiguousarray(value.copy()) for key, value in episode.items()}
        self.completed.append(copied)
        self.live = None
        for value in copied.values():
            value.setflags(write=False)
        return {"episode_id": episode_id, "length": len(copied["reward"])}

    def sampleable_episodes(self) -> tuple[Mapping[str, np.ndarray], ...]:
        result: list[Mapping[str, np.ndarray]] = list(self.completed)
        if self.live is not None:
            result.append(self.live.snapshot())
        return tuple(result)

    @staticmethod
    def _episode_hash(episode: Mapping[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for key in sorted(episode):
            value = np.ascontiguousarray(episode[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(repr(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, object]:
        """Checkpoint replay indexes and the live builder without duplicating completed payloads."""
        completed = []
        for ordinal, episode in enumerate(self.completed):
            episode_id = str(episode["episode_id"][0])
            length = int(len(episode["reward"]))
            completed.append(
                {
                    "episode_id": episode_id,
                    "filename": f"{episode_id}-{length}.npz",
                    "length": length,
                    "ordinal": ordinal,
                    "sha256": self._episode_hash(episode),
                }
            )
        return {
            "format_version": 1,
            "completed": completed,
            "live": None if self.live is None else self.live.state_dict(),
        }

    def load_state_dict(
        self, state: Mapping[str, object], *, recover_orphans: bool = True
    ) -> None:
        if not isinstance(state, Mapping) or set(state) != {
            "format_version", "completed", "live"
        }:
            raise ValueError("unsupported in-memory replay state")
        if type(state["format_version"]) is not int or state["format_version"] != 1:
            raise ValueError("unsupported in-memory replay state version")
        raw_completed = state["completed"]
        if not isinstance(raw_completed, list):
            raise TypeError("replay completed state must be a list")
        # Completed payloads are owned by durable replay.  Preserve any
        # matching in-memory payloads already present in this process; a cold
        # clone can recover them from its EpisodeStore/index later.
        expected = {
            str(item["episode_id"]): item
            for item in raw_completed
            if isinstance(item, Mapping) and type(item.get("episode_id")) is str
        }
        retained = []
        for episode in self.completed:
            episode_id = str(episode["episode_id"][0])
            metadata = expected.get(episode_id)
            if metadata is None:
                continue
            if int(metadata.get("length", -1)) != len(episode["reward"]):
                raise ValueError("replay completed length differs from checkpoint")
            if metadata.get("sha256") != self._episode_hash(episode):
                raise ValueError("replay completed hash differs from checkpoint")
            retained.append(episode)
        self.completed = retained
        raw_live = state["live"]
        if raw_live is None:
            self.live = None
            return
        if not isinstance(raw_live, Mapping):
            raise TypeError("replay live state must be a mapping")
        self.live = EpisodeBuilder(
            raw_live["episode_id"],
            raw_live["action_dim"],
            collector_id=raw_live["collector_id"],
            environment_id=raw_live["environment_id"],
        )
        self.live.load_state_dict(raw_live)
        if self.live.is_finished:
            raise ValueError("replay live state must be unfinished")

    def close(self) -> None:
        self.closed = True
        self.live = None


def validate_policy_echo(request: PolicyRequest, result: PolicyResult) -> None:
    """Reject IPC policy results that do not echo the outstanding route."""
    fields = ("request_id", "collector_id", "environment_id", "episode_id", "episode_step", "policy_version")
    for name in fields:
        if getattr(request, name) != getattr(result, name):
            raise ValueError(f"policy result field {name} does not match request")


validate_policy_result = validate_policy_echo


def _one_hot_action(action: object, action_dim: int) -> np.ndarray:
    if type(action) is not np.ndarray or action.dtype != np.float32:
        raise TypeError("local policy action must be a float32 NumPy array")
    if action.shape != (action_dim,) or not action.flags.c_contiguous:
        raise ValueError("local policy action has the wrong shape or contiguity")
    if np.count_nonzero(action) != 1 or np.count_nonzero(action == 1.0) != 1:
        raise ValueError("local policy action must be one-hot")
    return action


class SingleEnvDriver:
    """Collect one live episode without serializing recurrent state per step."""

    def __init__(
        self,
        env: gym.Env,
        config: Mapping[str, Any],
        *,
        engine_factory: Callable[[], Any] | None = None,
        agent: Any | None = None,
        allow_deferred_engine_factory: bool = False,
        replay: InMemoryReplay | None = None,
        counters: TrainingCounters | None = None,
        collector_id: str = "collector-0",
        environment_id: str = "environment-0",
        before_run: Callable[[], None] | None = None,
        env_factory: Callable[[], gym.Env] | None = None,
    ) -> None:
        self.config = config
        if (
            engine_factory is not None
            and agent is None
            and bool(config.get("experiment", {}).get("formal", False))
            and not allow_deferred_engine_factory
        ):
            raise ValueError("formal cold driver requires an internal deferred engine factory")
        if agent is not None and bool(config.get("experiment", {}).get("formal", False)) and int((counters or TrainingCounters()).agent_interactions) < int(config.get("training", {}).get("prefill", 2500)):
            raise ValueError("cold formal driver cannot accept a preconstructed agent")
        self.env = self._compose_env(env, config)
        if engine_factory is None:
            if agent is None:
                raise TypeError("SingleEnvDriver requires engine_factory or agent")
            engine_factory = lambda: agent
        self._engine_factory = engine_factory
        self.replay = replay or InMemoryReplay()
        self.counters = counters or TrainingCounters()
        self.collector_id = collector_id
        self.environment_id = environment_id
        self.before_run = before_run
        self._env_factory = env_factory
        self.engine: Any | None = agent
        self._sampler: SequenceSampler | None = None
        self._train_every: Every | None = None
        self._pretrain_once = Once()
        self._observation: Mapping[str, Any] | None = None
        self._state: AgentState | None = None
        self._episode: EpisodeBuilder | None = None
        self._episode_start = True
        self._episode_ordinal = 0
        self._did_first_reset = False
        self._closed = False
        self.resume_boundary = False
        self.resume_trace: list[str] = []
        self.last_train_metrics: dict[str, float] = {}
        self.last_episode_return: float | None = None
        self.last_episode_length: int | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self.run_started_monotonic: float | None = None

        action_dim = getattr(getattr(self.env, "action_space", None), "n", None)
        if not isinstance(action_dim, (int, np.integer)) or int(action_dim) <= 0:
            raise ValueError("single environment must expose a discrete action space")
        action_dim = int(action_dim)
        self.action_dim = action_dim
        configured = config.get("action_dim", action_dim)
        if type(configured) is not int or configured != action_dim:
            raise ValueError("config action_dim does not match environment action space")

    @staticmethod
    def _compose_env(env: gym.Env, config: Mapping[str, Any]) -> gym.Env:
        protocol = str(config.get("environment", {}).get("start_protocol", "none"))
        if isinstance(env, EventObservationWrapper):
            guarded = getattr(env, "env", None)
            if not isinstance(guarded, StartGuard) or guarded.protocol != protocol:
                raise ValueError("event wrapper must directly wrap matching StartGuard")
            return env
        meanings = getattr(getattr(env, "unwrapped", env), "get_action_meanings", None)
        values = None if not callable(meanings) else meanings()
        if values is not None:
            if type(values) not in (list, tuple) or len(values) != getattr(env.action_space, "n", -1) or any(type(v) is not str for v in values):
                raise ValueError("environment action meanings are malformed")
            if protocol == "fire" and "FIRE" not in values:
                raise ValueError("FIRE start protocol requires a FIRE action")
        elif protocol == "fire":
            raise ValueError("FIRE start protocol requires callable action meanings")
        if isinstance(env, StartGuard):
            if env.protocol != protocol:
                raise ValueError("StartGuard protocol does not match config")
            guarded = env
        else:
            guarded = StartGuard(env, protocol=protocol)
        return EventObservationWrapper(guarded)

    @property
    def engine_ready(self) -> bool:
        return self.engine is not None

    def _live_episode(self) -> EpisodeBuilder | None:
        return getattr(self.replay, "live", getattr(self.replay, "_live", None))

    def _ensure_engine(self) -> Any:
        if self.engine is None:
            self.engine = self._engine_factory()
        if self._sampler is None:
            training = self.config["training"]
            interval = float(training["batch_size"]) * float(training["batch_length"]) / float(training["train_ratio"])
            self._train_every = Every(interval)
            self._sampler = SequenceSampler(
                lambda: self.replay.sampleable_episodes(),
                seed=int(training["sequence_sampler_seed"]),
            )
        return self.engine

    def _reset(self) -> None:
        seed = None
        if not self._did_first_reset:
            seed = int(self.config.get("game", {}).get("seed", 0))
            self._did_first_reset = True
        observation, info = self.env.reset(seed=seed)
        self.counters.reset_noop_frames += int(info.get("reset_noop_frames", 0))
        self.counters.forced_start_frames += int(info.get("forced_start_frames", 0))
        if self.resume_boundary:
            self.resume_trace.extend(
                [
                    f"reset_noop_frames={int(info.get('reset_noop_frames', 0))}",
                    f"forced_start_frames={int(info.get('forced_start_frames', 0))}",
                ]
            )
            if str(self.config.get("environment", {}).get("start_protocol", "none")) == "fire":
                self.resume_trace.append("FIRE")
        observation = self._normalize_observation(observation)
        # The event wrapper's reset contract is an explicit zero mask.
        observation["event"] = np.zeros((64, 64), dtype=np.uint8)
        episode_id = f"episode-{self._episode_ordinal}"
        self._episode_ordinal += 1
        policy_version = 0 if self.engine is None else int(self.engine.policy_version)
        episode = EpisodeBuilder(
            episode_id,
            self.action_dim,
            collector_id=self.collector_id,
            environment_id=self.environment_id,
        )
        episode.start(observation, policy_version=policy_version)
        self.replay.set_live(episode)
        self._episode = episode
        self._observation = observation
        self._state = None
        self._episode_start = True
        self._episode_return = 0.0
        self._episode_length = 0

    @staticmethod
    def _normalize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        return validate_observation(observation, copy=True)

    def _prefill_action(self) -> np.ndarray:
        logits = torch.zeros(self.action_dim, device=torch.device("cpu"))
        sampled = OneHotDist(logits=logits).sample()
        return np.ascontiguousarray(sampled.detach().cpu().numpy(), dtype=np.float32)

    def _run_updates(self, *, first_learned_call: bool) -> None:
        if self._sampler is None or self._train_every is None:
            return
        if first_learned_call:
            updates = int(self.config["training"]["pretrain_batches"])
        else:
            updates = self._train_every(self.counters.agent_interactions)
        for _ in range(updates):
            batch = self._sampler.sample(
                int(self.config["training"]["batch_size"]),
                int(self.config["training"]["batch_length"]),
            )
            metrics = self.engine.train_batch(batch)
            if isinstance(metrics, Mapping):
                self.last_train_metrics = {
                    str(name): float(value)
                    for name, value in metrics.items()
                    if isinstance(value, (int, float, np.integer, np.floating))
                    and np.isfinite(float(value))
                }
            self.counters.gradient_updates += 1

    def _learned_action(self) -> np.ndarray:
        engine = self._ensure_engine()
        first = self._pretrain_once()
        self._run_updates(first_learned_call=first)
        action, state = engine.policy(
            self._observation,
            self._state,
            episode_start=self._episode_start,
            deterministic=False,
        )
        self._state = state
        return _one_hot_action(action, self.action_dim)

    def _append_step(
        self,
        action: np.ndarray,
        observation: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
    ) -> None:
        if self._episode is None:
            raise RuntimeError("episode is not active")
        terminal = bool(terminated or truncated)
        if terminated and truncated:
            raise ValueError("environment cannot be both terminated and truncated")
        if "raw_reward" in info:
            raw_reward = info["raw_reward"]
            if isinstance(raw_reward, (bool, np.bool_)) or not isinstance(raw_reward, (int, float, np.integer, np.floating)) or not np.isfinite(raw_reward):
                raise ValueError("raw_reward must be finite")
            if float(raw_reward) != float(reward):
                raise ValueError("raw_reward does not match returned reward")
            reward = float(raw_reward)
        transition = {
            "image": observation["image"],
            "event": observation["event"],
            "action": action,
            "reward": np.float32(reward),
            "discount": np.float32(0.0 if terminated else 1.0),
            "is_first": np.bool_(False),
            "is_terminal": np.bool_(terminal),
            "episode_id": self._episode.episode_id,
            "episode_step": np.int64(len(self._episode.snapshot()["reward"])),
            "collector_id": self.collector_id,
            "environment_id": self.environment_id,
            "policy_version": np.int64(0 if self.engine is None else self.engine.policy_version),
        }
        self._episode.append(transition)
        self._episode_return += float(reward)
        self._episode_length += 1
        if terminal:
            reason = "terminated" if terminated else "truncated"
            completed = self._episode.finish(reason)
            self.replay.append_completed(completed)
            self._episode = None
            self._observation = None
            self._state = None
            self._episode_start = True
            self.last_episode_return = float(self._episode_return)
            self.last_episode_length = int(self._episode_length)
        else:
            self.replay.set_live(self._episode)
            self._observation = observation
            self._episode_start = False

        self.counters.agent_interactions += 1
        self.counters.ale_frames += int(info.get("ale_frames", 0))
        self.counters.reset_noop_frames += int(info.get("reset_noop_frames", 0))
        self.counters.forced_start_frames += int(info.get("forced_start_frames", 0))

    def run_until(self, target_interactions: int) -> "SingleEnvDriver":
        if type(target_interactions) is not int or target_interactions < 0:
            raise ValueError("target_interactions must be a non-negative integer")
        if target_interactions < self.counters.agent_interactions:
            raise ValueError("learn target is lower than current agent interactions")
        if self._closed:
            raise RuntimeError("driver is closed")
        if self.before_run is not None:
            self.before_run()
            self.before_run = None
        if self.run_started_monotonic is None:
            self.run_started_monotonic = time.monotonic()
        while self.counters.agent_interactions < target_interactions:
            if self._observation is None:
                self._reset()
            prefill = self.counters.agent_interactions < int(self.config["training"]["prefill"])
            action = self._prefill_action() if prefill else self._learned_action()
            action_index = int(action.argmax())
            observation, reward, terminated, truncated, info = self.env.step(action_index)
            normalized = self._normalize_observation(observation)
            self._append_step(action, normalized, float(reward), bool(terminated), bool(truncated), info)
        # Keep a fresh, sampleable live episode at every non-zero boundary.  The
        # reset creates no interaction and therefore does not alter the target.
        if target_interactions > 0 and self._observation is None and not self._closed:
            self._reset()
        return self

    def state_dict(self) -> dict[str, object]:
        counters = {
            name: int(getattr(self.counters, name))
            for name in (
                "agent_interactions",
                "ale_frames",
                "reset_noop_frames",
                "forced_start_frames",
                "evaluation_interactions",
                "gradient_updates",
            )
        }
        return {
            "format_version": 1,
            "counters": counters,
            "replay": self.replay.state_dict(),
            "schedules": {
                "pretrain": self._pretrain_once.state_dict(),
                "train_every": None if self._train_every is None else self._train_every.state_dict(),
            },
            "sampler": None if self._sampler is None else self._sampler.state_dict(),
            "episode_ordinal": self._episode_ordinal,
            "did_first_reset": self._did_first_reset,
            "episode_start": self._episode_start,
            "resume_boundary": self.resume_boundary,
            "resume_trace": list(self.resume_trace),
        }

    def load_state_dict(
        self, state: Mapping[str, object], *, recover_replay_orphans: bool = True
    ) -> None:
        self._validate_state_dict(state)
        if not isinstance(state, Mapping) or type(state.get("format_version")) is not int or state.get("format_version") != 1:
            raise ValueError("unsupported driver checkpoint state")
        raw_counters = state.get("counters")
        if not isinstance(raw_counters, Mapping):
            raise ValueError("driver checkpoint counters are missing")
        for name in (
            "agent_interactions",
            "ale_frames",
            "reset_noop_frames",
            "forced_start_frames",
            "evaluation_interactions",
            "gradient_updates",
        ):
            value = raw_counters.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"driver checkpoint counter {name} is invalid")
            setattr(self.counters, name, value)
        self.replay.load_state_dict(
            state["replay"], recover_orphans=recover_replay_orphans
        )
        schedules = state.get("schedules")
        if not isinstance(schedules, Mapping):
            raise ValueError("driver checkpoint schedules are missing")
        self._pretrain_once.load_state_dict(schedules["pretrain"])
        train_every = schedules.get("train_every")
        if train_every is None:
            self._train_every = None
        else:
            training = self.config["training"]
            interval = float(training["batch_size"]) * float(training["batch_length"]) / float(training["train_ratio"])
            self._train_every = Every(interval)
            self._train_every.load_state_dict(train_every)
        sampler_state = state.get("sampler")
        self._sampler = None
        if sampler_state is not None:
            if self.engine is None:
                raise RuntimeError("driver engine must be restored before sampler state")
            self._sampler = SequenceSampler(
                lambda: self.replay.sampleable_episodes(),
                seed=int(self.config["training"]["sequence_sampler_seed"]),
            )
            self._sampler.load_state_dict(sampler_state)
        if type(state.get("episode_ordinal")) is not int or state["episode_ordinal"] < 0:
            raise ValueError("driver checkpoint episode ordinal is invalid")
        self._episode_ordinal = state["episode_ordinal"]
        if type(state.get("did_first_reset")) is not bool or type(state.get("episode_start")) is not bool:
            raise ValueError("driver checkpoint flags are invalid")
        self._did_first_reset = state["did_first_reset"]
        self._episode_start = state["episode_start"]
        self.resume_boundary = bool(state.get("resume_boundary", False))
        self.resume_trace = list(state.get("resume_trace", []))
        self._episode = self._live_episode()
        self._observation = None
        self._state = None

    def _validate_state_dict(self, state: Mapping[str, object]) -> None:
        """Validate every nested field before replacing live counters/state."""
        if not isinstance(state, Mapping) or type(state.get("format_version")) is not int or state.get("format_version") != 1:
            raise ValueError("unsupported driver checkpoint state")
        raw_counters = state.get("counters")
        if not isinstance(raw_counters, Mapping):
            raise ValueError("driver checkpoint counters are missing")
        for name in (
            "agent_interactions", "ale_frames", "reset_noop_frames",
            "forced_start_frames", "evaluation_interactions", "gradient_updates",
        ):
            value = raw_counters.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"driver checkpoint counter {name} is invalid")
        replay_state = state.get("replay")
        if not isinstance(replay_state, Mapping):
            raise ValueError("driver checkpoint replay is missing")
        if set(replay_state) == {"format_version", "capacity", "next_ordinal", "completed", "live"}:
            if type(replay_state.get("format_version")) is not int or replay_state.get("format_version") != 1:
                raise ValueError("unsupported durable replay state")
            if type(replay_state.get("capacity")) is not int or replay_state["capacity"] < 1:
                raise ValueError("durable replay capacity is invalid")
            if type(replay_state.get("next_ordinal")) is not int or replay_state["next_ordinal"] < 0:
                raise ValueError("durable replay ordinal is invalid")
            if not isinstance(replay_state.get("completed"), list):
                raise ValueError("durable replay completed state is invalid")
        else:
            probe_replay = InMemoryReplay()
            probe_replay.load_state_dict(replay_state)
        schedules = state.get("schedules")
        if not isinstance(schedules, Mapping) or set(schedules) != {"pretrain", "train_every"}:
            raise ValueError("driver checkpoint schedules are missing")
        pretrain = Once()
        pretrain.load_state_dict(schedules["pretrain"])
        train_every = schedules["train_every"]
        if train_every is not None:
            if not isinstance(train_every, Mapping) or "_last" not in train_every:
                raise ValueError("driver checkpoint train schedule is invalid")
            training = self.config["training"]
            cadence = Every(float(training["batch_size"]) * float(training["batch_length"]) / float(training["train_ratio"]))
            cadence.load_state_dict(train_every)
        sampler_state = state.get("sampler")
        if sampler_state is not None:
            sampler = SequenceSampler(lambda: (), seed=int(self.config["training"]["sequence_sampler_seed"]))
            sampler.load_state_dict(sampler_state)
        if type(state.get("episode_ordinal")) is not int or state["episode_ordinal"] < 0:
            raise ValueError("driver checkpoint episode ordinal is invalid")
        for name in ("did_first_reset", "episode_start"):
            if type(state.get(name)) is not bool:
                raise ValueError("driver checkpoint flags are invalid")
        if type(state.get("resume_boundary", False)) is not bool:
            raise ValueError("driver checkpoint resume_boundary is invalid")
        if not isinstance(state.get("resume_trace", []), list):
            raise ValueError("driver checkpoint resume_trace is invalid")

    def resume_boundary_reset(self) -> None:
        """Publish an interrupted live fragment and start a fresh environment episode."""
        if self._closed:
            raise RuntimeError("driver is closed")
        if self._episode is not None:
            try:
                self._episode.finish("interrupted")
                self.replay.append_completed(self._episode.snapshot())
            except (RuntimeError, ValueError):
                pass
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        if self._env_factory is not None:
            fresh = self._env_factory()
            self.env = self._compose_env(fresh, self.config)
        self._episode = None
        self._observation = None
        self._state = None
        self._episode_start = True
        self._did_first_reset = False
        self.resume_boundary = True
        self.resume_trace = ["environment_closed", "reset", "episode_started"]
        self._reset()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._episode is not None:
            try:
                snapshot = self._episode.finish("interrupted")
                self.replay.set_live(self._episode)
                # Keep the interrupted live episode sampleable; do not publish it.
                del snapshot
            except (RuntimeError, ValueError):
                pass
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        close_replay = getattr(self.replay, "close", None)
        if callable(close_replay):
            close_replay()


__all__ = [
    "InMemoryReplay",
    "SingleEnvDriver",
    "validate_policy_echo",
    "validate_policy_result",
]
