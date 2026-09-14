"""Public EADream training and inference API."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eadream.driver import InMemoryReplay, SingleEnvDriver
from eadream.engine.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointManager,
    CheckpointRecord,
    capture_rng_state,
    restore_rng_state,
)
from eadream.engine.agent import AgentState, EADreamAgent
from eadream.interfaces import TrainingCounters
from eadream.replay.store import EpisodeStore


def seed_source(seed: int) -> None:
    """Seed source-visible RNGs before the first environment run/construction."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EADream:
    """Single-environment EADream owner with absolute interaction targets."""

    def __init__(self, env: Any, config: Mapping[str, Any], env_factory: Any | None = None) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        if env_factory is not None and not callable(env_factory):
            raise TypeError("env_factory must be callable")
        self.config = config
        action_dim = getattr(getattr(env, "action_space", None), "n", None)
        if not isinstance(action_dim, (int, np.integer)) or int(action_dim) <= 0:
            raise ValueError("EADream requires a discrete environment action space")
        action_dim = int(action_dim)
        configured = config.get("action_dim", action_dim)
        if type(configured) is not int or configured != action_dim:
            raise ValueError("config action_dim does not match environment action space")
        meanings = getattr(getattr(env, "unwrapped", env), "get_action_meanings", None)
        protocol = str(config.get("environment", {}).get("start_protocol", "none"))
        if callable(meanings):
            raw_values = meanings()
            if type(raw_values) not in (list, tuple):
                raise ValueError("environment action meanings must be a list or tuple")
            values = tuple(raw_values)
            if len(values) != action_dim or any(type(value) is not str for value in values):
                raise ValueError("environment action meanings do not match action space")
        elif protocol == "fire":
            raise ValueError("FIRE start protocol requires callable action meanings")
        else:
            values = tuple(str(index) for index in range(action_dim))
        if protocol == "fire" and "FIRE" not in values:
            raise ValueError("FIRE start protocol requires a FIRE action")
        self.config = dict(config)
        self.config["action_meanings"] = values
        if int(self.config.get("runtime", {}).get("num_envs", 1)) != 1:
            raise ValueError("phase one supports exactly one environment")

        self.env = env
        self.action_dim = action_dim
        self._engine: EADreamAgent | None = None
        self._externally_injected_engine = False
        self.counters = TrainingCounters()
        self.replay = InMemoryReplay()
        self._checkpoint_directory: Path | None = None
        self._seeded = False
        self._closed = False
        self.driver = SingleEnvDriver(
            env,
            self.config,
            engine_factory=self._construct_engine,
            allow_deferred_engine_factory=True,
            replay=self.replay,
            counters=self.counters,
            before_run=self._prepare_run,
            env_factory=env_factory,
        )
        self.env = self.driver.env

        self.config_hash = self._hash_config(self.config)
        provenance = self.config.get("provenance", {})
        self.provenance_hash = self._hash_config(provenance)

    @staticmethod
    def _hash_config(value: object) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def engine(self) -> EADreamAgent | None:
        return self._engine

    @engine.setter
    def engine(self, value: EADreamAgent | None) -> None:
        if value is not None and not getattr(self, "_engine_construction", False):
            self._externally_injected_engine = True
        self._engine = value

    @property
    def device(self) -> torch.device:
        if self.engine is not None:
            return self.engine.device
        requested = str(self.config.get("runtime", {}).get("device", "cpu"))
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _prepare_run(self) -> None:
        if not self._seeded:
            seed = int(self.config.get("game", {}).get("seed", 0))
            seed_source(seed)
            self._seeded = True

    def _construct_engine(self) -> EADreamAgent:
        if self._engine is None:
            runtime = self.config.get("runtime", {})
            self._engine = EADreamAgent(
                self.config,
                action_dim=self.action_dim,
                device=runtime.get("device", "cpu"),
            )
        return self.engine

    def learn(self, total_interactions: int) -> "EADream":
        if self._closed:
            raise RuntimeError("EADream is closed")
        prefill = int(self.config.get("training", {}).get("prefill", 2500))
        if bool(self.config.get("experiment", {}).get("formal", False)) and self.counters.agent_interactions < prefill:
            if self._externally_injected_engine or self._engine is not None or self.driver.engine is not None:
                raise RuntimeError("formal cold learn requires deferred engine construction after prefill")
        self.driver.run_until(total_interactions)
        return self

    def predict(
        self,
        observation: Mapping[str, Any],
        state: AgentState | None = None,
        episode_start: bool | np.ndarray | torch.Tensor = False,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, AgentState]:
        if self._closed:
            raise RuntimeError("EADream is closed")
        if bool(self.config.get("experiment", {}).get("formal", False)) and self.counters.agent_interactions < int(self.config.get("training", {}).get("prefill", 2500)):
            raise RuntimeError("formal predict requires completing prefill first")
        self._prepare_run()
        engine = self._construct_engine()
        return engine.policy(
            observation,
            state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.driver.close()
        if self.engine is not None:
            self.engine.cpu()
        self.engine = None

    def snapshot(self) -> dict[str, Any]:
        """Return the complete mutable training state for checkpoint publication."""
        if self._closed:
            raise RuntimeError("EADream is closed")
        engine = self._construct_engine()
        driver_state = self.driver.state_dict()
        inference = engine.inference_state_dict()
        return {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "provenance_hash": self.provenance_hash,
            "world_model": _cpu_state_dict(engine.world_model.state_dict()),
            "actor": _cpu_state_dict(engine.behavior.actor.state_dict()),
            "value": _cpu_state_dict(engine.behavior.value.state_dict()),
            "slow_value": _cpu_state_dict(engine.behavior.slow_value.state_dict()),
            "optimizers": {
                "world_model": _cpu_state_dict(engine.world_model.optimizer.optimizer.state_dict()),
                "actor": _cpu_state_dict(engine.behavior.actor_optimizer.optimizer.state_dict()),
                "value": _cpu_state_dict(engine.behavior.value_optimizer.optimizer.state_dict()),
            },
            "scalers": {
                "world_model": _cpu_state_dict(engine.world_model.optimizer.scaler.state_dict()),
                "actor": _cpu_state_dict(engine.behavior.actor_optimizer.scaler.state_dict()),
                "value": _cpu_state_dict(engine.behavior.value_optimizer.scaler.state_dict()),
            },
            "reward_ema": _cpu_state_dict(engine.behavior.reward_ema.state_dict()),
            "harmonizers": _cpu_state_dict(engine.world_model.harmonizers.state_dict()),
            "schedules": copy_mapping(driver_state["schedules"]),
            "counters": copy_mapping(driver_state["counters"]),
            "replay": copy_mapping(driver_state["replay"]),
            "sampler": copy_mapping(driver_state["sampler"]),
            "driver": driver_state,
            "rng": capture_rng_state(),
            "action_meanings": tuple(engine.action_meanings),
            "observation_contract_hash": inference["observation_contract_hash"],
            "agent_interactions": int(self.counters.agent_interactions),
            "policy_version": int(engine.policy_version),
            "resume_boundary": bool(self.driver.resume_boundary),
            "resume_trace": list(self.driver.resume_trace),
            "replay_root": "replay" if isinstance(self.replay, EpisodeStore) else None,
            "update_counts": {
                "world_model": int(engine.world_model.update_count),
                "behavior": int(engine.behavior.update_count),
            },
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        """Restore a validated payload in dependency order and reset the environment."""
        if self._closed:
            raise RuntimeError("EADream is closed")
        if payload.get("config_hash") != self.config_hash:
            raise CheckpointCompatibilityError("checkpoint config hash is incompatible")
        if payload.get("provenance_hash") != self.provenance_hash:
            raise CheckpointCompatibilityError("checkpoint provenance hash is incompatible")
        counters_payload = payload.get("counters")
        driver_payload = payload.get("driver")
        if not isinstance(counters_payload, Mapping) or not isinstance(driver_payload, Mapping):
            raise CheckpointCompatibilityError("checkpoint counters or driver state is invalid")
        if payload.get("agent_interactions") != counters_payload.get("agent_interactions"):
            raise CheckpointCompatibilityError("checkpoint interaction counters disagree")
        if driver_payload.get("counters") != counters_payload:
            raise CheckpointCompatibilityError("checkpoint driver counters disagree")
        configured_meanings = tuple(self.config.get("action_meanings", ()))
        if configured_meanings and tuple(payload.get("action_meanings", ())) != configured_meanings:
            raise CheckpointCompatibilityError("checkpoint action meanings are incompatible")
        engine = self._construct_engine()
        expected_meanings = tuple(engine.action_meanings)
        if tuple(payload["action_meanings"]) != expected_meanings:
            raise ValueError("checkpoint action meanings are incompatible")
        if payload["observation_contract_hash"] != engine.inference_state_dict()["observation_contract_hash"]:
            raise ValueError("checkpoint observation contract hash is incompatible")
        replay_root = payload.get("replay_root")
        if replay_root is not None and self._checkpoint_directory is not None:
            try:
                probe_store = EpisodeStore(self._checkpoint_directory / str(replay_root))
                probe_store.load_state_dict(payload["replay"], recover_orphans=False)
            except Exception as error:
                raise CheckpointCompatibilityError("checkpoint durable replay is invalid") from error
        try:
            self.driver._validate_state_dict(payload["driver"])
        except Exception as error:
            raise CheckpointCompatibilityError("checkpoint driver state is invalid") from error
        _validate_restore_payload(engine, payload)

        # Keep a rollback image for runtime failures after validation. This
        # covers nested loaders that reject a semantically invalid value.
        old_engine = self._engine
        old_replay = self.replay
        old_driver_state = copy.deepcopy(self.driver.state_dict())
        old_engine_state = self.snapshot()
        old_rng = capture_rng_state()
        try:
            # Module tensors first, then optimizer/scaler slots that reference them.
            engine.world_model.load_state_dict(payload["world_model"], strict=True)
            engine.behavior.actor.load_state_dict(payload["actor"], strict=True)
            engine.behavior.value.load_state_dict(payload["value"], strict=True)
            engine.behavior.slow_value.load_state_dict(payload["slow_value"], strict=True)
            engine.world_model.harmonizers.load_state_dict(payload["harmonizers"], strict=True)
            engine.behavior.reward_ema.load_state_dict(payload["reward_ema"])
            engine.world_model.optimizer.optimizer.load_state_dict(payload["optimizers"]["world_model"])
            engine.behavior.actor_optimizer.optimizer.load_state_dict(payload["optimizers"]["actor"])
            engine.behavior.value_optimizer.optimizer.load_state_dict(payload["optimizers"]["value"])
            engine.world_model.optimizer.scaler.load_state_dict(payload["scalers"]["world_model"])
            engine.behavior.actor_optimizer.scaler.load_state_dict(payload["scalers"]["actor"])
            engine.behavior.value_optimizer.scaler.load_state_dict(payload["scalers"]["value"])
            _move_optimizer_state(engine.world_model.optimizer.optimizer, engine.device)
            _move_optimizer_state(engine.behavior.actor_optimizer.optimizer, engine.device)
            _move_optimizer_state(engine.behavior.value_optimizer.optimizer, engine.device)

            engine.policy_version = int(payload.get("policy_version", 0))
            counts = payload.get("update_counts", {})
            engine.world_model.update_count = int(counts.get("world_model", 0))
            engine.behavior.update_count = int(counts.get("behavior", 0))
            self.driver.engine = engine
            if replay_root is not None and self._checkpoint_directory is not None:
                durable = EpisodeStore(self._checkpoint_directory / str(replay_root))
                durable.load_state_dict(payload["replay"], recover_orphans=False)
                self.replay = durable
                self.driver.replay = durable
            self.driver.load_state_dict(payload["driver"], recover_replay_orphans=False)
            self.driver.resume_boundary_reset()
            self.env = self.driver.env
            self._seeded = True
            self.driver.before_run = None
            # Resetting the environment/MOG2 may consume framework RNG; restore the
            # learner RNG only after that boundary has been completed.
            restore_rng_state(payload["rng"])
        except BaseException:
            if old_engine is not None:
                _restore_engine_state(old_engine, old_engine_state)
            self._engine = old_engine
            self.replay = old_replay
            self.driver.engine = old_engine
            self.driver.replay = old_replay
            try:
                self.driver.load_state_dict(old_driver_state)
            except Exception:
                pass
            restore_rng_state(old_rng)
            raise

    def save_checkpoint(self, path: Path) -> CheckpointRecord:
        destination = Path(path)
        self._checkpoint_directory = destination.parent
        prior_replay = self.replay
        replay_store, before_ids = self._publish_replay_store(destination.parent / "replay")
        manager = CheckpointManager(
            destination.parent,
            config_hash=self.config_hash,
            provenance_hash=self.provenance_hash,
            filename=destination.name,
        )
        try:
            return manager.save(self.snapshot())
        except BaseException:
            if isinstance(replay_store, EpisodeStore):
                current_ids = {
                    record["episode_id"] for record in replay_store.state_dict()["completed"]
                }
                replay_store.remove_completed(current_ids - before_ids)
            self.replay = prior_replay
            self.driver.replay = prior_replay
            raise

    def load_checkpoint(self, path: Path) -> CheckpointRecord:
        destination = Path(path)
        if self.driver._env_factory is None:
            raise RuntimeError("checkpoint resume requires an environment factory")
        self._checkpoint_directory = destination.parent
        manager = CheckpointManager(
            destination.parent,
            config_hash=self.config_hash,
            provenance_hash=self.provenance_hash,
            filename=destination.name,
        )
        return manager.load(destination, self.restore)

    def _publish_replay_store(self, directory: Path) -> tuple[EpisodeStore, set[str]]:
        """Move in-memory completed episodes into the durable EpisodeStore."""
        if isinstance(self.replay, EpisodeStore):
            if self.replay.directory.resolve() == directory.resolve():
                before = {record["episode_id"] for record in self.replay.state_dict()["completed"]}
                return self.replay, before
            source = self.replay
            store = EpisodeStore(directory, capacity=source.capacity)
            before: set[str] = set()
            for episode in source.sampleable_episodes():
                if bool(episode["is_terminal"][-1]) or float(episode["discount"][-1]) != 1.0:
                    store.append_completed(episode)
            live = getattr(source, "_live", None)
            if live is not None:
                store.set_live(live)
            self.replay = store
            self.driver.replay = store
            return store, before
        store = EpisodeStore(directory)
        before = set()
        for episode in self.replay.completed:
            episode_id = str(episode["episode_id"][0])
            if episode_id not in {record["episode_id"] for record in store.state_dict()["completed"]}:
                store.append_completed(episode)
        if self.replay.live is not None:
            store.set_live(self.replay.live)
        self.replay = store
        self.driver.replay = store
        return store, before


def copy_mapping(value: object) -> object:
    return copy.deepcopy(value)


def _cpu_state_dict(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_state_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state_dict(item) for item in value)
    return copy.deepcopy(value)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _restore_engine_state(engine: EADreamAgent, payload: Mapping[str, Any]) -> None:
    """Restore the mutable engine portion of a local rollback image."""
    engine.world_model.load_state_dict(payload["world_model"], strict=True)
    engine.behavior.actor.load_state_dict(payload["actor"], strict=True)
    engine.behavior.value.load_state_dict(payload["value"], strict=True)
    engine.behavior.slow_value.load_state_dict(payload["slow_value"], strict=True)
    engine.world_model.harmonizers.load_state_dict(payload["harmonizers"], strict=True)
    engine.behavior.reward_ema.load_state_dict(payload["reward_ema"])
    engine.world_model.optimizer.optimizer.load_state_dict(payload["optimizers"]["world_model"])
    engine.behavior.actor_optimizer.optimizer.load_state_dict(payload["optimizers"]["actor"])
    engine.behavior.value_optimizer.optimizer.load_state_dict(payload["optimizers"]["value"])
    engine.world_model.optimizer.scaler.load_state_dict(payload["scalers"]["world_model"])
    engine.behavior.actor_optimizer.scaler.load_state_dict(payload["scalers"]["actor"])
    engine.behavior.value_optimizer.scaler.load_state_dict(payload["scalers"]["value"])
    _move_optimizer_state(engine.world_model.optimizer.optimizer, engine.device)
    _move_optimizer_state(engine.behavior.actor_optimizer.optimizer, engine.device)
    _move_optimizer_state(engine.behavior.value_optimizer.optimizer, engine.device)
    engine.policy_version = int(payload["policy_version"])
    engine.world_model.update_count = int(payload["update_counts"]["world_model"])
    engine.behavior.update_count = int(payload["update_counts"]["behavior"])


def _validate_restore_payload(engine: EADreamAgent, payload: Mapping[str, Any]) -> None:
    """Validate model, optimizer, replay, and schedule shapes before mutation."""
    for name, module, state in (
        ("world model", engine.world_model, payload["world_model"]),
        ("actor", engine.behavior.actor, payload["actor"]),
        ("value", engine.behavior.value, payload["value"]),
        ("slow value", engine.behavior.slow_value, payload["slow_value"]),
    ):
        if not isinstance(state, Mapping) or set(state) != set(module.state_dict()):
            raise CheckpointCompatibilityError(f"checkpoint {name} state fields differ")
        current = module.state_dict()
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[key].shape:
                raise CheckpointCompatibilityError(f"checkpoint {name} tensor shape differs")
    optimizers = payload["optimizers"]
    scalers = payload["scalers"]
    if not isinstance(optimizers, Mapping) or set(optimizers) != {"world_model", "actor", "value"}:
        raise CheckpointCompatibilityError("checkpoint optimizer fields differ")
    if not isinstance(scalers, Mapping) or set(scalers) != {"world_model", "actor", "value"}:
        raise CheckpointCompatibilityError("checkpoint scaler fields differ")
    bundles = {
        "world_model": engine.world_model.optimizer,
        "actor": engine.behavior.actor_optimizer,
        "value": engine.behavior.value_optimizer,
    }
    for name, bundle in bundles.items():
        old = copy.deepcopy(bundle.optimizer.state_dict())
        try:
            bundle.optimizer.load_state_dict(optimizers[name])
        except Exception as error:
            bundle.optimizer.load_state_dict(old)
            raise CheckpointCompatibilityError(f"checkpoint {name} optimizer state is invalid") from error
        bundle.optimizer.load_state_dict(old)
        scaler = bundle.scaler
        old_scaler = copy.deepcopy(scaler.state_dict())
        try:
            scaler.load_state_dict(scalers[name])
        except Exception as error:
            scaler.load_state_dict(old_scaler)
            raise CheckpointCompatibilityError(f"checkpoint {name} scaler state is invalid") from error
        scaler.load_state_dict(old_scaler)
    harmonizers = payload["harmonizers"]
    expected_harmonizers = engine.world_model.harmonizers.state_dict()
    if not isinstance(harmonizers, Mapping) or set(harmonizers) != set(expected_harmonizers):
        raise CheckpointCompatibilityError("checkpoint harmonizer fields differ")
    for key, current in expected_harmonizers.items():
        value = harmonizers[key]
        if not isinstance(value, torch.Tensor) or value.shape != current.shape or not torch.isfinite(value).all():
            raise CheckpointCompatibilityError("checkpoint harmonizer state is invalid")
    reward_ema = payload["reward_ema"]
    if not isinstance(reward_ema, Mapping) or set(reward_ema) != {"alpha", "values"}:
        raise CheckpointCompatibilityError("checkpoint reward EMA fields differ")
    alpha = reward_ema["alpha"]
    values = reward_ema["values"]
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
            not np.isfinite(alpha) or not 0.0 < float(alpha) <= 1.0 or
            not isinstance(values, torch.Tensor) or values.shape != (2,) or
            not torch.isfinite(values).all()):
        raise CheckpointCompatibilityError("checkpoint reward EMA state is invalid")
    replay_state = payload["replay"]
    try:
        if isinstance(replay_state, Mapping) and set(replay_state) == {
            "format_version", "capacity", "next_ordinal", "completed", "live"
        }:
            if type(replay_state.get("format_version")) is not int or replay_state["format_version"] != 1:
                raise ValueError("durable replay format is invalid")
            if type(replay_state.get("capacity")) is not int or replay_state["capacity"] < 1:
                raise ValueError("durable replay capacity is invalid")
            if type(replay_state.get("next_ordinal")) is not int or replay_state["next_ordinal"] < 0:
                raise ValueError("durable replay ordinal is invalid")
            if not isinstance(replay_state.get("completed"), list):
                raise ValueError("durable replay completed state is invalid")
        else:
            replay_probe = InMemoryReplay()
            replay_probe.load_state_dict(replay_state)
    except Exception as error:
        raise CheckpointCompatibilityError("checkpoint replay state is invalid") from error
    driver = payload.get("driver")
    if not isinstance(driver, Mapping) or set(driver) < {
        "format_version", "counters", "replay", "schedules", "sampler", "episode_ordinal"
    }:
        raise CheckpointCompatibilityError("checkpoint driver state is invalid")


__all__ = ["AgentState", "EADream", "EADreamAgent", "seed_source"]
