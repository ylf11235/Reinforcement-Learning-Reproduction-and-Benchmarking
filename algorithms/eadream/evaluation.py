"""Isolated EADream evaluation and diagnostic protocols.

The functions in this module deliberately own no trainer state.  They create (or
accept) fresh environments, reset recurrent policy state at every episode, and
only aggregate the raw reward reported by the environment.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eadream.algorithm import seed_source
from eadream.engine.checkpoint import capture_rng_state, restore_rng_state
from eadream.envs.atari import protocol_for


def _rng_hash() -> str:
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        state += tuple(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())
    return hashlib.sha256(repr(state).encode("utf-8")).hexdigest()


def _make_env(factory: Any, *, mode: str, seed: int | None = None) -> Any:
    if not callable(factory):
        return factory
    attempts = (
        lambda: factory(mode=mode, seed=seed),
        lambda: factory(mode=mode),
        lambda: factory(mode),
        lambda: factory(),
    )
    error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, KeyError) as exc:
            error = exc
    assert error is not None
    raise error


def _predict(policy: Any, observation: Any, state: Any, *, episode_start: bool) -> tuple[Any, Any]:
    predict = getattr(policy, "predict")
    try:
        result = predict(observation, state=state, episode_start=episode_start, deterministic=True)
    except TypeError:
        try:
            result = predict(observation, state, episode_start, True)
        except TypeError:
            result = predict(observation, deterministic=True)
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _action_index(action: Any) -> int:
    array = np.asarray(action)
    if array.ndim == 0:
        return int(array)
    if array.ndim == 1:
        return int(np.argmax(array))
    return int(np.argmax(array[0]))


def _episode(policy: Any, env: Any, *, seed: int | None, first_reset: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    observation, reset_info = env.reset(seed=seed)
    state = None
    episode_start = True
    raw_return = 0.0
    interactions = ale_frames = 0
    event_pixels = 0
    forced = {"forced_start_actions": 0, "forced_start_frames": 0, "forced_start_reward": 0.0}
    reset_noops = int(reset_info.get("reset_noop_frames", 0)) if isinstance(reset_info, Mapping) else 0
    terminated = truncated = False
    termination_reason = "unknown"
    while not (terminated or truncated):
        action, state = _predict(policy, observation, state, episode_start=episode_start)
        observation, reward, terminated, truncated, info = env.step(_action_index(action))
        info = info if isinstance(info, Mapping) else {}
        raw_return += float(info.get("raw_reward", reward))
        interactions += 1
        ale_frames += int(info.get("ale_frames", 0))
        event = observation.get("event") if isinstance(observation, Mapping) else None
        if event is not None:
            event_pixels += int(np.count_nonzero(event))
        for key in forced:
            forced[key] += info.get(key, 0)
        episode_start = False
    termination_reason = "truncated" if truncated else "terminated"
    return {
        "seed": seed,
        "raw_return": float(raw_return),
        "return": float(raw_return),
        "return_kind": "raw_unclipped",
        "policy_interactions": interactions,
        "ale_frames": ale_frames,
        "reset_noop_frames": reset_noops,
        "forced_start_actions": int(forced["forced_start_actions"]),
        "forced_start_frames": int(forced["forced_start_frames"]),
        "forced_start_reward": float(forced["forced_start_reward"]),
        "event_pixels": event_pixels,
        "termination_reason": termination_reason,
        "wall_seconds": time.perf_counter() - started,
    }


def _evaluate(policy: Any, factory: Any, *, seeds: Sequence[int | None], mode: str, train_seed: int | None,
              protocol: str, fresh_each: bool) -> dict[str, Any]:
    caller_rng = capture_rng_state()
    prior = _rng_hash()
    if train_seed is not None:
        seed_source(int(train_seed))
    initial = _rng_hash()
    episodes: list[dict[str, Any]] = []
    env = None
    evaluator_final: str | None = None
    try:
        for index, seed in enumerate(seeds):
            if fresh_each or env is None:
                if env is not None and hasattr(env, "close"):
                    env.close()
                env = _make_env(factory, mode=mode, seed=seed)
            episodes.append(_episode(policy, env, seed=seed if index == 0 or fresh_each else None))
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()
        evaluator_final = _rng_hash()
        restore_rng_state(caller_rng)
    result: dict[str, Any] = {
        "protocol": protocol,
        "mode": mode,
        "return_kind": "raw_unclipped",
        "episodes": episodes,
        "initial_policy_rng_sha256": initial,
        "final_policy_rng_sha256": evaluator_final,
        "previous_policy_rng_sha256": prior,
        "scored": True,
    }
    returns = [episode["raw_return"] for episode in episodes]
    if returns:
        result.update({"mean_return": float(np.mean(returns)), "median_return": float(np.median(returns)),
                       "std_return": float(np.std(returns)), "episodes_count": len(returns)})
    return result


def evaluate_formal(policy: Any, env_factory: Any, *, train_seed: int = 0, episodes: int = 100,
                    config: Mapping[str, Any] | None = None, checkpoint_hash: str | None = None,
                    config_hash: str | None = None, rom_hash: str | None = None) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    slug = str((config or {}).get("game", {}).get("slug", ""))
    configured = str((config or {}).get("environment", {}).get("start_protocol", "none"))
    formal_seeds: list[int | None] = [int(train_seed)] + [None] * (episodes - 1)
    return _evaluate(policy, env_factory, seeds=formal_seeds, mode="formal_eval", train_seed=train_seed,
                     protocol="eadream_atari16", fresh_each=False) | {
                         "start_protocol": protocol_for(slug, "formal_eval", configured) if slug else configured,
                         "checkpoint_sha256": checkpoint_hash, "config_sha256": config_hash, "rom_sha256": rom_hash,
                     }


def evaluate_audit(policy: Any, env_factory: Any, *, seeds: Sequence[int] = range(20000, 20030),
                   train_seed: int = 0, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    slug = str((config or {}).get("game", {}).get("slug", ""))
    configured = str((config or {}).get("environment", {}).get("start_protocol", "none"))
    return _evaluate(policy, env_factory, seeds=list(seeds), mode="audit", train_seed=train_seed,
                     protocol="pposb3_audit", fresh_each=True) | {
                         "start_protocol": protocol_for(slug, "audit", configured) if slug else configured,
                     }


def run_no_fire_diagnostic(policy: Any, env_factory: Any, *, slug: str, noop_steps: int = 200,
                           seed: int = 0) -> dict[str, Any]:
    if slug not in {"bowling", "tennis"}:
        raise ValueError("no-FIRE diagnostic is only defined for Bowling and Tennis")
    caller_rng = capture_rng_state()
    env = _make_env(env_factory, mode="upstream_diagnostic", seed=seed)
    native_hashes: set[str] = set(); policy_hashes: set[str] = set(); event_count = 0
    try:
        observation, reset_info = env.reset(seed=seed)
        steps = 0; terminated = truncated = False
        while steps < noop_steps and not (terminated or truncated):
            render = env.render() if hasattr(env, "render") else None
            if render is not None:
                native_hashes.add(hashlib.sha256(np.asarray(render).tobytes()).hexdigest())
            if isinstance(observation, Mapping) and observation.get("image") is not None:
                policy_hashes.add(hashlib.sha256(np.asarray(observation["image"]).tobytes()).hexdigest())
                if observation.get("event") is not None:
                    event_count += int(np.count_nonzero(observation["event"]))
            observation, _, terminated, truncated, info = env.step(0)
            steps += 1
            render = env.render() if hasattr(env, "render") else None
            if render is not None:
                native_hashes.add(hashlib.sha256(np.asarray(render).tobytes()).hexdigest())
            if isinstance(observation, Mapping) and observation.get("image") is not None:
                policy_hashes.add(hashlib.sha256(np.asarray(observation["image"]).tobytes()).hexdigest())
                if observation.get("event") is not None:
                    event_count += int(np.count_nonzero(observation["event"]))
        return {"scored": False, "slug": slug, "start_protocol": "none", "noop_steps": steps,
                "unique_native_frame_hashes": len(native_hashes), "unique_policy_frame_hashes": len(policy_hashes),
                "native_frame_hashes": sorted(native_hashes), "policy_frame_hashes": sorted(policy_hashes),
                "event_pixels": event_count, "reset_noop_frames": int(reset_info.get("reset_noop_frames", 0)) if isinstance(reset_info, Mapping) else 0,
                "termination_reason": "truncated" if truncated else "terminated" if terminated else "step_budget"}
    finally:
        if hasattr(env, "close"):
            env.close()
        restore_rng_state(caller_rng)


def save_result(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    payload = dict(result)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    payload["result_sha256"] = hashlib.sha256(encoded).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = ["evaluate_formal", "evaluate_audit", "run_no_fire_diagnostic", "save_result"]
