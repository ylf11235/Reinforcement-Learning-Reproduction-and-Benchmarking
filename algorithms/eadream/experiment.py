"""Durable EADream Atari experiment lifecycle, logs, and command line tools."""

from __future__ import annotations

import sys

if __package__ in {None, ""}:
    from pathlib import Path as _Path

    ROOT = _Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import argparse
import copy
import json
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from eadream.algorithm import EADream, seed_source
from eadream.config import ConfigError, ResolvedRun, resolve_campaign, resolve_config
from eadream.engine.checkpoint import capture_rng_state, restore_rng_state
from eadream.envs.atari import EADreamAtariAdapter, make_atari_env, protocol_for
from eadream.envs.wrappers import EventObservationWrapper, StartGuard
from eadream.interfaces import TrainingCounters
from eadream.provenance import capture_provenance, canonical_hash, validate_preflight


class RunStateError(RuntimeError):
    """Raised when a durable run cannot make the requested state transition."""


ALLOWED = {
    "PENDING": {"PREFILLING", "FAILED"},
    "PREFILLING": {"TRAINING", "FAILED"},
    "TRAINING": {"EVALUATING", "FAILED"},
    "EVALUATING": {"COMPLETED", "FAILED"},
    "COMPLETED": set(),
    "FAILED": set(),
}


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_text(record: Mapping[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _atomic_json(path: Path, record: Mapping[str, Any]) -> None:
    _atomic_text(path, _json_text(record))


def _atomic_yaml(path: Path, record: Mapping[str, Any]) -> None:
    _atomic_text(path, yaml.safe_dump(dict(record), sort_keys=True, allow_unicode=False))


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


class RunLifecycle:
    """Atomic state journal for one immutable resolved configuration."""

    def __init__(self, run_dir: Path, input_config: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> None:
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "run_state.json"
        self.events_path = self.run_dir / "logs" / "transitions.jsonl"
        self.input_config = dict(input_config)
        self.resolved_config = dict(resolved_config)
        config_hash = self.resolved_config.get("resolved_config_sha256")
        if type(config_hash) is not str or len(config_hash) != 64:
            raise RunStateError("resolved config requires a SHA256 config hash")
        self.config_hash = config_hash
        if self.state_path.is_file():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RunStateError("existing run state is unreadable") from error
            if state.get("resolved_config_sha256") != self.config_hash:
                raise RunStateError("existing output has a different config hash")
            if state.get("status") not in ALLOWED:
                raise RunStateError("existing run state has an invalid status")
            self.state = state
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for relative in ("logs", "tensorboard", "checkpoints", "replay", "evaluations"):
            (self.run_dir / relative).mkdir(exist_ok=True)
        _atomic_yaml(self.run_dir / "input_config.yaml", self.input_config)
        _atomic_yaml(self.run_dir / "resolved_config.yaml", self.resolved_config)
        self.state: dict[str, Any] = {
            "status": "PENDING",
            "resolved_config_sha256": self.config_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counters": _counter_record(TrainingCounters()),
        }
        self._write_transition(previous=None, reason="created")

    def _write_transition(self, *, previous: str | None, reason: str) -> None:
        self.state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.state_path, self.state)
        _append_jsonl(self.events_path, {"at_utc": self.state["updated_at_utc"], "from": previous, "to": self.state["status"], "reason": reason, "resolved_config_sha256": self.config_hash})

    def transition(self, status: str, *, counters: TrainingCounters | None = None, result: Mapping[str, Any] | None = None) -> None:
        if status not in ALLOWED:
            raise RunStateError(f"unknown run status {status!r}")
        previous = self.state["status"]
        if status not in ALLOWED[previous]:
            raise RunStateError(f"invalid run state transition {previous} -> {status}")
        self.state["status"] = status
        if counters is not None:
            self.state["counters"] = _counter_record(counters)
        if result is not None:
            self.state["result"] = dict(result)
        self._write_transition(previous=previous, reason="transition")

    def mark_prefilling(self, counters: TrainingCounters | None = None) -> None:
        self.transition("PREFILLING", counters=counters)

    def mark_training(self, counters: TrainingCounters | None = None) -> None:
        self.transition("TRAINING", counters=counters)

    def mark_evaluating(self, counters: TrainingCounters | None = None) -> None:
        self.transition("EVALUATING", counters=counters)

    def mark_completed(self, result: Mapping[str, Any], counters: TrainingCounters | None = None) -> None:
        self.transition("COMPLETED", counters=counters, result=result)

    def mark_failed(self, error: BaseException, counters: TrainingCounters | None = None) -> None:
        if self.state["status"] in {"COMPLETED", "FAILED"}:
            return
        checkpoint = _last_checkpoint_record(self.run_dir / "checkpoints")
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "last_checkpoint": checkpoint,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.state["failure"] = failure
        self.transition("FAILED", counters=counters)


def _counter_record(counters: TrainingCounters) -> dict[str, int]:
    return {name: int(getattr(counters, name)) for name in ("agent_interactions", "ale_frames", "gradient_updates", "reset_noop_frames", "forced_start_frames", "evaluation_interactions")}


def _last_checkpoint_record(directory: Path) -> dict[str, Any] | None:
    records: list[dict[str, Any]] = []
    for path in directory.rglob("*.json") if directory.is_dir() else ():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if type(record) is dict and isinstance(record.get("agent_interactions"), int):
            records.append({"path": str(path), **record})
    return max(records, key=lambda item: item["agent_interactions"], default=None)


class RunMetricLogger:
    """Durably records JSONL metrics and owns one TensorBoard writer."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "logs" / "metrics.jsonl"
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self._closed = False

    @staticmethod
    def _step(counters: TrainingCounters) -> int:
        return int(counters.agent_interactions)

    def _record(self, kind: str, counters: TrainingCounters, **values: Any) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("metric logger is closed")
        record: dict[str, Any] = {
            "kind": kind,
            **_counter_record(counters),
            "episode_return": values.pop("episode_return", None),
            "episode_length": values.pop("episode_length", None),
            "throughput": float(values.pop("throughput", 0.0)),
            "peak_vram_bytes": int(values.pop("peak_vram_bytes", 0)),
            "monotonic_seconds": time.monotonic(),
            "wall_time_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        _append_jsonl(self.path, record)
        return record

    def scalar(self, tag: str, value: float, counters: TrainingCounters) -> None:
        if type(tag) is not str or not tag:
            raise ValueError("metric tag must be a nonempty string")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("metric values must be finite")
        self._record("scalar", counters, tag=tag, value=numeric)
        step = self._step(counters)
        self.writer.add_scalar(tag, numeric, step)
        self.writer.add_scalar("train/ale_frames", int(counters.ale_frames), step)
        self.writer.add_scalar("train/gradient_updates", int(counters.gradient_updates), step)

    def training(self, metrics: Mapping[str, Any], counters: TrainingCounters, **extra: Any) -> None:
        finite = {str(name): float(value) for name, value in metrics.items() if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))}
        self._record("training", counters, metrics=finite, **extra)
        step = self._step(counters)
        for name, value in finite.items():
            self.writer.add_scalar(f"train/{name}", value, step)
        self.writer.add_scalar("train/ale_frames", int(counters.ale_frames), step)
        self.writer.add_scalar("train/gradient_updates", int(counters.gradient_updates), step)

    def image(self, tag: str, image: np.ndarray | torch.Tensor, counters: TrainingCounters) -> None:
        self._record("image", counters, tag=tag)
        self.writer.add_image(tag, image, self._step(counters), dataformats="HWC" if np.asarray(image).ndim == 3 else "CHW")

    def video(self, tag: str, video: np.ndarray | torch.Tensor, counters: TrainingCounters, fps: int = 30) -> None:
        self._record("video", counters, tag=tag, fps=int(fps))
        self.writer.add_video(tag, video, self._step(counters), fps=int(fps))

    def flush(self) -> None:
        if not self._closed:
            self.writer.flush()

    def close(self) -> None:
        if not self._closed:
            self.writer.flush()
            self.writer.close()
            self._closed = True


def _run_directory(config: Mapping[str, Any]) -> Path:
    output = config.get("output", {})
    root = Path(output["campaign_root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    slug = str(config["game"]["slug"])
    seed = int(config["game"]["seed"])
    return root / "games" / slug / f"seed_{seed:03d}"


def _default_env_factory(config: Mapping[str, Any]) -> Callable[[], Any]:
    return lambda: make_atari_env(config, mode="train")


def _metric_snapshot(agent: Any) -> Mapping[str, Any]:
    return getattr(getattr(agent, "driver", None), "last_train_metrics", {})


def _driver_metric_fields(agent: Any) -> dict[str, Any]:
    driver = getattr(agent, "driver", None)
    fields = {
        "episode_return": getattr(driver, "last_episode_return", None),
        "episode_length": getattr(driver, "last_episode_length", None),
        "throughput": 0.0,
        "peak_vram_bytes": 0,
    }
    started = getattr(driver, "run_started_monotonic", None)
    counters = getattr(agent, "counters", None)
    if started is not None and counters is not None:
        elapsed = max(time.monotonic() - float(started), 1e-9)
        fields["throughput"] = float(counters.agent_interactions) / elapsed
    if torch.cuda.is_available():
        try:
            fields["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        except (AssertionError, RuntimeError):
            pass
    return fields


def _diagnostic_targets(current: int, budget: int, *, prefill: int) -> list[int]:
    """Return source diagnostic milestones strictly after ``current``."""
    if budget <= 0:
        return []
    milestones: list[int] = []
    if prefill <= budget and prefill > current:
        milestones.append(prefill)
    first = 10_000
    cadence = 7_500
    milestone = first
    while milestone <= budget:
        if milestone > current:
            milestones.append(milestone)
        milestone += cadence
    return milestones


def _log_open_loop_panel(agent: Any, logger: RunMetricLogger) -> None:
    """Emit the Task 7 panel without perturbing learner or sampler RNG state."""
    driver = getattr(agent, "driver", None)
    engine = getattr(agent, "engine", None)
    sampler = getattr(driver, "_sampler", None)
    world_model = getattr(engine, "world_model", None)
    if sampler is None or world_model is None:
        return
    global_rng = capture_rng_state()
    sampler_rng = copy.deepcopy(sampler.state_dict())
    try:
        batch = sampler.sample(1, int(agent.config["training"]["batch_length"]))
        panel = world_model.video_pred(batch)
        logger.video(
            "diagnostic/open_loop_world_model",
            panel.detach().cpu().permute(0, 1, 4, 2, 3),
            agent.counters,
        )
    finally:
        sampler.load_state_dict(sampler_rng)
        restore_rng_state(global_rng)


def _preflight_env(factory: Callable[..., Any], config: Mapping[str, Any], mode: str) -> Any:
    """Construct the event-bearing environment used by Gate 3."""
    try:
        env = factory(config, mode=mode)
    except TypeError:
        env = factory(config, mode)
    if not isinstance(env, EventObservationWrapper):
        # Test doubles and alternate factories may return the raw 210x160 ALE
        # object.  Bring it through the exact production adapter/wrapper order
        # before checking the policy contract.
        space = getattr(env, "observation_space", None)
        if getattr(space, "shape", None) == (210, 160, 3):
            environment = config.get("environment", {})
            env = EADreamAtariAdapter(
                env,
                action_repeat=int(environment.get("action_repeat", 4)),
                size=int(environment.get("screen_size", 64)),
                noop_max_exclusive=int(environment.get("noop_max_exclusive", 30)),
                verify_render_source=mode == "video",
            )
            protocol = protocol_for(
                str(config.get("game", {}).get("slug", "")),
                mode,
                str(environment.get("start_protocol", "none")),
            )
            env = StartGuard(env, protocol=protocol)
        env = EventObservationWrapper(env)
    return env


def _preflight_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    if type(observation) is not dict:
        raise ValueError("preflight observation must be a plain dict")
    expected = {"image", "event", "is_first", "is_terminal"}
    if set(observation) != expected:
        raise ValueError(f"preflight observation keys must be {sorted(expected)}")
    image = np.asarray(observation["image"])
    event = np.asarray(observation["event"])
    if image.shape != (64, 64, 3) or image.dtype != np.uint8:
        raise ValueError("policy observation image must be uint8 [64,64,3]")
    if event.shape != (64, 64) or event.dtype != np.uint8:
        raise ValueError("policy observation event must be uint8 [64,64]")
    if type(observation["is_first"]) is not bool or type(observation["is_terminal"]) is not bool:
        raise ValueError("policy boundary flags must be bool")
    return {
        "policy_shape": list(image.shape),
        "policy_dtype": str(image.dtype),
        "event_shape": list(event.shape),
        "event_dtype": str(event.dtype),
    }


def _preflight_probe_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create a cheap non-formal model solely for recurrent/checkpoint wiring."""
    probe = copy.deepcopy(dict(config))
    probe.setdefault("experiment", {})["formal"] = False
    probe.setdefault("runtime", {}).update({"device": "cpu", "require_cuda": False, "compile": False})
    model = probe.setdefault("model", {})
    model.update({"dyn_stoch": 2, "dyn_discrete": 4, "dyn_deter": 16, "dyn_hidden": 16, "units": 16, "cnn_depth": 4, "encoder_embed_dim": 512})
    training = probe.setdefault("training", {})
    training.update({"batch_size": 1, "batch_length": 2, "prefill": 1, "pretrain_batches": 1})
    return probe


def run_atari16_preflight(
    campaign: Any,
    *,
    env_factory: Callable[..., Any] | None = None,
    output_path: Path | None = None,
    interactions: int = 32,
) -> dict[str, Any]:
    """Run Gate 3 checks for each installed Atari-16 game.

    The function is deliberately the only API that creates live ALE objects
    for preflight.  It is not called by non-integration tests.
    """
    from eadream.config import ResolvedCampaign, resolve_campaign

    if isinstance(campaign, (str, Path)):
        campaign = resolve_campaign(Path(campaign))
    if not isinstance(campaign, ResolvedCampaign):
        raise TypeError("campaign must be a ResolvedCampaign or campaign path")
    if type(interactions) is not int or interactions <= 0:
        raise ValueError("interactions must be a positive integer")
    factory = env_factory or make_atari_env
    by_game: dict[str, ResolvedRun] = {}
    for run in campaign.runs:
        by_game.setdefault(str(run.config["game"]["slug"]), run)
    records: list[dict[str, Any]] = []
    for slug, run in by_game.items():
        config = run.config
        record: dict[str, Any] = {"game": slug, "seed": int(run.seed), "status": "FAILED", "errors": []}
        train_env = formal_env = audit_env = diagnostic_env = video_env = None
        agent = None
        try:
            train_env = _preflight_env(factory, config, "train")
            validation = validate_preflight(config, train_env)
            record["validation"] = validation
            observation, reset_info = train_env.reset(seed=int(run.seed))
            record.update(_preflight_observation(observation))
            record["reset_noop_frames"] = int(reset_info.get("reset_noop_frames", 0))
            record["forced_start_actions"] = int(reset_info.get("forced_start_actions", 0))
            record["forced_start_frames"] = int(reset_info.get("forced_start_frames", 0))

            action_count = int(getattr(train_env.action_space, "n"))
            rewards: list[float] = []
            event_nonzero_steps = 0
            for index in range(interactions):
                action = index % action_count
                next_observation, reward, terminated, truncated, info = train_env.step(action)
                _preflight_observation(next_observation)
                if not np.isfinite(float(reward)):
                    raise ValueError("preflight reward is not finite")
                if "raw_reward" in info and float(info["raw_reward"]) != float(reward):
                    raise ValueError("preflight raw_reward differs from returned reward")
                event_nonzero_steps += int(np.any(np.asarray(next_observation["event"])))
                rewards.append(float(reward))
                if terminated or truncated:
                    next_observation, reset_info = train_env.reset(seed=None)
                    _preflight_observation(next_observation)
            record["steps"] = interactions
            record["raw_reward_finite"] = all(np.isfinite(rewards))
            record["event_checked"] = True
            record["event_nonzero_steps"] = event_nonzero_steps

            formal_env = _preflight_env(factory, config, "formal_eval")
            formal_observation, formal_info = formal_env.reset(seed=int(run.seed))
            formal_contract = _preflight_observation(formal_observation)
            record["formal_policy_shape"] = formal_contract["policy_shape"]
            record["formal_forced_start_actions"] = int(formal_info.get("forced_start_actions", 0))

            audit_env = _preflight_env(factory, config, "audit")
            audit_observation, audit_info = audit_env.reset(seed=int(run.seed))
            audit_contract = _preflight_observation(audit_observation)
            record["audit_policy_shape"] = audit_contract["policy_shape"]
            record["audit_forced_start_actions"] = int(audit_info.get("forced_start_actions", 0))

            video_env = _preflight_env(factory, config, "video")
            video_observation, _ = video_env.reset(seed=int(run.seed))
            _preflight_observation(video_observation)
            frame = np.asarray(video_env.render())
            if frame.shape != (210, 160, 3) or frame.dtype != np.uint8:
                raise ValueError("native render must be uint8 [210,160,3]")
            record["render_shape"] = list(frame.shape)
            record["render_dtype"] = str(frame.dtype)

            # Use a reduced non-formal probe to test recurrent inference and
            # checkpoint serialization without spending formal interactions.
            probe_config = _preflight_probe_config(config)
            probe_config["action_dim"] = action_count
            agent = EADream(train_env, probe_config, env_factory=lambda: _preflight_env(factory, probe_config, "train"))
            action, _state = agent.predict(observation, episode_start=True, deterministic=True)
            if action.shape != (action_count,) or action.dtype != np.float32:
                raise ValueError("recurrent probe returned an invalid action")
            record["recurrent_inference"] = True
            with tempfile.TemporaryDirectory(prefix="eadream-preflight-") as temporary:
                checkpoint = Path(temporary) / "probe.pt"
                agent.save_checkpoint(checkpoint)
                reloaded_env = _preflight_env(factory, probe_config, "train")
                reloaded = EADream(reloaded_env, probe_config, env_factory=lambda: _preflight_env(factory, probe_config, "train"))
                reloaded.load_checkpoint(checkpoint)
                reloaded.close()
            record["checkpoint_reload"] = True

            if slug in {"bowling", "tennis"}:
                diagnostic_env = _preflight_env(factory, config, "upstream_diagnostic")
                diagnostic, _ = diagnostic_env.reset(seed=int(run.seed))
                meanings = list(diagnostic_env.unwrapped.get_action_meanings())
                noop = meanings.index("NOOP")
                frames: list[bytes] = []
                for _ in range(200):
                    frames.append(np.asarray(diagnostic["image"]).tobytes())
                    diagnostic, _, terminated, truncated, _ = diagnostic_env.step(noop)
                    if terminated or truncated:
                        diagnostic, _ = diagnostic_env.reset(seed=None)
                record["no_fire_unique_frames"] = len(set(frames))
                record["fire_guard"] = all(
                    record[name] >= 1
                    for name in (
                        "forced_start_actions",
                        "formal_forced_start_actions",
                        "audit_forced_start_actions",
                    )
                )
                if not record["fire_guard"]:
                    raise ValueError("Bowling/Tennis FIRE guard did not trigger in every scored mode")
            record["status"] = "PASSED"
        except Exception as error:
            record["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            if agent is not None:
                try:
                    agent.close()
                except Exception:
                    pass
            for env in (diagnostic_env, audit_env, formal_env, video_env, train_env):
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        records.append(record)

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "runs" / "preflight" / "preflight_summary.json"
    output_path = Path(output_path)
    game_directory = output_path.parent / "games"
    for record in records:
        _atomic_json(game_directory / f"{record['game']}.json", record)

    summary = {
        "schema_version": 1,
        "gate": "3",
        "campaign_id": campaign.campaign_id,
        "passed": sum(item["status"] == "PASSED" for item in records),
        "failed": sum(item["status"] != "PASSED" for item in records),
        "games": records,
        "formal_scores": False,
    }
    _atomic_json(output_path, summary)
    return summary


def run_one(
    resolved_config: Mapping[str, Any],
    *,
    input_config: Mapping[str, Any] | None = None,
    env_factory: Callable[[], Any] | None = None,
    agent_factory: Callable[..., Any] = EADream,
    provenance_factory: Callable[[Mapping[str, Any], Any], Mapping[str, Any]] = capture_provenance,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute one resolved seed without allowing a semantic CLI override."""
    config = dict(resolved_config)
    run_dir = _run_directory(config)
    lifecycle = RunLifecycle(
        run_dir,
        config if input_config is None else input_config,
        config,
    )
    if lifecycle.state["status"] == "COMPLETED":
        from eadream.campaign import _compatible_completed
        if not _compatible_completed(ResolvedRun(config, str(config["resolved_config_sha256"]), int(config["game"]["seed"]), run_dir)):
            raise RunStateError("completed run has a missing or incompatible final checkpoint")
        return dict(lifecycle.state)
    if resume or lifecycle.state["status"] != "PENDING":
        raise RunStateError("cannot resume an incomplete EADream run: no periodic recovery checkpoint is saved; use a new output root")
    maker = env_factory or _default_env_factory(config)
    logger = RunMetricLogger(run_dir)
    env = None
    agent = None
    try:
        # Pin drift is a preflight failure, before an Atari object can step or
        # a lifecycle can claim that prefill has begun.
        validate_preflight(config)
        seed_source(int(config["game"]["seed"]))
        env = maker()
        provenance_path = run_dir / "provenance.json"
        if resume and provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        else:
            provenance = dict(provenance_factory(config, env))
        provenance_hash = provenance.get("provenance_sha256") or canonical_hash(provenance)
        provenance["provenance_sha256"] = provenance_hash
        _atomic_json(provenance_path, provenance)
        agent = agent_factory(env, config, env_factory=maker)
        # Checkpoints identify the immutable resolved experiment, rather than
        # runtime-only additions such as action meanings or an output path.
        if hasattr(agent, "config_hash"):
            agent.config_hash = str(config["resolved_config_sha256"])
        if hasattr(agent, "provenance_hash"):
            agent.provenance_hash = str(provenance_hash)
        if resume:
            checkpoint = run_dir / "checkpoints" / "final.pt"
            if not checkpoint.is_file() or not hasattr(agent, "load_checkpoint"):
                raise RunStateError("resume requires an existing compatible checkpoint")
            agent.load_checkpoint(checkpoint)
        counters = agent.counters
        budget = int(config["budget"]["agent_interactions"])
        prefill_target = min(int(config["training"]["prefill"]), budget)
        if lifecycle.state["status"] == "PENDING":
            lifecycle.mark_prefilling(counters)
        if counters.agent_interactions < prefill_target:
            agent.learn(prefill_target)
            logger.training(_metric_snapshot(agent), agent.counters, phase="prefill", **_driver_metric_fields(agent))
        # The source performs 100 pretraining updates at exactly interaction
        # 2,500, after the deferred engine is handed off from collection.
        if counters.agent_interactions == prefill_target and prefill_target == int(config["training"]["prefill"]):
            driver = getattr(agent, "driver", None)
            if getattr(agent, "engine", None) is None and hasattr(driver, "_ensure_engine"):
                driver._ensure_engine()
                if hasattr(driver, "_pretrain_once") and driver._pretrain_once():
                    driver._run_updates(first_learned_call=True)
            _log_open_loop_panel(agent, logger)
        lifecycle.mark_training(agent.counters)
        periodic_first = int(config.get("checkpoint", {}).get("first_periodic_agent_interactions", budget))
        periodic_every = int(config.get("checkpoint", {}).get("save_every_agent_interactions", budget))
        targets: list[int] = []
        if periodic_every > 0:
            target = periodic_first
            while target < budget:
                if target > agent.counters.agent_interactions:
                    targets.append(target)
                target += periodic_every
        if budget > agent.counters.agent_interactions:
            targets.append(budget)
        for target in targets:
            agent.learn(target)
            logger.training(_metric_snapshot(agent), agent.counters, phase="training", **_driver_metric_fields(agent))
            if target in _diagnostic_targets(target - 1, budget, prefill=prefill_target):
                _log_open_loop_panel(agent, logger)
        lifecycle.mark_evaluating(agent.counters)
        checkpoint_record: Any = None
        if hasattr(agent, "save_checkpoint"):
            checkpoint_record = agent.save_checkpoint(run_dir / "checkpoints" / "final.pt")
        result = {"agent_interactions": int(agent.counters.agent_interactions), "ale_frames": int(agent.counters.ale_frames), "gradient_updates": int(agent.counters.gradient_updates), "checkpoint": None if checkpoint_record is None else str(getattr(checkpoint_record, "path", checkpoint_record))}
        lifecycle.mark_completed(result, agent.counters)
        return dict(lifecycle.state)
    except BaseException as error:
        lifecycle.mark_failed(error, getattr(agent, "counters", None))
        raise
    finally:
        logger.close()
        if agent is not None:
            agent.close()
        elif env is not None:
            closer = getattr(env, "close", None)
            if callable(closer):
                closer()


def _load_input_config(path: Path) -> Mapping[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise ConfigError("configuration must be a mapping")
    return document


def _resolved_runs(path: Path, game: str | None, seed: int | None) -> tuple[ResolvedRun, ...]:
    raw = _load_input_config(path)
    if "campaign_id" in raw:
        return resolve_campaign(path, game_slug=game, seed=seed).runs
    if game is not None and game != raw.get("game", {}).get("slug"):
        raise ConfigError(f"game {game!r} does not match single-game YAML")
    if seed is None:
        raise ConfigError("single-game YAML requires --seed from suite.seeds")
    config, digest = resolve_config(path, seed=seed)
    return (ResolvedRun(config=config, config_hash=digest, seed=seed, run_dir=_run_directory(config)),)


def status_records(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir)
    paths = [path / "run_state.json"] if (path / "run_state.json").is_file() else sorted(path.rglob("run_state.json"))
    records = []
    for state_path in paths:
        try:
            records.append(json.loads(state_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            records.append({"status": "INVALID", "path": str(state_path)})
    return records


def summarize_run(run_dir: Path) -> dict[str, Any]:
    records = status_records(run_dir)
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "INVALID"))
        counts[status] = counts.get(status, 0) + 1
    report = {"run_dir": str(run_dir), "run_count": len(records), "status_counts": counts, "agent_interactions": sum(int(record.get("counters", {}).get("agent_interactions", 0)) for record in records)}
    _atomic_json(Path(run_dir) / "summary.json", report)
    return report


def archive_run(run_dir: Path, archive_root: Path) -> Path:
    from eadream.archive import archive_run as _archive_run
    try:
        return _archive_run(run_dir, archive_root)
    except ValueError as error:
        raise RunStateError(str(error)) from error


def run_campaign(config_path: Path, *, retry_failed: bool = False, measured_run_bytes: int | None = None, game_slug: str | None = None, seed: int | None = None) -> dict[str, Any]:
    """Resolve and execute the complete serial EADream campaign."""
    from eadream.campaign import run_campaign as _run_campaign
    campaign = resolve_campaign(Path(config_path), game_slug=game_slug, seed=seed)
    return _run_campaign(campaign, retry_failed=retry_failed, measured_run_bytes=measured_run_bytes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eadream.experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--game")
        command.add_argument("--seed", type=int)
        if name == "run":
            command.add_argument("--resume", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--run-dir", type=Path, required=True)
    archive = sub.add_parser("archive")
    archive.add_argument("--run-dir", type=Path, required=True)
    archive.add_argument("--archive-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status_records(args.run_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "summarize":
        print(json.dumps(summarize_run(args.run_dir), indent=2, sort_keys=True))
        return 0
    if args.command == "archive":
        archive_root = args.archive_root
        if archive_root is None:
            resolved_path = args.run_dir / "resolved_config.yaml"
            if not resolved_path.is_file():
                raise RunStateError("archive requires --archive-root when resolved config is unavailable")
            archive_root = Path(yaml.safe_load(resolved_path.read_text(encoding="utf-8"))["output"]["archive_root"])
        print(str(archive_run(args.run_dir, archive_root)))
        return 0
    runs = _resolved_runs(args.config, args.game, args.seed)
    if args.command == "validate":
        print(json.dumps([{ "game": run.config["game"]["slug"], "seed": run.seed, "resolved_config_sha256": run.config_hash} for run in runs], indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        if "campaign_id" in _load_input_config(args.config):
            from eadream.reference import run_static_gates

            static = run_static_gates(
                resolve_campaign(args.config),
                test_command=[sys.executable, "-m", "pytest", "-q", "eadream/tests"],
            )
            if not static["passed"]:
                print(json.dumps({"static": static, "live": None}, indent=2, sort_keys=True))
                return 1
            summary = run_atari16_preflight(resolve_campaign(args.config, game_slug=args.game, seed=args.seed))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0 if summary["failed"] == 0 else 1
        records = []
        for run in runs:
            validate_preflight(run.config)
            seed_source(run.seed)
            env = make_atari_env(run.config, mode="train")
            try:
                records.append(capture_provenance(run.config, env))
            finally:
                env.close()
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    if "campaign_id" in _load_input_config(args.config):
        if args.resume:
            raise RunStateError("cannot resume an incomplete EADream campaign; completed runs are skipped automatically")
        selection = {} if args.game is None and args.seed is None else {"game_slug": args.game, "seed": args.seed}
        print(json.dumps(run_campaign(args.config, retry_failed=False, **selection), indent=2, sort_keys=True))
        return 0
    records = [run_one(run.config, input_config=_load_input_config(args.config), resume=bool(args.resume)) for run in runs]
    print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
