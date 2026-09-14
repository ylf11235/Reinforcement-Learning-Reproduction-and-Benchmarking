"""Minimal Atari preflight runner for the PPO-SB3 campaign."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import ale_py
import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[2]  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_PARENT = ROOT / "algorithms"  # makes `import ppo_sb3` work from the repository root
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from baselines_common.envs.atari import StartProtocol, make_atari_env
from baselines_common.envs.video import record_videos
from baselines_common.logging import JsonlLogger
from baselines_common.utils import atomic_write_json, atomic_write_yaml, sha256_file
from ppo_sb3.callbacks import CampaignCallback
from ppo_sb3.config import (
    ConfigError,
    is_phase_manifest_with_campaign,
    load_game_config,
    resolve_campaign_configs,
    resolve_config,
    resolve_phase1_campaign,
)
from ppo_sb3.campaign import import_completed_games, run_phase_campaign
from ppo_sb3.archive import archive_campaign, build_summary
from ppo_sb3.evaluation import (
    checkpoint_path_from_metadata,
    evaluate_raw,
    save_model_atomic,
)
from ppo_sb3.provenance import capture_campaign_provenance, capture_game_provenance
from ppo_sb3.train import (
    build_resolved_model,
    configure_runtime,
    load_resolved_model,
    make_resolved_eval_env,
    make_resolved_train_env,
)


class RunStateError(RuntimeError):
    """Raised when a run directory cannot safely advance or resume."""


class RunLifecycle:
    """Persist the terminal-state contract for one game and seed combination."""

    _TERMINAL = {"COMPLETED", "FAILED"}

    def __init__(
        self,
        run_dir: Path,
        input_config: dict[str, Any],
        resolved_config: dict[str, Any],
        *,
        config_hash: str,
    ) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "run_state.json"
        self.config_hash = config_hash
        if self.state_path.is_file():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self._state.get("config_hash") != config_hash:
                raise RunStateError(
                    f"run directory config hash mismatch: {self._state.get('config_hash')} != {config_hash}"
                )
        else:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_yaml(self.run_dir / "input_config.yaml", input_config)
            atomic_write_yaml(self.run_dir / "resolved_config.yaml", resolved_config)
            self._state = {
                "schema_version": 1,
                "status": "PENDING",
                "config_hash": config_hash,
                "created_at_unix": time.time(),
                "updated_at_unix": time.time(),
            }
            self._write()

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def _write(self) -> None:
        self._state["updated_at_unix"] = time.time()
        atomic_write_json(self.state_path, self._state)

    def mark_running(self, *, resume: bool) -> None:
        status = self._state["status"]
        allowed = {"PENDING"}
        if resume:
            allowed.update({"RUNNING", "EVALUATING", "FAILED"})
        if status not in allowed:
            raise RunStateError(f"cannot start {status} run without compatible --resume")
        self._state["status"] = "RUNNING"
        self._state["resume"] = bool(resume)
        self._write()

    def mark_evaluating(self) -> None:
        if self._state["status"] != "RUNNING":
            raise RunStateError(f"cannot evaluate run in state {self._state['status']}")
        self._state["status"] = "EVALUATING"
        self._write()

    def mark_completed(self, result: dict[str, Any]) -> None:
        if self._state["status"] != "EVALUATING":
            raise RunStateError(f"cannot complete run in state {self._state['status']}")
        self._state["status"] = "COMPLETED"
        self._state["result"] = result
        self._write()

    def mark_failed(self, error: BaseException) -> None:
        if self._state["status"] == "COMPLETED":
            raise RunStateError("cannot mark a completed run as failed")
        checkpoint_dir = self.run_dir / "checkpoints" / "last"
        model_path = checkpoint_path_from_metadata(checkpoint_dir)
        metadata_path = checkpoint_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}
        self._state["status"] = "FAILED"
        self._state["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "last_known_timestep": metadata.get("transitions"),
            "last_checkpoint": str(model_path) if model_path.is_file() else None,
            "last_checkpoint_sha256": metadata.get("model_sha256"),
        }
        self._write()


@dataclass
class TrainingSession:
    """Resources kept live until the final raw-score evaluation is written."""

    model: PPO
    train_env: Any
    eval_env: Any
    logger: JsonlLogger
    run_dir: Path
    config_hash: str
    action_count: int
    device: str
    started_at_unix: float

    def close(self) -> None:
        self.eval_env.close()
        self.train_env.close()
        self.logger.close()


def _run_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(config["output"]["run_root"]))
    return configured if configured.is_absolute() else ROOT / configured


def _input_config_for(resolved_config: dict[str, Any]) -> dict[str, Any]:
    source = resolved_config.get("source_config")
    if not isinstance(source, str):
        raise ConfigError("resolved configuration has no source_config")
    return load_game_config(Path(source))


def _periodic_seeds(config: dict[str, Any]) -> list[int]:
    seeds = config["evaluation"].get("periodic_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("evaluation.periodic_seeds must be a non-empty list")
    return [int(seed) for seed in seeds]


def _audit_seeds(config: dict[str, Any]) -> list[int]:
    evaluation = config["evaluation"]
    configured = evaluation.get("audit_seeds")
    if isinstance(configured, list) and configured:
        return [int(seed) for seed in configured]
    count = int(evaluation["audit_episodes"])
    start = int(evaluation.get("audit_seed_start", 20_000))
    if count <= 0:
        raise ConfigError("evaluation.audit_episodes must be positive")
    return list(range(start, start + count))


def _default_train(
    config: dict[str, Any], run_dir: Path, resume: bool
) -> TrainingSession:
    device = configure_runtime(config)
    train_env = make_resolved_train_env(config)
    eval_env = make_resolved_eval_env(config)
    started_at_unix = time.time()
    try:
        action_count = int(train_env.action_space.n)
        logger = JsonlLogger(
            str(run_dir),
            {
                "resolved_config_sha256": config["resolved_config_sha256"],
                "env_id": config["game"]["env_id"],
                "game": config["game"]["slug"],
                "n_actions": action_count,
                "observation_shape": list(train_env.observation_space.shape),
                "device": device,
                "train_clip_reward": config["environment"]["train_clip_reward"],
                "eval_clip_reward": False,
            },
        )
        last_checkpoint = checkpoint_path_from_metadata(run_dir / "checkpoints" / "last")
        if resume:
            if not last_checkpoint.is_file():
                raise RunStateError(f"--resume requested but no last checkpoint exists: {last_checkpoint}")
            model = load_resolved_model(str(last_checkpoint), train_env, device=device)
        else:
            model = build_resolved_model(
                config, train_env, device=device, tensorboard_log=str(run_dir / "tensorboard")
            )

        effective = int(config["budget"]["effective"])
        completed = int(model.num_timesteps)
        if completed > effective:
            raise RunStateError(
                f"checkpoint has {completed} timesteps beyond configured target {effective}"
            )
        callback = CampaignCallback(
            config,
            eval_env,
            logger,
            run_dir,
            str(config["resolved_config_sha256"]),
        )
        remaining = effective - completed
        if remaining:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        # A resumed model at the exact target may not execute another rollout.
        callback._save_last(model)
        return TrainingSession(
            model=model,
            train_env=train_env,
            eval_env=eval_env,
            logger=logger,
            run_dir=run_dir,
            config_hash=str(config["resolved_config_sha256"]),
            action_count=action_count,
            device=device,
            started_at_unix=started_at_unix,
        )
    except Exception:
        eval_env.close()
        train_env.close()
        raise


def _default_finalize(session: TrainingSession, config: dict[str, Any]) -> dict[str, Any]:
    final_checkpoint = session.run_dir / "final_checkpoint" / "model.zip"
    saved_final = save_model_atomic(session.model, final_checkpoint)
    final_evaluation = evaluate_raw(
        session.model,
        session.eval_env,
        _audit_seeds(config),
        deterministic=bool(config["evaluation"]["deterministic"]),
    )
    final_evaluation["transitions"] = int(session.model.num_timesteps)
    final_evaluation["kind"] = "final_raw_audit"
    atomic_write_json(session.run_dir / "evaluation" / "final_raw_audit.json", final_evaluation)
    atomic_write_json(
        final_checkpoint.parent / "metadata.json",
        {
            "checkpoint_path": str(saved_final.path),
            "config_hash": session.config_hash,
            "model_sha256": saved_final.sha256,
            "transitions": int(session.model.num_timesteps),
            "evaluation": final_evaluation,
        },
    )
    session.logger.evaluate(final_evaluation)
    result = {
        "action_count": session.action_count,
        "device": session.device,
        "effective_timesteps": int(session.model.num_timesteps),
        "final_checkpoint": str(saved_final.path),
        "final_checkpoint_sha256": saved_final.sha256,
        "final_raw_return_mean": final_evaluation["mean"],
        "final_raw_return_median": final_evaluation["median"],
        "final_raw_return_std": final_evaluation["std"],
        "training_wall_time_s": time.time() - session.started_at_unix,
    }
    video = config.get("video", {})
    if bool(video.get("enabled", False)):
        checkpoint_kind = str(video.get("checkpoint", "best"))
        if checkpoint_kind != "best":
            raise ConfigError(f"unsupported video checkpoint selection: {checkpoint_kind}")
        best_checkpoint = checkpoint_path_from_metadata(
            session.run_dir / "best_checkpoint"
        )
        if not best_checkpoint.is_file():
            raise RunStateError(f"video requires best checkpoint: {best_checkpoint}")
        seeds = video.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("video.seeds must be a non-empty list when video is enabled")
        video_model = PPO.load(str(best_checkpoint), device=session.device)
        video_result = record_videos(
            video_model,
            session.eval_env,
            checkpoint=best_checkpoint,
            seeds=[int(seed) for seed in seeds],
            output_dir=session.run_dir / "videos" / checkpoint_kind,
            fps=int(video["fps"]),
            config_hash=session.config_hash,
            deterministic=bool(video.get("deterministic", config["evaluation"]["deterministic"])),
            codec=str(video.get("codec", "h264")),
            pixel_format=str(video.get("pixel_format", "yuv420p")),
        )
        result["video_episodes"] = len(video_result["episodes"])
        result["videos_index"] = str(session.run_dir / "videos" / checkpoint_kind / "episodes.json")
    return result


def run_game(
    config: dict[str, Any],
    *,
    resume: bool = False,
    retry_failed: bool = False,
    input_config: dict[str, Any] | None = None,
    train_fn: Callable[[dict[str, Any], Path, bool], Any] | None = None,
    finalizer: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run or resume one resolved game configuration through its terminal state."""
    config_hash = str(config["resolved_config_sha256"])
    lifecycle = RunLifecycle(
        _run_dir(config),
        input_config if input_config is not None else _input_config_for(config),
        config,
        config_hash=config_hash,
    )
    if lifecycle.state["status"] == "COMPLETED":
        return lifecycle.state

    allow_resume = bool(resume or retry_failed)
    session: Any = None
    try:
        campaign_context = config.get("campaign_context")
        if isinstance(campaign_context, dict):
            campaign_root = Path(str(campaign_context["run_root"]))
            campaign_provenance_path = campaign_root / "campaign" / "provenance.json"
            if not campaign_provenance_path.is_file():
                raise RunStateError("campaign provenance must exist before a Phase-1 game starts")
            campaign_provenance = json.loads(campaign_provenance_path.read_text(encoding="utf-8"))
            capture_game_provenance(lifecycle.run_dir, config, campaign_provenance)
        lifecycle.mark_running(resume=allow_resume)
        trainer = train_fn or _default_train
        session = trainer(config, lifecycle.run_dir, allow_resume)
        lifecycle.mark_evaluating()
        result = (finalizer or _default_finalize)(session, config)
        lifecycle.mark_completed(result)
        return lifecycle.state
    except Exception as error:
        lifecycle.mark_failed(error)
        raise
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def run_campaign(
    config_path: Path, *, game_slug: str | None = None, resume: bool = False, retry_failed: bool = False
) -> list[dict[str, Any]]:
    """Resolve a game YAML or manifest and execute selected games sequentially."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"invalid configuration: {config_path}")
    if is_phase_manifest_with_campaign(config_path):
        if game_slug is not None:
            raise ConfigError("Phase-1 campaign always runs its complete manifest order")
        if resume:
            raise ConfigError("Phase-1 automatic resume is disabled; use --retry-failed")
        campaign = resolve_phase1_campaign(config_path)
        capture_campaign_provenance(campaign.controls.run_root, campaign, sys.argv)
        return [run_phase_campaign(campaign, retry_failed=retry_failed)]
    if "game" in raw:
        if game_slug is not None and raw.get("game", {}).get("slug") != game_slug:
            raise ConfigError(f"game {game_slug!r} does not match {config_path}")
        resolved = [resolve_config(config_path)]
    else:
        resolved = resolve_campaign_configs(config_path, game_slug=game_slug)

    records: list[dict[str, Any]] = []
    for config, _ in resolved:
        try:
            records.append(run_game(config, resume=resume, retry_failed=retry_failed))
        except Exception as error:
            records.append(
                {
                    "status": "FAILED",
                    "game": config["game"]["slug"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            campaign = config.get("campaign", {})
            if not bool(campaign.get("continue_after_failure", False)):
                raise
    return records


def status_records(run_dir: Path) -> list[dict[str, Any]]:
    """Read persisted run states without loading ALE, PyTorch, or a checkpoint."""
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    campaign_state = run_dir / "campaign" / "campaign_state.json"
    if campaign_state.is_file():
        state = json.loads(campaign_state.read_text(encoding="utf-8"))
        return [{"run_dir": str(run_dir), **state}]
    records: list[dict[str, Any]] = []
    for state_path in sorted(run_dir.rglob("run_state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        records.append({"run_dir": str(state_path.parent), **state})
    return records


@dataclass(frozen=True)
class PreflightGame:
    slug: str
    env_id: str
    start_protocol: StartProtocol


@dataclass(frozen=True)
class PreflightConfig:
    path: Path
    experiment_id: str
    games: list[PreflightGame]
    device: str
    require_device: bool
    torch_deterministic: bool
    cudnn_benchmark: bool
    repeat_action_probability: float
    max_num_frames_per_episode: int
    noop_max: int
    frame_skip: int
    screen_size: int
    stack: int
    train_clip_reward: bool
    eval_clip_reward: bool
    render_mode: str
    seed: int
    training_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    eval_agent_steps: int
    eval_seed: int
    policy: str
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    run_root: Path


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _required(mapping: dict[str, Any], field: str) -> Any:
    if field not in mapping:
        raise ValueError(f"missing required field: {field}")
    return mapping[field]


def load_preflight_config(path: Path) -> PreflightConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    experiment = _mapping(_required(root, "experiment"), "experiment")
    suite = _mapping(_required(root, "suite"), "suite")
    runtime = _mapping(_required(root, "runtime"), "runtime")
    environment = _mapping(_required(root, "environment"), "environment")
    preflight = _mapping(_required(root, "preflight"), "preflight")
    output = _mapping(_required(root, "output"), "output")

    games_raw = _required(suite, "games")
    if not isinstance(games_raw, list):
        raise ValueError("suite.games must be a list")
    games: list[PreflightGame] = []
    for entry in games_raw:
        game = _mapping(entry, "suite.games entry")
        protocol = str(_required(game, "start_protocol"))
        if protocol not in {"none", "fire", "move"}:
            raise ValueError(f"invalid start protocol for {game.get('slug')}: {protocol}")
        games.append(
            PreflightGame(
                slug=str(_required(game, "slug")),
                env_id=str(_required(game, "env_id")),
                start_protocol=protocol,  # type: ignore[arg-type]
            )
        )

    expected_count = int(_required(suite, "expected_game_count"))
    if len(games) != expected_count or len({game.slug for game in games}) != len(games):
        raise ValueError("suite.games must have the expected number of unique slugs")

    return PreflightConfig(
        path=path,
        experiment_id=str(_required(experiment, "id")),
        games=games,
        device=str(_required(runtime, "device")),
        require_device=bool(_required(runtime, "require_device")),
        torch_deterministic=bool(_required(runtime, "torch_deterministic")),
        cudnn_benchmark=bool(_required(runtime, "cudnn_benchmark")),
        repeat_action_probability=float(_required(environment, "repeat_action_probability")),
        max_num_frames_per_episode=int(_required(environment, "max_num_frames_per_episode")),
        noop_max=int(_required(environment, "noop_max")),
        frame_skip=int(_required(environment, "frame_skip")),
        screen_size=int(_required(environment, "screen_size")),
        stack=int(_required(environment, "stack")),
        train_clip_reward=bool(_required(environment, "train_clip_reward")),
        eval_clip_reward=bool(_required(environment, "eval_clip_reward")),
        render_mode=str(_required(environment, "render_mode")),
        seed=int(_required(preflight, "seed")),
        training_timesteps=int(_required(preflight, "training_timesteps")),
        n_steps=int(_required(preflight, "n_steps")),
        batch_size=int(_required(preflight, "batch_size")),
        n_epochs=int(_required(preflight, "n_epochs")),
        eval_agent_steps=int(_required(preflight, "eval_agent_steps")),
        eval_seed=int(_required(preflight, "eval_seed")),
        policy=str(_required(preflight, "policy")),
        learning_rate=float(_required(preflight, "learning_rate")),
        gamma=float(_required(preflight, "gamma")),
        gae_lambda=float(_required(preflight, "gae_lambda")),
        clip_range=float(_required(preflight, "clip_range")),
        ent_coef=float(_required(preflight, "ent_coef")),
        vf_coef=float(_required(preflight, "vf_coef")),
        max_grad_norm=float(_required(preflight, "max_grad_norm")),
        run_root=ROOT / str(_required(output, "run_root")),
    )


def _resolve_device(config: PreflightConfig) -> str:
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        if config.require_device:
            raise RuntimeError("CUDA is required by this preflight configuration but is unavailable")
        return "cpu"
    return config.device


def _configure_torch(config: PreflightConfig) -> None:
    torch.use_deterministic_algorithms(config.torch_deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    if config.torch_deterministic:
        torch.backends.cudnn.deterministic = True


def _make_env(config: PreflightConfig, game: PreflightGame, *, clip_reward: bool):
    return make_atari_env(
        game.env_id,
        seed=config.seed,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        stack=config.stack,
        noop_max=config.noop_max,
        clip_reward=clip_reward,
        repeat_action_probability=config.repeat_action_probability,
        max_num_frames_per_episode=config.max_num_frames_per_episode,
        start_protocol=game.start_protocol,
        render_mode=config.render_mode,
    )


def validate_preflight(config: PreflightConfig) -> list[dict[str, Any]]:
    gym.register_envs(ale_py)
    records: list[dict[str, Any]] = []
    for game in config.games:
        env = _make_env(config, game, clip_reward=config.eval_clip_reward)
        try:
            observation, _ = env.reset(seed=config.seed)
            if observation.shape != (config.stack, config.screen_size, config.screen_size):
                raise ValueError(f"unexpected observation shape: {observation.shape}")
            if observation.dtype != np.uint8:
                raise ValueError(f"unexpected observation dtype: {observation.dtype}")
            records.append(
                {
                    "slug": game.slug,
                    "env_id": game.env_id,
                    "action_count": int(env.action_space.n),
                    "observation_shape": list(observation.shape),
                    "max_num_frames_per_episode": config.max_num_frames_per_episode,
                    "start_protocol": game.start_protocol,
                }
            )
        finally:
            env.close()
    return records


def _make_train_vec_env(config: PreflightConfig, game: PreflightGame):
    def factory():
        return Monitor(_make_env(config, game, clip_reward=config.train_clip_reward))

    return DummyVecEnv([factory])


def run_preflight_game(config: PreflightConfig, game: PreflightGame, device: str) -> dict[str, Any]:
    game_root = config.run_root / "games" / game.slug
    checkpoint_path = game_root / "checkpoints" / "ppo_preflight.zip"
    game_root.mkdir(parents=True, exist_ok=False)
    train_env = _make_train_vec_env(config, game)
    eval_env = _make_env(config, game, clip_reward=config.eval_clip_reward)
    try:
        model = PPO(
            config.policy,
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            device=device,
            verbose=0,
        )
        model.learn(total_timesteps=config.training_timesteps, progress_bar=False)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_path)
        loaded_model = PPO.load(checkpoint_path, device=device)

        observation, _ = eval_env.reset(seed=config.eval_seed)
        first_frame = eval_env.render()
        score = 0.0
        raw_reward_field_seen = False
        done = False
        steps = 0
        while not done and steps < config.eval_agent_steps:
            action, _ = loaded_model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = eval_env.step(int(action))
            score += float(reward)
            raw_reward_field_seen = raw_reward_field_seen or "raw_reward" in info
            done = bool(terminated or truncated)
            steps += 1
        final_frame = eval_env.render()

        record = {
            "slug": game.slug,
            "env_id": game.env_id,
            "status": "COMPLETED",
            "action_count": int(train_env.action_space.n),
            "training_timesteps": int(model.num_timesteps),
            "checkpoint": str(checkpoint_path),
            "raw_return_limited": score,
            "eval_agent_steps": steps,
            "episode_finished_within_limit": done,
            "raw_reward_field_seen": raw_reward_field_seen,
            "initial_rgb_shape": list(np.asarray(first_frame).shape),
            "final_rgb_shape": list(np.asarray(final_frame).shape),
            "start_protocol": game.start_protocol,
            "max_num_frames_per_episode": config.max_num_frames_per_episode,
        }
        (game_root / "preflight.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record
    finally:
        eval_env.close()
        train_env.close()


def run_preflight(config: PreflightConfig) -> list[dict[str, Any]]:
    if config.run_root.exists():
        raise FileExistsError(f"preflight run root already exists: {config.run_root}")
    device = _resolve_device(config)
    _configure_torch(config)
    validate_preflight(config)
    config.run_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for game in config.games:
            records.append(run_preflight_game(config, game, device))
        (config.run_root / "preflight_summary.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        (config.run_root / "FAILED.txt").write_text(
            "Preflight stopped before all games completed. Inspect per-game artifacts.\n",
            encoding="utf-8",
        )
        raise
    return records


def _is_preflight_config(path: Path) -> bool:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return isinstance(raw, dict) and "preflight" in raw


def _validate_campaign(path: Path, game_slug: str | None = None) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"invalid configuration: {path}")
    if is_phase_manifest_with_campaign(path):
        if game_slug is not None:
            raise ConfigError("Phase-1 campaign always validates its complete manifest order")
        campaign = resolve_phase1_campaign(path)
        return [
            {
                "game": resolved["game"]["slug"],
                "env_id": resolved["game"]["env_id"],
                "effective_timesteps": resolved["budget"]["effective"],
                "config_hash": config_hash,
                "game_config_hash": resolved["game_config_sha256"],
            }
            for resolved, config_hash in campaign.games
        ]
    if "game" in raw:
        resolved, config_hash = resolve_config(path)
        if game_slug is not None and resolved["game"]["slug"] != game_slug:
            raise ConfigError(f"game {game_slug!r} does not match {path}")
        return [
            {
                "game": resolved["game"]["slug"],
                "env_id": resolved["game"]["env_id"],
                "effective_timesteps": resolved["budget"]["effective"],
                "config_hash": config_hash,
            }
        ]
    return [
        {
            "game": resolved["game"]["slug"],
            "env_id": resolved["game"]["env_id"],
            "effective_timesteps": resolved["budget"]["effective"],
            "config_hash": config_hash,
        }
        for resolved, config_hash in resolve_campaign_configs(path, game_slug=game_slug)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "algorithms" / "ppo_sb3" / "configs" / "atari16_campaign.yaml",
    )
    validate_parser.add_argument("--game", dest="game_slug")

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "algorithms" / "ppo_sb3" / "configs" / "atari16_campaign.yaml",
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--game", dest="game_slug")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--retry-failed", action="store_true")

    import_parser = subparsers.add_parser("import-completed")
    import_parser.add_argument("--config", type=Path, required=True)
    import_parser.add_argument("--source-run-dir", type=Path, required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-dir", type=Path, required=True)

    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-dir", type=Path, required=True)

    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--run-dir", type=Path, required=True)
    archive_parser.add_argument("--archive-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "preflight":
        config = load_preflight_config(args.config)
        records = run_preflight(config)
        print(f"{len(records)}/{len(config.games)} preflight games completed")
        print(json.dumps(records, indent=2, sort_keys=True))
        return
    if args.command == "status":
        records = status_records(args.run_dir)
        print(json.dumps(records, indent=2, sort_keys=True))
        return
    if args.command == "summarize":
        state_path = args.run_dir / "campaign" / "campaign_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        report = build_summary(args.run_dir, json.loads(state_path.read_text(encoding="utf-8")))
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "archive":
        destination = archive_campaign(args.run_dir, args.archive_root)
        print(str(destination))
        return
    if args.command == "import-completed":
        campaign = resolve_phase1_campaign(args.config)
        imported = import_completed_games(campaign, args.source_run_dir)
        print(json.dumps({"imported": imported}, indent=2, sort_keys=True))
        return
    if args.command == "validate":
        if _is_preflight_config(args.config):
            config = load_preflight_config(args.config)
            records = validate_preflight(config)
            print(f"{len(records)}/{len(config.games)} environments valid")
        else:
            records = _validate_campaign(args.config, args.game_slug)
            print(f"{len(records)} campaign configurations valid")
        print(json.dumps(records, indent=2, sort_keys=True))
        return
    if _is_preflight_config(args.config):
        raise ConfigError("use the preflight command for a preflight YAML")
    records = run_campaign(
        args.config,
        game_slug=args.game_slug,
        resume=bool(args.resume),
        retry_failed=bool(args.retry_failed),
    )
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
