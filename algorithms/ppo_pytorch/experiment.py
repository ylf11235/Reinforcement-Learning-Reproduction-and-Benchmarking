"""Train and evaluate the isolated PPO-PyTorch Atari-10 experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines_common.utils import atomic_write_json, sha256_file
from PPO import PPO
from atari_env import build_env
from config import ConfigError, resolve_any, resolve_campaign
from evaluation import evaluate_policy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingLogger:
    """Persist lightweight training metrics as JSONL and TensorBoard events."""

    def __init__(self, run_dir: Path, *, flush_every: int = 10) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.train_log = (self.run_dir / "train_log.jsonl").open("a", encoding="utf-8")
        self.flush_every = max(1, int(flush_every))
        self._pending_calls = 0
        self._closed = False

    def _flush_if_needed(self) -> None:
        self._pending_calls += 1
        if self._pending_calls % self.flush_every == 0:
            self.writer.flush()
            self.train_log.flush()

    def log_update(
        self,
        *,
        step: int,
        metrics: dict[str, float],
        episode_count: int,
        elapsed_seconds: float,
    ) -> None:
        elapsed = max(float(elapsed_seconds), 1e-9)
        self.writer.add_scalar("train/loss", float(metrics.get("loss", 0.0)), int(step))
        self.writer.add_scalar(
            "train/normalized_return_mean",
            float(metrics.get("return_mean", 0.0)),
            int(step),
        )
        self.writer.add_scalar("train/episodes", int(episode_count), int(step))
        self.writer.add_scalar("time/fps", float(step) / elapsed, int(step))
        self._flush_if_needed()

    def log_episode(self, record: dict[str, Any]) -> None:
        step = int(record["step"])
        self.writer.add_scalar("rollout/ep_rew", float(record["reward"]), step)
        self.writer.add_scalar("rollout/ep_raw_rew", float(record["raw_reward"]), step)
        self.writer.add_scalar("rollout/ep_len", int(record["length"]), step)
        self.train_log.write(json.dumps(record, sort_keys=True) + "\n")
        self._flush_if_needed()

    def close(self) -> None:
        if self._closed:
            return
        self.writer.flush()
        self.train_log.flush()
        self.writer.close()
        self.train_log.close()
        self._closed = True


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["runtime"].get("device", "cpu"))
    require = bool(config["runtime"].get("require_device", False))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if require:
            raise RuntimeError(f"required device {requested} is unavailable")
        return torch.device("cpu")
    return torch.device(requested)


def _new_agent(config: dict[str, Any], env, device: torch.device) -> PPO:
    algorithm = config["algorithm"]
    return PPO(
        tuple(env.observation_space.shape),
        int(env.action_space.n),
        lr_actor=float(algorithm["lr_actor"]),
        lr_critic=float(algorithm["lr_critic"]),
        gamma=float(algorithm["gamma"]),
        K_epochs=int(algorithm["K_epochs"]),
        eps_clip=float(algorithm["eps_clip"]),
        entropy_coef=float(algorithm["entropy_coef"]),
        value_coef=float(algorithm["value_coef"]),
        device=device,
    )


def run_game(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    run_dir = Path(config["output"]["run_root"])
    checkpoint_dir = run_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last.pth"
    final_checkpoint = checkpoint_dir / "final.pth"
    result_path = run_dir / "result.json"
    config_path = run_dir / "config.json"
    target = min(int(config["budget"]["requested"]), 3200) if smoke else int(config["budget"]["requested"])
    if result_path.exists():
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            previous_config = json.loads(config_path.read_text(encoding="utf-8"))
            valid = (previous["status"] == "COMPLETED"
                     and previous["game"] == config["game"]["slug"]
                     and previous["transitions"] == target
                     and previous_config["resolved_config_sha256"] == config["resolved_config_sha256"]
                     and final_checkpoint.is_file()
                     and previous["checkpoints"]["final_sha256"] == sha256_file(final_checkpoint))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise ConfigError(f"cannot resume: existing result is incomplete or does not match config/checkpoint: {run_dir}")
        return previous
    if last_checkpoint.exists() or final_checkpoint.exists() or config_path.exists():
        raise ConfigError(f"cannot resume incomplete run from model-only checkpoints: {run_dir}; use a new run root")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(config_path, config)
    seed = int(config["game"].get("seed", 0))
    _set_seed(seed)
    device = _resolve_device(config)
    env = build_env(config, seed=seed, training=True)
    agent = _new_agent(config, env, device)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    algorithm = config["algorithm"]
    requested = int(config["budget"]["requested"])
    target = min(requested, 3200) if smoke else requested
    update_timestep = int(algorithm["update_timestep"])
    max_ep_len = int(algorithm["max_ep_len"])
    step = 0
    episode = 0
    episode_return = 0.0
    raw_episode_return = 0.0
    episode_length = 0
    observation, _ = env.reset(seed=seed)
    logger = TrainingLogger(run_dir)
    start = time.time()
    try:
        while step < target:
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            raw_reward = float(info.get("raw_reward", reward))
            episode_return += float(reward)
            raw_episode_return += raw_reward
            episode_length += 1
            step += 1
            boundary = terminated or truncated or episode_length >= max_ep_len
            agent.buffer.rewards.append(float(reward))
            agent.buffer.is_terminals.append(bool(boundary))
            if step % update_timestep == 0:
                metrics = agent.update()
                atomic_write_json(run_dir / "latest_update.json", {"step": step, **metrics})
                logger.log_update(
                    step=step,
                    metrics=metrics,
                    episode_count=episode,
                    elapsed_seconds=time.time() - start,
                )
                agent.save(last_checkpoint)
            if boundary:
                episode += 1
                logger.log_episode(
                    {
                        "episode": episode,
                        "step": step,
                        "reward": episode_return,
                        "raw_reward": raw_episode_return,
                        "length": episode_length,
                    }
                )
                observation, _ = env.reset(seed=seed + episode)
                episode_return = 0.0
                raw_episode_return = 0.0
                episode_length = 0
        if agent.buffer.rewards:
            metrics = agent.update()
            atomic_write_json(run_dir / "latest_update.json", {"step": step, **metrics})
            logger.log_update(
                step=step,
                metrics=metrics,
                episode_count=episode,
                elapsed_seconds=time.time() - start,
            )
        agent.save(final_checkpoint)
        evaluation = evaluate_policy(agent, config, episodes=1 if smoke else 30, seed_start=20_000)
        result = {
            "status": "COMPLETED",
            "game": config["game"]["slug"],
            "env_id": config["game"]["env_id"],
            "transitions": step,
            "episodes": episode,
            "wall_time_seconds": time.time() - start,
            "device": str(device),
            "evaluation": evaluation,
            "checkpoints": {
                "last": str(last_checkpoint),
                "final": str(final_checkpoint),
                "final_sha256": sha256_file(final_checkpoint),
            },
        }
        atomic_write_json(run_dir / "result.json", result)
        return result
    finally:
        logger.close()
        env.close()


def run_campaign(config_path: Path, *, smoke: bool = False) -> list[dict[str, Any]]:
    configs = resolve_campaign(config_path)
    root = Path(configs[0]["campaign"]["run_root"])
    campaign_dir = root / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "campaign_state.json"
    previous_games = json.loads(state_path.read_text(encoding="utf-8")).get("games", {}) if state_path.exists() else {}
    atomic_write_json(state_path, {"status": "RUNNING", "games": previous_games})
    records: list[dict[str, Any]] = []
    for config in configs:
        slug = config["game"]["slug"]
        try:
            result = run_game(config, smoke=smoke)
        except Exception as error:  # campaign policy is record-and-continue
            result = {
                "status": "FAILED",
                "game": slug,
                "env_id": config["game"]["env_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        records.append(result)
        previous_games[slug] = result
        atomic_write_json(state_path, {"status": "RUNNING", "games": previous_games})
    status = "COMPLETED" if all(r["status"] == "COMPLETED" for r in records) else "COMPLETED_WITH_FAILURES"
    state = {"status": status, "games": {r["game"]: r for r in records}}
    atomic_write_json(campaign_dir / "campaign_state.json", state)
    atomic_write_json(campaign_dir / "campaign_result.json", state)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--config", type=Path, required=True)
    campaign.add_argument("--smoke", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "validate":
        configs = resolve_any(args.config)
        print(json.dumps([{"game": c["game"]["slug"], "env_id": c["game"]["env_id"], "config_hash": c["resolved_config_sha256"]} for c in configs], indent=2))
    elif args.command == "run":
        configs = resolve_any(args.config)
        if len(configs) != 1:
            raise ConfigError("run expects a single game YAML")
        print(json.dumps(run_game(configs[0], smoke=args.smoke), indent=2, sort_keys=True))
    elif args.command == "campaign":
        print(json.dumps(run_campaign(args.config, smoke=args.smoke), indent=2, sort_keys=True))
    else:
        state_path = args.run_dir / "campaign" / "campaign_state.json"
        print(state_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
