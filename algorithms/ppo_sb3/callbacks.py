"""Evaluation, checkpoint and JSONL callback for the SB3 PPO baseline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from baselines_common.envs.atari import make_atari_env
from baselines_common.logging import JsonlLogger
from baselines_common.utils import atomic_write_json

from ppo_sb3.evaluation import (
    SavedCheckpoint,
    evaluate_raw,
    save_model_atomic,
    update_best_checkpoint,
)


class BaselineCallback(BaseCallback):
    def __init__(self, cfg, eval_env, logger: JsonlLogger, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.eval_env = eval_env
        self.jsonl = logger
        self.next_eval = max(1, int(cfg.eval_every))
        self.next_checkpoint = max(1, int(cfg.checkpoint_every))
        self.completed_returns = []
        self.completed_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is not None:
                self.completed_returns.append(float(episode["r"]))
                self.completed_lengths.append(int(episode["l"]))
        while self.num_timesteps >= self.next_eval:
            result = self.evaluate()
            self.jsonl.evaluate(result)
            self.model.save(os.path.join(self.cfg.out_dir, "checkpoints", f"ppo_{self.num_timesteps}.zip"))
            self.next_eval += max(1, int(self.cfg.eval_every))
        while self.num_timesteps >= self.next_checkpoint:
            self.model.save(os.path.join(self.cfg.out_dir, "checkpoints", "ppo_last.zip"))
            self.next_checkpoint += max(1, int(self.cfg.checkpoint_every))
        return True

    def _on_rollout_end(self) -> None:
        values = dict(getattr(self.model.logger, "name_to_value", {}))
        record = {
            "transitions": int(self.num_timesteps),
            "rollout_episode_return_mean": float(np.mean(self.completed_returns)) if self.completed_returns else float("nan"),
            "rollout_episode_length_mean": float(np.mean(self.completed_lengths)) if self.completed_lengths else float("nan"),
            "rollout_episode_count": len(self.completed_returns),
        }
        aliases = {
            "train/policy_gradient_loss": "policy_gradient_loss",
            "train/value_loss": "value_loss",
            "train/entropy_loss": "entropy_loss",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/loss": "loss",
            "train/explained_variance": "explained_variance",
            "rollout/ep_rew_mean": "sb3_rollout_ep_rew_mean",
            "rollout/ep_len_mean": "sb3_rollout_ep_len_mean",
        }
        for source, target in aliases.items():
            if source in values:
                value = values[source]
                record[target] = float(value)
        self.jsonl.train(record)
        self.completed_returns.clear()
        self.completed_lengths.clear()

    def evaluate(self) -> dict:
        scores = []
        for episode in range(int(self.cfg.eval_episodes)):
            obs, _ = self.eval_env.reset(seed=int(self.cfg.seed) + 31337 + episode)
            done = False
            score = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(int(action))
                score += float(reward)
                done = bool(terminated or truncated)
            scores.append(score)
        return {
            "transitions": int(self.num_timesteps),
            "eval_mean": float(np.mean(scores)),
            "eval_min": float(np.min(scores)),
            "eval_max": float(np.max(scores)),
            "eval_episodes": len(scores),
        }


class CampaignCallback(BaseCallback):
    """Checkpoint complete rollouts and evaluate them with raw ALE rewards."""

    def __init__(
        self,
        config: dict[str, Any],
        eval_env,
        logger: JsonlLogger,
        run_dir: Path,
        config_hash: str,
        *,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.config = config
        self.eval_env = eval_env
        self.jsonl = logger
        self.run_dir = run_dir
        self.config_hash = config_hash
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        self.last_checkpoint_destination = run_dir / "checkpoints" / "last" / "model.zip"
        self.last_checkpoint = self.last_checkpoint_destination
        evaluation = config["evaluation"]
        checkpoint = config["checkpoint"]
        self.eval_every = int(evaluation["periodic_every_timesteps"])
        self.checkpoint_every = int(checkpoint["save_every_timesteps"])
        self.next_eval = self.eval_every
        self.next_checkpoint = self.checkpoint_every

    def _on_training_start(self) -> None:
        self.next_eval = self._next_after(int(self.model.num_timesteps), self.eval_every)
        self.next_checkpoint = self._next_after(
            int(self.model.num_timesteps), self.checkpoint_every
        )

    @staticmethod
    def _next_after(current: int, interval: int) -> int:
        if interval <= 0:
            raise ValueError("checkpoint and evaluation intervals must be positive")
        return (current // interval + 1) * interval

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode is not None:
                self.completed_returns.append(float(episode["r"]))
                self.completed_lengths.append(int(episode["l"]))
        return True

    def _save_last(self, model: Any | None = None) -> SavedCheckpoint:
        checkpoint_model = self.model if model is None else model
        saved = save_model_atomic(checkpoint_model, self.last_checkpoint_destination)
        self.last_checkpoint = saved.path
        atomic_write_json(
            self.last_checkpoint.parent / "metadata.json",
            {
                "checkpoint_path": str(saved.path),
                "config_hash": self.config_hash,
                "model_sha256": saved.sha256,
                "transitions": int(checkpoint_model.num_timesteps),
            },
        )
        return saved

    def _log_rollout(self) -> None:
        values = dict(getattr(self.model.logger, "name_to_value", {}))
        record = {
            "transitions": int(self.model.num_timesteps),
            "rollout_episode_return_mean": float(np.mean(self.completed_returns))
            if self.completed_returns
            else None,
            "rollout_episode_length_mean": float(np.mean(self.completed_lengths))
            if self.completed_lengths
            else None,
            "rollout_episode_count": len(self.completed_returns),
        }
        aliases = {
            "train/policy_gradient_loss": "policy_gradient_loss",
            "train/value_loss": "value_loss",
            "train/entropy_loss": "entropy_loss",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/loss": "loss",
            "train/explained_variance": "explained_variance",
            "rollout/ep_rew_mean": "sb3_rollout_ep_rew_mean",
            "rollout/ep_len_mean": "sb3_rollout_ep_len_mean",
        }
        for source, target in aliases.items():
            if source in values:
                record[target] = float(values[source])
        self.jsonl.train(record)
        self.completed_returns.clear()
        self.completed_lengths.clear()

    def _periodic_evaluation(self) -> dict[str, Any]:
        result = evaluate_raw(
            self.model,
            self.eval_env,
            self.config["evaluation"]["periodic_seeds"],
            deterministic=bool(self.config["evaluation"]["deterministic"]),
        )
        result["transitions"] = int(self.model.num_timesteps)
        result["kind"] = "periodic_raw"
        self.jsonl.evaluate(result)
        evaluation_path = self.run_dir / "evaluation" / f"periodic_{self.model.num_timesteps}.json"
        atomic_write_json(evaluation_path, result)
        update_best_checkpoint(
            self.last_checkpoint,
            result,
            self.run_dir / "best_checkpoint" / "metadata.json",
            config_hash=self.config_hash,
        )
        return result

    def _on_rollout_end(self) -> None:
        self._log_rollout()
        self._save_last()
        while self.num_timesteps >= self.next_checkpoint:
            periodic_path = (
                self.run_dir / "checkpoints" / "periodic" / f"step_{self.next_checkpoint:08d}.zip"
            )
            save_model_atomic(self.model, periodic_path)
            self.next_checkpoint += self.checkpoint_every
        while self.num_timesteps >= self.next_eval:
            self._periodic_evaluation()
            self.next_eval += self.eval_every
