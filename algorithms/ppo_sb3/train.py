"""Stable-Baselines3 PPO baseline for ALE/Pong-v5."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ALGORITHMS = os.path.join(ROOT, "algorithms")  # makes `import ppo_sb3` work from the repository root
if ALGORITHMS not in sys.path:
    sys.path.insert(0, ALGORITHMS)

from baselines_common.envs.atari import make_atari_env
from baselines_common.logging import JsonlLogger

from ppo_sb3.callbacks import BaselineCallback


@dataclass
class PPOConfig:
    env_id: str = "ALE/Pong-v5"
    num_envs: int = 8
    total_timesteps: int = 2_000_000
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    learning_rate: float = 2.5e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_skip: int = 4
    screen_size: int = 84
    stack: int = 4
    noop_max: int = 30
    clip_reward: bool = True
    eval_every: int = 100_000
    eval_episodes: int = 5
    checkpoint_every: int = 100_000
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "results/ppo_sb3/pong"
    tensorboard_log: Optional[str] = None

    def dict(self):
        return asdict(self)


EnvironmentPurpose = Literal["train", "evaluation"]


def environment_settings(
    config: dict[str, Any], *, purpose: EnvironmentPurpose
) -> dict[str, Any]:
    """Translate a resolved config into one environment-factory argument mapping."""
    environment = config["environment"]
    game = config["game"]
    if purpose == "train":
        clip_reward = bool(environment["train_clip_reward"])
        render_mode = None
    elif purpose == "evaluation":
        # The formal score contract always uses the unmodified ALE return.
        clip_reward = False
        render_mode = str(environment["render_mode"])
    else:
        raise ValueError(f"unknown environment purpose: {purpose}")
    return {
        "env_id": str(game["env_id"]),
        "frame_skip": int(environment["frame_skip"]),
        "screen_size": int(environment["screen_size"]),
        "stack": int(environment["stack"]),
        "noop_max": int(environment["noop_max"]),
        "clip_reward": clip_reward,
        "repeat_action_probability": float(environment["repeat_action_probability"]),
        "max_num_frames_per_episode": int(environment["max_num_frames_per_episode"]),
        "start_protocol": str(environment["start_protocol"]),
        "render_mode": render_mode,
    }


def configure_runtime(config: dict[str, Any]) -> str:
    """Set deterministic flags and enforce the configured runtime device."""
    runtime = config["runtime"]
    requested_device = str(runtime["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if bool(runtime["require_device"]):
            raise RuntimeError("CUDA is required by this configuration but is unavailable")
        requested_device = "cpu"
    np.random.seed(int(config["game"]["seed"]))
    torch.manual_seed(int(config["game"]["seed"]))
    torch.use_deterministic_algorithms(bool(runtime["torch_deterministic"]), warn_only=True)
    torch.backends.cudnn.benchmark = bool(runtime["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(runtime["torch_deterministic"])
    return requested_device


def make_resolved_train_env(config: dict[str, Any]):
    """Create independent clipped-reward workers with SeedSequence-derived seeds."""
    num_envs = int(config["runtime"]["num_envs"])
    base_seed = int(config["game"]["seed"])
    settings = environment_settings(config, purpose="train")
    seeds = np.random.SeedSequence(base_seed).spawn(num_envs)

    def factory(worker_seed: int):
        def make_worker():
            return Monitor(make_atari_env(seed=worker_seed, **settings))

        return make_worker

    worker_seeds = [int(sequence.generate_state(1, dtype=np.uint32)[0]) for sequence in seeds]
    return DummyVecEnv([factory(worker_seed) for worker_seed in worker_seeds])


def make_resolved_eval_env(config: dict[str, Any]):
    """Create the separate one-environment raw-reward evaluator."""
    settings = environment_settings(config, purpose="evaluation")
    return make_atari_env(seed=int(config["game"]["seed"]) + 31_337, **settings)


def build_resolved_model(config: dict[str, Any], env, *, device: str, tensorboard_log: str | None = None) -> PPO:
    """Build PPO without assuming a fixed action count or a particular Atari game."""
    algorithm = config["algorithm"]
    game = config["game"]
    return PPO(
        str(algorithm["policy"]),
        env,
        learning_rate=float(algorithm["learning_rate"]),
        n_steps=int(algorithm["n_steps"]),
        batch_size=int(algorithm["batch_size"]),
        n_epochs=int(algorithm["n_epochs"]),
        gamma=float(algorithm["gamma"]),
        gae_lambda=float(algorithm["gae_lambda"]),
        clip_range=float(algorithm["clip_range"]),
        clip_range_vf=algorithm.get("clip_range_vf"),
        normalize_advantage=bool(algorithm["normalize_advantage"]),
        ent_coef=float(algorithm["ent_coef"]),
        vf_coef=float(algorithm["vf_coef"]),
        max_grad_norm=float(algorithm["max_grad_norm"]),
        use_sde=bool(algorithm["use_sde"]),
        sde_sample_freq=int(algorithm["sde_sample_freq"]),
        target_kl=algorithm.get("target_kl"),
        stats_window_size=int(algorithm["stats_window_size"]),
        policy_kwargs=dict(algorithm.get("policy_kwargs", {})),
        seed=int(game["seed"]),
        device=device,
        tensorboard_log=tensorboard_log,
        verbose=0,
    )


def load_resolved_model(checkpoint: str, env, *, device: str) -> PPO:
    """Load a compatible PPO checkpoint and attach the new training environment."""
    return PPO.load(checkpoint, env=env, device=device)


def _make_worker(cfg: PPOConfig, index: int):
    def factory():
        env = make_atari_env(
            cfg.env_id,
            cfg.seed + 1000 * index + 7,
            frame_skip=cfg.frame_skip,
            screen_size=cfg.screen_size,
            stack=cfg.stack,
            noop_max=cfg.noop_max,
            clip_reward=cfg.clip_reward,
        )
        return Monitor(env)

    return factory


def make_sb3_env(cfg: PPOConfig):
    env = DummyVecEnv([_make_worker(cfg, i) for i in range(cfg.num_envs)])
    shape = env.observation_space.shape
    if shape[-1] in (1, 3, 4) and shape[0] not in (1, 3, 4):
        env = VecTransposeImage(env)
    return env


def parse_args(argv=None) -> PPOConfig:
    cfg = PPOConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", dest="env_id", default=cfg.env_id)
    parser.add_argument("--num-envs", type=int, default=cfg.num_envs)
    parser.add_argument("--total-timesteps", type=int, default=cfg.total_timesteps)
    parser.add_argument("--n-steps", type=int, default=cfg.n_steps)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--n-epochs", type=int, default=cfg.n_epochs)
    parser.add_argument("--gamma", type=float, default=cfg.gamma)
    parser.add_argument("--gae-lambda", type=float, default=cfg.gae_lambda)
    parser.add_argument("--clip-range", type=float, default=cfg.clip_range)
    parser.add_argument("--clip-range-vf", type=float, default=cfg.clip_range_vf)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--ent-coef", type=float, default=cfg.ent_coef)
    parser.add_argument("--vf-coef", type=float, default=cfg.vf_coef)
    parser.add_argument("--max-grad-norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--frame-skip", type=int, default=cfg.frame_skip)
    parser.add_argument("--screen-size", type=int, default=cfg.screen_size)
    parser.add_argument("--stack", type=int, default=cfg.stack)
    parser.add_argument("--noop-max", type=int, default=cfg.noop_max)
    parser.add_argument("--clip-reward", type=int, default=int(cfg.clip_reward))
    parser.add_argument("--eval-every", type=int, default=cfg.eval_every)
    parser.add_argument("--eval-episodes", type=int, default=cfg.eval_episodes)
    parser.add_argument("--checkpoint-every", type=int, default=cfg.checkpoint_every)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", default=cfg.device)
    parser.add_argument("--out-dir", default=cfg.out_dir)
    parser.add_argument("--tensorboard-log", default=cfg.tensorboard_log)
    args = parser.parse_args(argv)
    values = vars(args)
    values["clip_reward"] = bool(values["clip_reward"])
    return PPOConfig(**values)


def main(argv=None):
    cfg = parse_args(argv)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    tensorboard_log = cfg.tensorboard_log or os.path.join(cfg.out_dir, "tensorboard")
    env = make_sb3_env(cfg)
    eval_env = make_atari_env(
        cfg.env_id,
        cfg.seed + 31337,
        frame_skip=cfg.frame_skip,
        screen_size=cfg.screen_size,
        stack=cfg.stack,
        noop_max=cfg.noop_max,
        clip_reward=cfg.clip_reward,
    )
    device = cfg.device if cfg.device != "cuda" or __import__("torch").cuda.is_available() else "cpu"
    config_record = cfg.dict()
    config_record.update({
        "n_actions": int(env.action_space.n),
        "observation_shape": list(env.observation_space.shape),
        "resolved_device": device,
        "tensorboard_log_resolved": tensorboard_log,
    })
    with JsonlLogger(cfg.out_dir, config_record) as logger:
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            clip_range_vf=cfg.clip_range_vf,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            seed=cfg.seed,
            device=device,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
        callback = BaselineCallback(cfg, eval_env, logger)
        print(f"[PPOSB3] env={cfg.env_id} actions={env.action_space.n} device={device} workers={cfg.num_envs}", flush=True)
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=False)
        final_eval = callback.evaluate()
        final_eval["final"] = True
        logger.evaluate(final_eval)
        print(f"[PPOSB3] FINAL EVAL trans={model.num_timesteps} mean={final_eval['eval_mean']:.2f}", flush=True)
        model.save(os.path.join(cfg.out_dir, "checkpoints", "ppo_final.zip"))
    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
