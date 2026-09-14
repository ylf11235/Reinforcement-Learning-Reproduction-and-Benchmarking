"""Raw-return evaluation for PPO-PyTorch checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from atari_env import build_env


def evaluate_policy(
    agent,
    config: dict[str, Any],
    *,
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    returns: list[float] = []
    lengths: list[int] = []
    max_ep_len = int(config["algorithm"]["max_ep_len"])
    for episode_index in range(int(episodes)):
        env = build_env(config, seed=seed_start + episode_index, training=False)
        try:
            observation, _ = env.reset(seed=seed_start + episode_index)
            episode_return = 0.0
            length = 0
            while length < max_ep_len:
                action = agent.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
        finally:
            env.close()
    values = np.asarray(returns, dtype=np.float64)
    return {
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "raw_return_mean": float(values.mean()) if len(values) else 0.0,
        "raw_return_median": float(np.median(values)) if len(values) else 0.0,
        "raw_return_std": float(values.std()) if len(values) else 0.0,
        "raw_returns": returns,
        "episode_lengths": lengths,
    }
