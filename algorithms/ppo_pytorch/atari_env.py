"""Gymnasium/ALE environment construction for the isolated backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines_common.envs.atari import make_atari_env  # noqa: E402


def build_env(config: dict[str, Any], *, seed: int | None = None, training: bool = True):
    environment = config["environment"]
    game = config["game"]
    seed_value = game.get("seed", 0) if seed is None else seed
    env = make_atari_env(
        game["env_id"],
        seed=int(seed_value) if seed_value is not None else 0,
        frame_skip=int(environment["frame_skip"]),
        screen_size=int(environment["screen_size"]),
        stack=int(environment["stack"]),
        noop_max=int(environment["noop_max"]),
        clip_reward=bool(environment.get("train_clip_reward", True)) if training else False,
        repeat_action_probability=float(environment["repeat_action_probability"]),
        max_num_frames_per_episode=int(environment["max_num_frames_per_episode"]),
        start_protocol=str(environment.get("start_protocol", "none")),
        render_mode="rgb_array",
    )
    return env
