"""Train/eval env factories over the shared baselines_common contracts:
Atari, DMC (dm_control, state input), and CarRacing (pixel, continuous)."""
from __future__ import annotations

import sys
from pathlib import Path

# Repository root on sys.path so baselines_common resolves (repo
# convention); experiment.py also inserts this path for CLI runs.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baselines_common.envs.atari import make_atari_env  # noqa: E402
from baselines_common.envs.car_racing import (  # noqa: E402
    CarRacingAdapter, make_car_racing_env)
from baselines_common.envs.dmc import DMCGymAdapter  # noqa: E402


def build_training_env(resolved_cfg: dict):
    """Training env factory: DMC / CarRacing adapters or the shared Atari
    contract."""
    if resolved_cfg["experiment"]["variant"] == "dmc":
        env_id = resolved_cfg["game"]["env_id"]  # Dmc-<domain>-<task>
        _, domain, task = env_id.split("-", 2)
        return DMCGymAdapter(domain, task, resolved_cfg["runtime"]["seed"])
    if resolved_cfg["experiment"]["variant"] == "car_racing":
        env_cfg = resolved_cfg["environment"]
        return CarRacingAdapter(
            make_car_racing_env(),
            frame_stack=env_cfg["stack"],
            action_repeat=env_cfg["frame_skip"],
            max_frames=env_cfg["max_num_frames_per_episode"],
            seed=resolved_cfg["runtime"]["seed"],
        )
    env_cfg = resolved_cfg["environment"]
    return make_atari_env(
        resolved_cfg["game"]["env_id"],
        resolved_cfg["runtime"]["seed"],
        frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"],
        stack=env_cfg["stack"],
        noop_max=env_cfg["noop_max"],
        clip_reward=env_cfg["clip_reward"],
        repeat_action_probability=env_cfg["repeat_action_probability"],
        max_num_frames_per_episode=env_cfg["max_num_frames_per_episode"],
        start_protocol=env_cfg["start_protocol"],
    )


def build_eval_env(resolved_cfg: dict, eval_seed: int,
                   render_mode: str | None = None):
    """Eval env: always raw rewards, deterministic; ``render_mode`` (Atari
    only) enables frame rendering for video recording."""
    if resolved_cfg["experiment"]["variant"] == "dmc":
        env_id = resolved_cfg["game"]["env_id"]
        _, domain, task = env_id.split("-", 2)
        return DMCGymAdapter(domain, task, int(eval_seed))
    if resolved_cfg["experiment"]["variant"] == "car_racing":
        env_cfg = resolved_cfg["environment"]
        return CarRacingAdapter(
            make_car_racing_env(),
            frame_stack=env_cfg["stack"],
            action_repeat=env_cfg["frame_skip"],
            max_frames=env_cfg["max_num_frames_per_episode"],
            seed=int(eval_seed),
        )
    env_cfg = resolved_cfg["environment"]
    return make_atari_env(
        resolved_cfg["game"]["env_id"],
        eval_seed,
        frame_skip=env_cfg["frame_skip"],
        screen_size=env_cfg["screen_size"],
        stack=env_cfg["stack"],
        noop_max=env_cfg["noop_max"],
        clip_reward=False,  # never inherit training reward clipping
        repeat_action_probability=env_cfg["repeat_action_probability"],
        max_num_frames_per_episode=env_cfg["max_num_frames_per_episode"],
        start_protocol=env_cfg["start_protocol"],
        render_mode=render_mode,
    )
