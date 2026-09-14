"""Deterministic RGB Atari episode recording for trained policy checkpoints.

Expects an SB3-style policy: ``model.predict(observation, deterministic=...)``
returning ``(action, state)``.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from baselines_common.utils import atomic_write_json, scalar_action, sha256_file


def _rgb_frame(env: Any) -> np.ndarray:
    frame = np.asarray(env.render())
    if frame.ndim != 3 or frame.shape[-1] not in {3, 4}:
        raise ValueError(f"expected RGB(A) render frame, received shape {frame.shape}")
    if frame.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB render frame, received {frame.dtype}")
    return frame[..., :3]


def _codec_name(codec: str) -> str:
    if codec == "h264":
        return "libx264"
    return codec


def record_episode_video(
    model: Any,
    env: Any,
    *,
    checkpoint: Path,
    seed: int,
    output_path: Path,
    fps: int,
    config_hash: str,
    deterministic: bool = True,
    codec: str = "h264",
    pixel_format: str = "yuv420p",
) -> dict[str, Any]:
    """Record one fixed-seed raw-RGB episode and return its audit metadata."""
    if fps <= 0:
        raise ValueError("video fps must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.NamedTemporaryFile(
            suffix=".mp4", dir=output_path.parent, delete=False
        ).name
    )
    try:
        observation, _ = env.reset(seed=int(seed))
        initial_frame = _rgb_frame(env)
        current_frame = initial_frame
        frame_count = 0
        raw_return = 0.0
        length = 0
        termination_reason: str | None = None
        with imageio.get_writer(
            str(temporary_path),
            format="FFMPEG",
            mode="I",
            fps=int(fps),
            codec=_codec_name(codec),
            pixelformat=pixel_format,
            macro_block_size=1,
            ffmpeg_log_level="error",
        ) as writer:
            writer.append_data(initial_frame)
            frame_count += 1
            while True:
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = env.step(scalar_action(action))
                raw_reward = float(dict(info).get("raw_reward", reward))
                if not math.isfinite(raw_reward):
                    raise ValueError(f"non-finite raw reward while recording seed {seed}")
                raw_return += raw_reward
                length += 1
                current_frame = _rgb_frame(env)
                writer.append_data(current_frame)
                frame_count += 1
                if terminated or truncated:
                    termination_reason = "terminated" if terminated else "truncated"
                    break
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError(f"video encoder did not produce a complete file: {temporary_path}")
        os.replace(temporary_path, output_path)
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "config_hash": config_hash,
            "deterministic": bool(deterministic),
            "final_rgb_shape": list(current_frame.shape),
            "fps": int(fps),
            "frame_count": frame_count,
            "initial_rgb_shape": list(initial_frame.shape),
            "length": length,
            "pixel_format": pixel_format,
            "raw_return": raw_return,
            "seed": int(seed),
            "termination_reason": termination_reason,
            "video": str(output_path),
            "video_sha256": sha256_file(output_path),
        }
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def record_videos(
    model: Any,
    env: Any,
    *,
    checkpoint: Path,
    seeds: list[int],
    output_dir: Path,
    fps: int,
    config_hash: str,
    deterministic: bool = True,
    codec: str = "h264",
    pixel_format: str = "yuv420p",
) -> dict[str, Any]:
    """Record all configured fixed seeds and write one immutable episode index."""
    records = [
        record_episode_video(
            model,
            env,
            checkpoint=checkpoint,
            seed=int(seed),
            output_path=output_dir / f"episode_seed_{int(seed)}.mp4",
            fps=fps,
            config_hash=config_hash,
            deterministic=deterministic,
            codec=codec,
            pixel_format=pixel_format,
        )
        for seed in seeds
    ]
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_hash": config_hash,
        "episodes": records,
    }
    atomic_write_json(output_dir / "episodes.json", result)
    return result
