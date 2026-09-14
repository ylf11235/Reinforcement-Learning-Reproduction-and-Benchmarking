"""Native ALE video capture for EADream."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from .evaluation import _action_index, _make_env, _predict
from eadream.engine.checkpoint import capture_rng_state, restore_rng_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_video(policy: Any, env_factory: Any, *, seed: int, output_path: Path,
                 checkpoint: Path | None = None, fps: int = 30,
                 config_hash: str | None = None, rom_hash: str | None = None,
                 checkpoint_hash: str | None = None) -> dict[str, Any]:
    """Record one episode from native ``env.render()`` frames.

    The writer is always closed, including on policy/environment failures.  The
    returned metadata is JSON-serialisable and contains a hash of the MP4.
    """
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    env = _make_env(env_factory, mode="video", seed=seed)
    caller_rng = capture_rng_state()
    writer = None
    frames = 0
    native_shape: list[int] | None = None
    raw_return = 0.0
    state = None
    terminated = truncated = False
    try:
        observation, reset_info = env.reset(seed=seed)
        writer = imageio.get_writer(output, fps=int(fps), macro_block_size=None)

        def append_frame() -> None:
            nonlocal frames, native_shape
            frame = np.asarray(env.render())
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("native ALE render must be uint8 HxWx3")
            writer.append_data(np.ascontiguousarray(frame))
            frames += 1
            native_shape = list(frame.shape)

        append_frame()
        interactions = 0
        while not (terminated or truncated):
            action, state = _predict(policy, observation, state, episode_start=interactions == 0)
            observation, reward, terminated, truncated, info = env.step(_action_index(action))
            info = info if isinstance(info, Mapping) else {}
            raw_return += float(info.get("raw_reward", reward))
            interactions += 1
            append_frame()
    finally:
        if writer is not None:
            writer.close()
        if hasattr(env, "close"):
            env.close()
        restore_rng_state(caller_rng)
    metadata = {
        "seed": int(seed), "length": interactions, "frame_count": frames,
        "raw_return": float(raw_return), "return_kind": "raw_unclipped",
        "termination_reason": "truncated" if truncated else "terminated",
        "fps": int(fps), "native_frame_shape": native_shape,
        "checkpoint_sha256": checkpoint_hash or (_sha256(Path(checkpoint)) if checkpoint is not None else None),
        "config_sha256": config_hash, "rom_sha256": rom_hash,
    }
    metadata["video_sha256"] = _sha256(output)
    metadata["metadata_sha256"] = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return metadata


def record_videos(policy: Any, env_factory: Any, *, seeds: Sequence[int], output_dir: Path,
                  checkpoint: Path | None = None, fps: int = 30, config_hash: str | None = None,
                  rom_hash: str | None = None) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    episodes = []
    for seed in seeds:
        episodes.append(record_video(policy, env_factory, seed=int(seed), output_path=directory / f"episode_seed_{int(seed):05d}.mp4",
                                     checkpoint=checkpoint, fps=fps, config_hash=config_hash, rom_hash=rom_hash))
    index = {"episodes": episodes, "count": len(episodes)}
    index["index_sha256"] = hashlib.sha256(json.dumps(index, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (directory / "episodes.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return index


# Naming parallel to PPOSB3's public helper.
record_episode_video = record_video

__all__ = ["record_video", "record_episode_video", "record_videos"]
