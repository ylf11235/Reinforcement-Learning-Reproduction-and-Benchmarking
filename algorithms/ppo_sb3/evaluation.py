"""Raw-score evaluation and atomic checkpoint selection helpers."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from baselines_common.utils import scalar_action, sha256_file


_CHECKPOINT_REPLACE_ATTEMPTS = 3
_CHECKPOINT_REPLACE_DELAY_S = 0.1


@dataclass(frozen=True)
class SavedCheckpoint:
    path: Path
    sha256: str


def checkpoint_path_from_metadata(checkpoint_dir: Path) -> Path:
    """Resolve the newest checkpoint while remaining compatible with older metadata."""
    canonical = checkpoint_dir / "model.zip"
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        return canonical
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return canonical
    recorded = metadata.get("checkpoint_path")
    if not isinstance(recorded, str) or not recorded:
        return canonical
    return checkpoint_dir / Path(recorded).name


def evaluate_raw(
    model: Any, env: Any, seeds: Iterable[int], deterministic: bool = True
) -> dict[str, Any]:
    """Evaluate fixed seeds using the pre-clipping reward recorded in ``info``."""
    episode_rows: list[dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        episode_return = 0.0
        length = 0
        while True:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(scalar_action(action))
            details = dict(info)
            raw_reward = float(details.get("raw_reward", reward))
            if not math.isfinite(raw_reward):
                raise ValueError(f"non-finite raw reward for evaluation seed {seed}")
            episode_return += raw_reward
            length += 1
            if terminated or truncated:
                episode_rows.append(
                    {
                        "seed": int(seed),
                        "raw_return": episode_return,
                        "length": length,
                        "termination_reason": "terminated" if terminated else "truncated",
                    }
                )
                break

    if not episode_rows:
        raise ValueError("raw evaluation requires at least one seed")
    returns = np.asarray([row["raw_return"] for row in episode_rows], dtype=np.float64)
    return {
        "episodes": episode_rows,
        "seeds": [row["seed"] for row in episode_rows],
        "returns": returns.tolist(),
        "lengths": [row["length"] for row in episode_rows],
        "termination_reasons": [row["termination_reason"] for row in episode_rows],
        "episode_count": len(episode_rows),
        "mean": float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "raw_return_mean": float(np.mean(returns)),
        "raw_return_median": float(np.median(returns)),
        "raw_return_std": float(np.std(returns)),
    }


def _metric(evaluation: dict[str, Any], short_name: str) -> float:
    value = evaluation.get(short_name, evaluation.get(f"raw_return_{short_name}"))
    if value is None:
        raise ValueError(f"evaluation is missing {short_name}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"evaluation {short_name} must be finite")
    return value


def _candidate_key(evaluation: dict[str, Any]) -> tuple[float, float, float, int]:
    transitions = int(evaluation["transitions"])
    if transitions < 0:
        raise ValueError("evaluation transitions must be non-negative")
    # Tuple ordering encodes mean descending, median descending, std ascending, step ascending.
    return (_metric(evaluation, "mean"), _metric(evaluation, "median"), -_metric(evaluation, "std"), -transitions)


def _atomic_copy(source: Path, destination: Path) -> SavedCheckpoint:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, tempfile.NamedTemporaryFile(
        mode="wb", suffix=".zip", dir=destination.parent, delete=False
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
        temporary = Path(output_handle.name)
    try:
        digest = sha256_file(temporary)
        saved_path = _replace_checkpoint(temporary, destination)
        return SavedCheckpoint(path=saved_path, sha256=digest)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def update_best_checkpoint(
    model_path: Path,
    evaluation: dict[str, Any],
    metadata_path: Path,
    *,
    config_hash: str,
) -> bool:
    """Atomically promote a complete checkpoint only when it wins the stable order."""
    candidate_key = _candidate_key(evaluation)
    destination = metadata_path.parent / "model.zip"
    if metadata_path.is_file():
        current = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_evaluation = current.get("evaluation")
        if not isinstance(current_evaluation, dict):
            raise ValueError(f"invalid best-checkpoint metadata: {metadata_path}")
        if candidate_key <= _candidate_key(current_evaluation):
            return False

    saved = _atomic_copy(model_path, destination)
    metadata = {
        "checkpoint_path": str(saved.path),
        "config_hash": config_hash,
        "created_at_unix": time.time(),
        "evaluation": evaluation,
        "model_sha256": saved.sha256,
        "source_checkpoint": str(model_path),
        "transitions": int(evaluation["transitions"]),
    }
    _atomic_json(metadata_path, metadata)
    return True


def _replace_checkpoint(temporary: Path, destination: Path) -> Path:
    for attempt in range(_CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, destination)
            return destination
        except PermissionError:
            if attempt + 1 < _CHECKPOINT_REPLACE_ATTEMPTS:
                time.sleep(_CHECKPOINT_REPLACE_DELAY_S)
    fallback = destination.with_name(
        f"{destination.stem}.{os.getpid()}.{time.time_ns()}{destination.suffix}"
    )
    os.replace(temporary, fallback)
    return fallback


def save_model_atomic(model: Any, destination: Path) -> SavedCheckpoint:
    """Save an SB3 model to a verified temporary zip, then replace the target."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.stem}.{os.getpid()}.tmp.zip"
    try:
        model.save(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"SB3 did not create a complete checkpoint at {temporary}")
        digest = sha256_file(temporary)
        saved_path = _replace_checkpoint(temporary, destination)
        return SavedCheckpoint(path=saved_path, sha256=digest)
    finally:
        if temporary.exists():
            temporary.unlink()
