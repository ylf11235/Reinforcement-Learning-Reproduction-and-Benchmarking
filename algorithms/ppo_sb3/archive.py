"""Campaign score summaries and complete-only archive creation."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from baselines_common.utils import atomic_write_json

from ppo_sb3.evaluation import checkpoint_path_from_metadata
from ppo_sb3.provenance import sha256_path


SUMMARY_COLUMNS = [
    "slug",
    "env_id",
    "status",
    "requested_timesteps",
    "effective_timesteps",
    "final_raw_return_mean",
    "final_raw_return_median",
    "final_raw_return_std",
    "best_periodic_raw_mean",
    "action_count",
    "training_wall_time_s",
    "train_fps",
    "final_checkpoint_sha256",
    "best_checkpoint_sha256",
    "video_sha256",
    "failure_kind",
    "error_type",
    "error_message",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.is_file() else {}


def _optional_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _video_hashes(run_dir: Path) -> dict[str, str]:
    index = _optional_json(run_dir / "videos" / "best" / "episodes.json")
    hashes: dict[str, str] = {}
    for episode in index.get("episodes", []):
        if isinstance(episode, dict) and isinstance(episode.get("video"), str):
            hashes[Path(episode["video"]).name] = episode.get("video_sha256")
    return hashes


def _completed_summary_row(slug: str, record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    run_dir = Path(str(record.get("run_dir", "")))
    resolved = _optional_config(run_dir / "resolved_config.yaml")
    best_metadata = _optional_json(run_dir / "best_checkpoint" / "metadata.json")
    best_evaluation = best_metadata.get("evaluation", {})
    if not isinstance(best_evaluation, dict):
        best_evaluation = {}
    best_hash = best_metadata.get("model_sha256")
    row = {
        "slug": slug,
        "env_id": record.get("env_id"),
        "status": "COMPLETED",
        "requested_timesteps": resolved.get("budget", {}).get("requested"),
        "effective_timesteps": result.get("effective_timesteps"),
        "final_raw_return_mean": result.get("final_raw_return_mean"),
        "final_raw_return_median": result.get("final_raw_return_median"),
        "final_raw_return_std": result.get("final_raw_return_std"),
        "best_periodic_raw_mean": best_evaluation.get(
            "raw_return_mean", best_evaluation.get("mean")
        ),
        "action_count": result.get("action_count"),
        "training_wall_time_s": result.get("training_wall_time_s"),
        "train_fps": result.get("train_fps"),
        "final_checkpoint_sha256": result.get("final_checkpoint_sha256"),
        "best_checkpoint_sha256": best_hash,
        "video_sha256": _video_hashes(run_dir),
        "failure_kind": None,
        "error_type": None,
        "error_message": None,
    }
    if row["train_fps"] is None and row["training_wall_time_s"]:
        row["train_fps"] = row["effective_timesteps"] / row["training_wall_time_s"]
    return row


def _failed_summary_row(slug: str, record: dict[str, Any]) -> dict[str, Any]:
    row = {column: None for column in SUMMARY_COLUMNS}
    row.update(
        {
            "slug": slug,
            "env_id": record.get("env_id"),
            "status": "FAILED",
            "effective_timesteps": record.get("last_known_timestep"),
            "failure_kind": record.get("failure_kind"),
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }
    )
    return row


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    return output.getvalue()


def build_summary(campaign_root: Path, campaign_state: dict[str, Any]) -> dict[str, Any]:
    """Write ordered score, failure, and throughput summaries for any terminal campaign."""
    rows: list[dict[str, Any]] = []
    records = campaign_state.get("games", {})
    for slug in campaign_state.get("selected_slugs", []):
        record = records.get(slug, {})
        if record.get("status") == "COMPLETED":
            rows.append(_completed_summary_row(slug, record))
        else:
            rows.append(_failed_summary_row(slug, record))

    summary_dir = campaign_root / "summary"
    failures = [row for row in rows if row["status"] == "FAILED"]
    throughput = [row for row in rows if row["status"] == "COMPLETED"]
    report = {
        "schema_version": 1,
        "campaign_id": campaign_state.get("campaign_id"),
        "campaign_status": campaign_state.get("status"),
        "selected_game_count": len(rows),
        "completed_game_count": len(rows) - len(failures),
        "failed_game_count": len(failures),
        "rows": rows,
    }
    atomic_write_json(summary_dir / "scores.json", {"rows": rows})
    atomic_write_json(summary_dir / "campaign_report.json", report)
    _atomic_write_text(summary_dir / "scores.csv", _csv_text(rows, SUMMARY_COLUMNS))
    _atomic_write_text(
        summary_dir / "failures.csv",
        _csv_text(failures, ["slug", "env_id", "failure_kind", "error_type", "error_message"]),
    )
    _atomic_write_text(
        summary_dir / "throughput.csv",
        _csv_text(
            throughput,
            ["slug", "effective_timesteps", "training_wall_time_s", "train_fps", "status"],
        ),
    )
    markdown = ["# Phase-1 Scores", "", "| Game | Status | Final raw mean |", "| --- | --- | --- |"]
    markdown.extend(
        f"| {row['slug']} | {row['status']} | {row['final_raw_return_mean']} |" for row in rows
    )
    _atomic_write_text(summary_dir / "scores.md", "\n".join(markdown) + "\n")
    atomic_write_json(campaign_root / "campaign" / "failures.json", {"rows": failures})
    return report


def _finite_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_hash(path: Path, expected: Any, description: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    actual = sha256_path(path)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(f"{description} hash mismatch: {path}")
    return actual


def _validate_game_archive_artifacts(slug: str, record: dict[str, Any]) -> None:
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"missing completed result for {slug}")
    for score in ("final_raw_return_mean", "final_raw_return_median", "final_raw_return_std"):
        if not _finite_score(result.get(score)):
            raise ValueError(f"non-finite {score} for {slug}")

    run_dir = Path(str(record.get("run_dir", "")))
    final_dir = run_dir / "final_checkpoint"
    final_model = checkpoint_path_from_metadata(final_dir)
    final_metadata = _optional_json(final_dir / "metadata.json")
    final_hash = _require_hash(final_model, final_metadata.get("model_sha256"), "final checkpoint")
    if result.get("final_checkpoint_sha256") != final_hash:
        raise ValueError(f"final checkpoint hash mismatch with result for {slug}")

    best_dir = run_dir / "best_checkpoint"
    best_model = checkpoint_path_from_metadata(best_dir)
    best_metadata = _optional_json(best_dir / "metadata.json")
    best_hash = _require_hash(best_model, best_metadata.get("model_sha256"), "best checkpoint")
    final_evaluation = _optional_json(run_dir / "evaluation" / "final_raw_audit.json")
    for score in ("mean", "median", "std"):
        if not _finite_score(final_evaluation.get(score)):
            raise ValueError(f"missing finite final raw evaluation {score} for {slug}")

    resolved = _optional_config(run_dir / "resolved_config.yaml")
    video = resolved.get("video", {})
    seeds = video.get("seeds") if isinstance(video, dict) else None
    if not bool(video.get("enabled", False)) or not isinstance(seeds, list) or not seeds:
        raise ValueError(f"missing required video settings for {slug}")
    index_path = run_dir / "videos" / "best" / "episodes.json"
    index = _optional_json(index_path)
    if index.get("checkpoint_sha256") != best_hash:
        raise ValueError(f"best video checkpoint hash mismatch for {slug}")
    episodes = index.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(seeds):
        raise ValueError(f"missing required best videos for {slug}")
    seen_seeds: set[int] = set()
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError(f"invalid video episode record for {slug}")
        if isinstance(episode.get("seed"), int):
            seen_seeds.add(episode["seed"])
        _require_hash(Path(str(episode.get("video", ""))), episode.get("video_sha256"), "video")
    if seen_seeds and seen_seeds != {int(seed) for seed in seeds}:
        raise ValueError(f"best video seeds do not match configuration for {slug}")


def validate_archive_source(campaign_root: Path) -> None:
    """Reject incomplete, failed, or hash-inconsistent campaign artifacts."""
    campaign_dir = campaign_root / "campaign"
    state_path = campaign_dir / "campaign_state.json"
    if not state_path.is_file():
        raise ValueError("missing campaign state")
    state = _read_json(state_path)
    if state.get("status") != "COMPLETED":
        if state.get("status") == "COMPLETED_WITH_FAILURES":
            raise ValueError("archive requires zero failed games")
        raise ValueError(f"archive requires COMPLETED campaign, got {state.get('status')!r}")
    input_manifest = _optional_config(campaign_dir / "input_manifest.yaml")
    suite = input_manifest.get("suite")
    expected_game_count = suite.get("expected_game_count") if isinstance(suite, dict) else None
    if (
        not isinstance(expected_game_count, int)
        or isinstance(expected_game_count, bool)
        or expected_game_count <= 0
    ):
        raise ValueError("campaign manifest requires a positive suite.expected_game_count")
    selected = state.get("selected_slugs")
    if (
        not isinstance(selected, list)
        or len(selected) != expected_game_count
        or len(set(selected)) != expected_game_count
    ):
        raise ValueError(
            f"archive requires exactly {expected_game_count} unique selected games"
        )
    required_campaign_artifacts = [
        campaign_dir / "input_manifest.yaml",
        campaign_dir / "resolved_campaign.yaml",
        campaign_dir / "provenance.json",
        campaign_dir / "source_hashes.json",
    ]
    if any(not path.is_file() for path in required_campaign_artifacts) or not (
        campaign_dir / "source_snapshot"
    ).is_dir():
        raise ValueError("missing required campaign provenance artifacts")
    records = state.get("games")
    if not isinstance(records, dict):
        raise ValueError("campaign game records must be a mapping")
    for slug in selected:
        record = records.get(slug)
        if not isinstance(record, dict) or record.get("status") != "COMPLETED":
            raise ValueError(f"archive requires completed game {slug}")
        _validate_game_archive_artifacts(slug, record)


def _hash_manifest(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_path(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt", "ARCHIVE_COMPLETE.json"}
    }


def archive_campaign(campaign_root: Path, archive_root: Path) -> Path:
    """Copy and attest a zero-failure completed campaign without overwriting archives."""
    validate_archive_source(campaign_root)
    if archive_root.exists():
        raise FileExistsError(archive_root)
    state = _read_json(campaign_root / "campaign" / "campaign_state.json")
    build_summary(campaign_root, state)
    shutil.copytree(campaign_root, archive_root)
    manifest = _hash_manifest(archive_root)
    atomic_write_json(archive_root / "MANIFEST.json", {"files": manifest})
    sums = "".join(f"{digest}  {path}\n" for path, digest in sorted(manifest.items()))
    _atomic_write_text(archive_root / "SHA256SUMS.txt", sums)
    for relative, expected in manifest.items():
        if sha256_path(archive_root / relative) != expected:
            raise ValueError(f"archive copy hash mismatch: {relative}")
    atomic_write_json(
        archive_root / "ARCHIVE_COMPLETE.json",
        {
            "schema_version": 1,
            "completed_at_unix": time.time(),
            "file_count": len(manifest),
            "manifest_sha256": sha256_path(archive_root / "MANIFEST.json"),
            "sha256sums_sha256": sha256_path(archive_root / "SHA256SUMS.txt"),
        },
    )
    return archive_root
