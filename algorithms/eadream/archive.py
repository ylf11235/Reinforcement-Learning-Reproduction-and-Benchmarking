"""Raw-score summaries and immutable, hash-attested run archives."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _bootstrap(values: list[float], *, seed: int = 0, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    import random
    rng = random.Random(seed)
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)]
    means.sort()
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "stderr": None, "bootstrap_ci95": None}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": mean, "median": statistics.median(values), "std": std, "stderr": std / math.sqrt(len(values)), "bootstrap_ci95": _bootstrap(values)}


def build_summary(campaign_root: Path, campaign_state: dict[str, Any]) -> dict[str, Any]:
    """Produce JSON, CSV and Markdown summaries without mixing protocols."""
    rows: list[dict[str, Any]] = []
    for key in campaign_state.get("selected_runs", []):
        record = campaign_state.get("runs", {}).get(key, {})
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        formal = result.get("formal_returns", result.get("formal_raw_returns", []))
        audit = result.get("audit_returns", result.get("audit_raw_returns", []))
        if isinstance(formal, (int, float)): formal = [formal]
        if isinstance(audit, (int, float)): audit = [audit]
        rows.append({"run": key, "slug": key.split("/", 1)[0], "seed": record.get("seed"), "status": record.get("status"), "formal": _stats([float(v) for v in formal if isinstance(v, (int, float))]), "audit": _stats([float(v) for v in audit if isinstance(v, (int, float))]), "throughput": result.get("throughput", result.get("train_fps")), "peak_vram_bytes": result.get("peak_vram_bytes"), "replay_bytes": result.get("replay_bytes"), "checkpoint_bytes": result.get("checkpoint_bytes"), "provenance_sha256": record.get("provenance_sha256")})
    summary = {"schema_version": 1, "campaign_id": campaign_state.get("campaign_id"), "campaign_status": campaign_state.get("status"), "rows": rows, "failures": [row for row in rows if row["status"] == "FAILED"]}
    out = Path(campaign_root) / "summary"
    _atomic_json(out / "scores.json", summary)
    fields = ["run", "slug", "seed", "status", "formal_mean", "formal_median", "formal_std", "audit_mean", "audit_median", "audit_std", "throughput", "peak_vram_bytes", "replay_bytes", "checkpoint_bytes", "provenance_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({"run": row["run"], "slug": row["slug"], "seed": row["seed"], "status": row["status"], "formal_mean": row["formal"]["mean"], "formal_median": row["formal"]["median"], "formal_std": row["formal"]["std"], "audit_mean": row["audit"]["mean"], "audit_median": row["audit"]["median"], "audit_std": row["audit"]["std"], **{k: row[k] for k in fields[10:]}})
    (out / "scores.md").write_text("# EADream scores\n\n| Run | Status | Formal mean | Audit mean |\n|---|---|---:|---:|\n" + "\n".join(f"| {r['run']} | {r['status']} | {r['formal']['mean']} | {r['audit']['mean']} |" for r in rows) + "\n", encoding="utf-8")
    return summary


def _manifest(root: Path) -> dict[str, str]:
    excluded = {"ARCHIVE_COMPLETE.json", "MANIFEST.json", "SHA256SUMS.txt"}
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name not in excluded}


def _verify_replay_payloads(source: Path) -> None:
    """Validate EpisodeStore completion markers before excluding payloads."""
    for marker in source.rglob("*.complete.json"):
        record = _json(marker)
        filename = record.get("filename")
        digest = record.get("sha256")
        if not isinstance(filename, str) or not isinstance(digest, str):
            raise ValueError(f"invalid replay marker: {marker}")
        payload = marker.parent / filename
        if not payload.is_file() or _sha256(payload) != digest:
            raise ValueError(f"replay payload hash mismatch: {payload}")


def archive_run(run_dir: Path, archive_root: Path, *, scope: str = "result") -> Path:
    """Atomically publish a run archive; destination is never overwritten."""
    if scope not in {"result", "continuation"}:
        raise ValueError("archive scope must be result or continuation")
    source = Path(run_dir).resolve()
    if not (source / "run_state.json").is_file():
        raise ValueError(f"missing run state: {source}")
    _verify_replay_payloads(source)
    destination = Path(archive_root).resolve() / source.name
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}.", dir=destination.parent))
    try:
        staged = temporary / source.name
        shutil.copytree(source, staged)
        if scope == "result":
            replay = staged / "replay"
            if replay.is_dir():
                for payload in replay.glob("*.npz"):
                    payload.unlink()
        manifest = _manifest(staged)
        _atomic_json(staged / "MANIFEST.json", {"files": manifest})
        (staged / "SHA256SUMS.txt").write_text("".join(f"{digest}  {path}\n" for path, digest in sorted(manifest.items())), encoding="utf-8")
        for relative, digest in manifest.items():
            if _sha256(staged / relative) != digest:
                raise ValueError(f"archive copy hash mismatch: {relative}")
        _atomic_json(staged / "ARCHIVE_COMPLETE.json", {"schema_version": 1, "scope": scope, "completed_at_unix": time.time(), "file_count": len(manifest), "manifest_sha256": _sha256(staged / "MANIFEST.json")})
        os.replace(staged, destination)
        return destination
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


__all__ = ["build_summary", "archive_run"]
