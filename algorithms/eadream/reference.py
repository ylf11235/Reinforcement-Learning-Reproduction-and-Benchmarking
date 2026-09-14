"""Offline reference-parity metadata and Gate 0--2 helpers.

This module deliberately never downloads source files or ROMs.  The released
score table is a diagnostic comparison input; it is not a per-seed equality
target for a modern ALE installation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eadream.config import ConfigError, ResolvedCampaign, resolve_campaign, validate_formal_versions


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = ROOT / "eadream" / "reference_manifest.json"
SCORE_DATA = ROOT / "eadream" / "reference_data" / "eadream_atari100k.json"
REFERENCE_COMMIT = "269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241"
_SCORE_TABLE_SHA256 = "673afbf333195ba50f4440a41dd1a8e13eeda4a87af47a70ef1116c6a99cdf1b"
_PINNED_REFERENCE_FILES: dict[str, tuple[int, str]] = {
    "EADream/configsc.yaml": (5225, "85af3c422b22dac91ad15a8bb1c4ae41d78b71f2318805daf0a0aacb33655fa8"),
    "EADream/dreamer.py": (15533, "1b88e45b6d90be3b9f8bea69e413cdffb3fa1f55f4e38da605721eb602468a07"),
    "EADream/models.py": (30289, "856f1cf9bd3aa26826af7f2e85ca912e47d311f29b03d8a823a7e804067347fc"),
    "EADream/networks.py": (53897, "37d5b9a26a1f5f4e23854c8135f86746de2a1d112078a67c8ea9fa86d133a597"),
    "EADream/tools.py": (38916, "e6f1df2d4cb60944222e8d815203fc72b67196587bf429583f678c16b883bf77"),
    "EADream/envs/atari.py": (5146, "099a9f1e92f25abb0f3e6ea0465d87ff15927aab064f67e5f2aae8ed05cc121d"),
    "EADream/envs/wrappers.py": (26599, "9f0751da39ebc582f4bd3a2a2a6ede9ad0353a886691d84a3620888a223ab5f6"),
    "EADream/eval.py": (14240, "1200118d5fdbb633fbc8ee9973c94e7d6c39ade45a5e88ff320a037ace405f92"),
    "EADream/requirements.txt": (428, "0e96193ee5d7bc644201d4f8d69eb20c8c98a3b5dd9b3eb2986cf0515cf5fbd3"),
    "LICENSE": (35148, "8b1ba204bb69a0ade2bfcf65ef294a920f6bb361b317dba43c7ef29d96332b9b"),
}
PAPER_OVERLAP = frozenset(
    {
        "alien",
        "chopper_command",
        "frostbite",
        "gopher",
        "kung_fu_master",
        "private_eye",
        "seaquest",
    }
)


class ReferenceError(ValueError):
    """Raised when checked-in reference metadata is malformed or drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReferenceError(f"cannot read reference document {path}") from error
    if type(value) is not dict:
        raise ReferenceError(f"reference document {path} must be a mapping")
    return value


def load_released_scores(path: Path | None = None) -> dict[str, list[float]]:
    """Load the immutable seven-game, five-seed released score values."""
    document = _read_json(SCORE_DATA if path is None else Path(path))
    if set(document) != {"schema_version", "source", "scores"} or document.get("schema_version") != 1 or not isinstance(document.get("scores"), dict):
        raise ReferenceError("released score manifest schema is invalid")
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ReferenceError("released score source metadata is missing")
    if set(source) != {"raw_url", "commit", "path", "bytes", "sha256"}:
        raise ReferenceError("released score source metadata schema is invalid")
    expected_url = f"https://raw.githubusercontent.com/MarquisDarwin/EAWM/{REFERENCE_COMMIT}/results/atari/EADream.json"
    if source.get("raw_url") != expected_url or source.get("path") != "results/atari/EADream.json":
        raise ReferenceError("released score source URL is not pinned")
    if source.get("commit") != REFERENCE_COMMIT or source.get("bytes") != 3497:
        raise ReferenceError("released score source identity is not pinned")
    if source.get("sha256") != "8b3987c7c8f2716c392e7268a32c85f01e20c6441c0eac910648a5747aa08c9e":
        raise ReferenceError("released score source hash is not pinned")
    scores: dict[str, list[float]] = {}
    for slug, values in document["scores"].items():
        if type(slug) is not str or not isinstance(values, list) or len(values) != 5:
            raise ReferenceError(f"released scores for {slug!r} must contain five values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ReferenceError(f"released scores for {slug!r} contain a non-finite value")
        scores[slug] = [float(value) for value in values]
    if set(scores) != set(PAPER_OVERLAP):
        raise ReferenceError("released score games do not match the paper overlap")
    encoded = json.dumps(scores, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != _SCORE_TABLE_SHA256:
        raise ReferenceError("released score values differ from the pinned table")
    return scores


def verify_reference_manifest(reference_root: Path | None = None) -> dict[str, Any]:
    """Validate the pinned source manifest and fixed Atari-16 ROM table.

    ``reference_root`` is accepted for callers that stage an external source
    tree, but no external file is required for this metadata-only check.
    """
    manifest_path = Path(reference_root) / "eadream" / "reference_manifest.json" if reference_root is not None and (Path(reference_root) / "eadream" / "reference_manifest.json").is_file() else REFERENCE_MANIFEST
    document = _read_json(manifest_path)
    if document.get("reference_commit") != REFERENCE_COMMIT:
        raise ReferenceError("reference commit is not pinned")
    raw_files = document.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise ReferenceError("reference file manifest is missing")
    parsed_files: dict[str, tuple[int, str]] = {}
    files: dict[str, dict[str, Any]] = {}
    for name, metadata in raw_files.items():
        if type(name) is not str or "atarirom" in name.lower() or "roms" in name.lower():
            raise ReferenceError("reference manifest must not include ROM paths")
        if not isinstance(metadata, (list, tuple)) or len(metadata) != 2:
            raise ReferenceError(f"reference file metadata for {name!r} is invalid")
        size, digest = metadata
        if type(size) is not int or size < 0 or type(digest) is not str or len(digest) != 64:
            raise ReferenceError(f"reference file metadata for {name!r} is invalid")
        parsed_files[name] = (size, digest)
        files[name] = {"bytes": size, "sha256": digest}
    if parsed_files != _PINNED_REFERENCE_FILES:
        raise ReferenceError("reference file metadata differs from the pinned source manifest")

    manifest_yaml = ROOT / "eadream" / "configs" / "atari16.yaml"
    try:
        from eadream.config import load_manifest

        specs = load_manifest(manifest_yaml)
    except Exception as error:
        raise ReferenceError("fixed Atari-16 manifest is invalid") from error
    roms = {spec.slug: spec.rom_sha256 for spec in specs}
    result = {
        "schema_version": 1,
        "commit": REFERENCE_COMMIT,
        "reference_commit": REFERENCE_COMMIT,
        "manifest_sha256": _sha256(manifest_path),
        "files": files,
        "file_metadata": files,
        "roms": roms,
        "score_source": _read_json(SCORE_DATA).get("source", {}),
    }
    return result


def verify_reference_files(reference_root: Path) -> dict[str, Any]:
    """Hash explicitly supplied upstream files without network access.

    Missing files are reported as ``missing`` rather than treated as a formal
    failure because a clean checkout intentionally does not contain the
    external upstream source tree.
    """
    root = Path(reference_root)
    manifest = verify_reference_manifest()
    records: dict[str, dict[str, Any]] = {}
    for name, metadata in manifest["file_metadata"].items():
        path = root / Path(name)
        if not path.is_file() and root.name.lower() == "eadream":
            path = root / Path(name).name
        if not path.is_file():
            records[name] = {"status": "missing", "path": str(path)}
            continue
        digest = _sha256(path)
        records[name] = {
            "status": "passed" if digest == metadata["sha256"] else "failed",
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest,
            "expected_sha256": metadata["sha256"],
        }
    return {
        "root": str(root),
        "passed": sum(record["status"] == "passed" for record in records.values()),
        "failed": sum(record["status"] == "failed" for record in records.values()),
        "missing": sum(record["status"] == "missing" for record in records.values()),
        "files": records,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _report_hash(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def run_static_gates(
    campaign: ResolvedCampaign | Path | None = None,
    *,
    output_path: Path | None = None,
    test_command: Sequence[str] | None = None,
    run_tests: bool = True,
) -> dict[str, Any]:
    """Run Gate 0--2 checks without constructing an ALE environment.

    Unit/golden and checkpoint tests are represented by one non-integration
    subprocess record.  Tests may pass ``run_tests=False`` when exercising
    aggregation itself with synthetic evidence.
    """
    started = time.monotonic()
    if campaign is None:
        campaign = ROOT / "eadream" / "configs" / "atari16_campaign.yaml"
    if output_path is None:
        output_path = ROOT / "eadream" / "runs" / "preflight" / "static_gates.json"
    output_path = Path(output_path)
    resolved: ResolvedCampaign | None = None
    resolution_error: Exception | None = None
    try:
        resolved = resolve_campaign(Path(campaign)) if isinstance(campaign, (str, Path)) else campaign
    except Exception as error:
        resolution_error = error
    checks: dict[str, Any] = {}
    errors: list[str] = []
    try:
        reference = verify_reference_manifest()
        checks["reference_manifest"] = {"status": "passed", "commit": reference["commit"], "file_count": len(reference["files"]), "rom_count": len(reference["roms"])}
    except Exception as error:
        checks["reference_manifest"] = {"status": "failed", "error": str(error)}
        errors.append(str(error))
    try:
        scores = load_released_scores()
        checks["released_scores"] = {"status": "passed", "games": len(scores), "values_per_game": sorted({len(values) for values in scores.values()})}
    except Exception as error:
        checks["released_scores"] = {"status": "failed", "error": str(error)}
        errors.append(str(error))
    try:
        if resolution_error is not None:
            raise resolution_error
        if not isinstance(resolved, ResolvedCampaign) or len(resolved.runs) != 80:
            raise ConfigError("Atari-16 campaign must resolve exactly 80 runs")
        checks["configuration"] = {"status": "passed", "runs": len(resolved.runs), "games": len({run.config["game"]["slug"] for run in resolved.runs}), "seeds": sorted({run.seed for run in resolved.runs})}
    except Exception as error:
        checks["configuration"] = {"status": "failed", "error": str(error)}
        errors.append(str(error))
    try:
        checks["dependencies"] = {"status": "passed", "versions": validate_formal_versions()}
    except Exception as error:
        checks["dependencies"] = {"status": "failed", "error": str(error)}
        errors.append(str(error))
    # CUDA is a Gate 0 requirement, but never initialize a device here.
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    checks["cuda"] = {"status": "passed" if cuda_ok else "failed", "available": cuda_ok, "torch_cuda": torch.version.cuda}
    if not cuda_ok:
        errors.append("formal EADream requires CUDA")

    command_record: dict[str, Any] = {"command": list(test_command or ()), "duration_seconds": 0.0, "exit_code": None, "report_sha256": None}
    if run_tests and test_command:
        # No test suite ships with this package; callers supply their own
        # command (e.g. an external golden-test runner) to record evidence.
        command = list(test_command)
        test_started = time.monotonic()
        completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, check=False)
        report_path = output_path.parent / "static_gates_tests.log"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        command_record = {"command": command, "duration_seconds": time.monotonic() - test_started, "exit_code": int(completed.returncode), "report_sha256": _report_hash(report_path), "report_path": str(report_path)}
        if completed.returncode != 0:
            errors.append("static test command failed")
    test_status = "not_run" if command_record["exit_code"] is None else "passed" if command_record["exit_code"] == 0 else "failed"
    if test_status == "not_run":
        errors.append("static unit/golden and checkpoint tests were not run")
    checks["unit_and_golden"] = {"status": test_status, **command_record}
    checks["checkpoint_round_trip"] = {"status": test_status, **command_record}

    result = {"schema_version": 1, "gate": "0-2", "passed": not errors, "failed": len(errors), "errors": errors, "checks": checks, "duration_seconds": time.monotonic() - started, "command": command_record["command"]}
    _atomic_json(output_path, result)
    return result


__all__ = [
    "PAPER_OVERLAP",
    "REFERENCE_COMMIT",
    "ReferenceError",
    "load_released_scores",
    "run_static_gates",
    "verify_reference_files",
    "verify_reference_manifest",
]
