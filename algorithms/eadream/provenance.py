"""Preflight validation and immutable provenance for EADream runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from eadream.config import ConfigError, installed_distribution_versions, validate_formal_versions


class PreflightError(RuntimeError):
    """Raised before collection when a run cannot satisfy its declared contract."""


ROOT = Path(__file__).resolve().parents[1]


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_manifest() -> dict[str, str]:
    root = ROOT / "eadream"
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): _file_sha256(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _rom_sha256(env: Any) -> str | None:
    """Return the installed ALE ROM content digest when ALE exposes it."""
    ale = getattr(getattr(env, "unwrapped", env), "ale", None)
    if ale is None:
        return None
    get_rom = getattr(ale, "getROM", None)
    if callable(get_rom):
        try:
            return hashlib.sha256(bytes(get_rom())).hexdigest()
        except (TypeError, ValueError):
            pass
    get_path = getattr(ale, "getROMPath", None)
    if callable(get_path):
        try:
            path = Path(get_path())
            if path.is_file():
                return _file_sha256(path)
        except (OSError, TypeError, ValueError):
            pass
    # ALE-Py 0.12.x no longer exposes getROM/getROMPath on ALEInterface, but
    # the installed, registered game is loaded from its wheel-owned ROM
    # resource. Hash that exact resource as the version-compatible fallback.
    game = getattr(getattr(env, "unwrapped", env), "_game", None)
    if type(game) is str and game and all(char.isalnum() or char == "_" for char in game):
        try:
            import ale_py

            rom_directory = Path(ale_py.__file__).resolve().parent / "roms"
            candidate = rom_directory / f"{game}.bin"
            if candidate.is_file() and candidate.resolve().parent == rom_directory.resolve():
                return _file_sha256(candidate)
        except (OSError, TypeError, ValueError):
            pass
    return None


def _action_meanings(env: Any) -> list[str] | None:
    method = getattr(getattr(env, "unwrapped", env), "get_action_meanings", None)
    if not callable(method):
        return None
    values = method()
    if type(values) not in (list, tuple) or any(type(item) is not str for item in values):
        raise PreflightError("ALE action meanings are malformed")
    return list(values)


_EXPECTED_ACTION_MEANINGS: dict[str, tuple[str, ...]] = {
    "alien": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "asteroids": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE"),
    "bowling": ("NOOP", "FIRE", "UP", "DOWN", "UPFIRE", "DOWNFIRE"),
    "chopper_command": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "enduro": ("NOOP", "FIRE", "RIGHT", "LEFT", "DOWN", "DOWNRIGHT", "DOWNLEFT", "RIGHTFIRE", "LEFTFIRE"),
    "frostbite": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "gopher": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE"),
    "kung_fu_master": ("NOOP", "UP", "RIGHT", "LEFT", "DOWN", "DOWNRIGHT", "DOWNLEFT", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "montezuma_revenge": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "pitfall": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "private_eye": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "seaquest": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "skiing": ("NOOP", "RIGHT", "LEFT"),
    "solaris": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "tennis": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
    "venture": ("NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT", "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE", "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"),
}


def _cuda_record() -> dict[str, Any]:
    available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "available": available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": None,
        "driver_version": None,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_index": 0 if available else None,
        "name": None,
        "total_memory": None,
        "memory_total_bytes": None,
        "compute_capability": None,
    }
    if available:
        try:
            props = torch.cuda.get_device_properties(0)
            result.update({"name": props.name, "total_memory": int(props.total_memory), "memory_total_bytes": int(props.total_memory), "compute_capability": f"{props.major}.{props.minor}"})
        except (AssertionError, RuntimeError):
            pass
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=2,
            )
            value = query.stdout.strip().splitlines()[0] if query.returncode == 0 and query.stdout.strip() else None
            if value:
                fields = [field.strip() for field in value.split(",")]
                result["driver"] = fields[0]
                result["driver_version"] = fields[0]
                if len(fields) > 1 and fields[1]:
                    result["name"] = fields[1]
                if len(fields) > 2 and fields[2]:
                    try:
                        result["memory_total_bytes"] = int(float(fields[2]) * 1024 * 1024)
                        result["total_memory"] = result["memory_total_bytes"]
                    except ValueError:
                        pass
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def validate_preflight(config: Mapping[str, Any], env: Any | None = None) -> dict[str, Any]:
    """Validate formal pins and, when supplied, the installed ALE boundary."""
    formal = bool(config.get("experiment", {}).get("formal", False))
    runtime = config.get("runtime", {})
    versions = installed_distribution_versions()
    if formal:
        try:
            validate_formal_versions()
        except ConfigError as error:
            raise PreflightError(str(error)) from error
        requested_python = str(runtime.get("python_version", ""))
        actual_python = platform.python_version()
        if requested_python and actual_python != requested_python:
            raise PreflightError(f"Python must be {requested_python}, got {actual_python}")
        if not torch.cuda.is_available():
            raise PreflightError("formal EADream requires CUDA")
        requested_cuda = str(runtime.get("torch_cuda_version", ""))
        if requested_cuda and torch.version.cuda != requested_cuda:
            raise PreflightError(f"Torch CUDA must be {requested_cuda}, got {torch.version.cuda!r}")
    result: dict[str, Any] = {"package_versions": versions, "cuda": _cuda_record()}
    if formal:
        # Keep the source identity check offline and fail closed before an ALE
        # object can step.  The reference helper performs no network or ROM
        # access and is imported lazily to avoid a module cycle at import time.
        try:
            from eadream.reference import verify_reference_manifest

            reference = verify_reference_manifest()
            expected_commit = config.get("provenance", {}).get("reference_commit")
            if expected_commit is not None and expected_commit != reference["commit"]:
                raise PreflightError("resolved reference commit differs from embedded manifest")
            result["reference"] = {
                "commit": reference["commit"],
                "manifest_sha256": reference["manifest_sha256"],
                "file_count": len(reference["files"]),
                "rom_count": len(reference["roms"]),
            }
        except PreflightError:
            raise
        except Exception as error:
            raise PreflightError(f"reference manifest validation failed: {error}") from error
    if env is None:
        return result

    meanings = _action_meanings(env)
    action_count = getattr(getattr(env, "action_space", None), "n", None)
    if not isinstance(action_count, (int, np.integer)) or int(action_count) <= 0:
        raise PreflightError("environment must expose a positive discrete action space")
    action_count = int(action_count)
    if meanings is None or len(meanings) != action_count:
        raise PreflightError("ALE action meanings must match the minimal action space")
    expected_meanings = _EXPECTED_ACTION_MEANINGS.get(str(config.get("game", {}).get("slug", "")))
    if formal and expected_meanings is not None and tuple(meanings) != expected_meanings:
        raise PreflightError("ALE action meanings differ from the expected minimal action set")
    protocol = str(config.get("environment", {}).get("start_protocol", "none"))
    if protocol == "fire" and "FIRE" not in meanings:
        raise PreflightError("FIRE protocol requires a FIRE action")
    actual_rom = _rom_sha256(env)
    expected_rom = config.get("game", {}).get("rom_sha256")
    if formal and actual_rom is None:
        raise PreflightError("cannot obtain installed ALE ROM SHA256")
    if actual_rom is not None and expected_rom is not None and actual_rom != expected_rom:
        raise PreflightError("installed ALE ROM SHA256 differs from resolved config")
    result.update({"action_meanings": meanings, "action_count": action_count, "actual_rom_sha256": actual_rom})
    return result


def capture_provenance(config: Mapping[str, Any], env: Any) -> dict[str, Any]:
    """Capture the pre-step facts required to reproduce a resolved run."""
    validation = validate_preflight(config, env)
    reference = ROOT / str(config.get("provenance", {}).get("reference_manifest", "eadream/reference_manifest.json"))
    if not reference.is_file():
        raise PreflightError(f"reference manifest is missing: {reference}")
    reference_document = json.loads(reference.read_text(encoding="utf-8"))
    record: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": os.getpid(),
        "command_line": list(sys.argv),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "platform": platform.platform()},
        "package_versions": validation["package_versions"],
        "torch": {"version": torch.__version__, "cuda": validation["cuda"], "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "opencv": {"version": cv2.__version__, "build": cv2.getBuildInformation()},
        "ale": {"version": validation["package_versions"].get("ale-py"), "action_meanings": validation["action_meanings"], "action_count": validation["action_count"]},
        "rom": {"expected_sha256": config.get("game", {}).get("rom_sha256"), "actual_sha256": validation["actual_rom_sha256"]},
        "resolved_config": dict(config),
        "resolved_config_sha256": config.get("resolved_config_sha256"),
        "reference": {"commit": reference_document.get("reference_commit"), "manifest_sha256": _file_sha256(reference), "files": reference_document.get("files", {})},
        "implementation_files": _implementation_manifest(),
        "cold_start": {"rng_seed_before_environment_and_model_construction": config.get("game", {}).get("seed"), "construction_order": ["seed", "environment", "agent", "engine_after_prefill"], "diagnostic_rng_isolation": True},
        "source_semantics": {"fire_deviation": "FIRE used for Bowling/Tennis train/evaluation/audit/video; upstream diagnostic is no-FIRE", "max_pool": "source_fixed_two_buffer", "encoder": "image / 255.0 - 0.5", "evaluation_rssm": "sampled stochastic state"},
        "preflight": validation,
    }
    record["provenance_sha256"] = canonical_hash(record)
    return record
