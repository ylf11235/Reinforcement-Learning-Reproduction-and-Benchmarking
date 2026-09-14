"""Reproducibility records for PPO-SB3 Atari campaign runs."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ale_py
import torch

from baselines_common.utils import atomic_write_json

from ppo_sb3.config import ROOT, ResolvedCampaign, installed_versions


_SNAPSHOT_IGNORES = shutil.ignore_patterns("__pycache__", "runs", "archives", ".pytest_cache")


def sha256_path(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_target(destination: Path, source: Path) -> Path:
    try:
        relative = source.resolve().relative_to(ROOT)
    except ValueError:
        relative = Path("external") / source.name
    return destination / relative


def snapshot_sources(destination: Path, sources: list[Path]) -> dict[str, str]:
    """Copy source inputs once and return hashes for every copied regular file."""
    if destination.exists():
        raise FileExistsError(f"source snapshot already exists: {destination}")
    destination.mkdir(parents=True)
    for source in sources:
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        target = _snapshot_target(destination, source)
        if source.is_dir():
            shutil.copytree(source, target, ignore=_SNAPSHOT_IGNORES)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return {
        file.relative_to(destination).as_posix(): sha256_path(file)
        for file in sorted(destination.rglob("*"))
        if file.is_file()
    }


def runtime_details() -> dict[str, Any]:
    """Collect installed package and CUDA facts without initializing an environment."""
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": installed_versions(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _rom_id(env_id: str) -> str:
    match = re.fullmatch(r"ALE/([A-Za-z0-9]+)-v\d+", env_id)
    if match is None:
        raise ValueError(f"cannot derive an ALE ROM id from {env_id!r}")
    return re.sub(r"(?<!^)(?=[A-Z])", "_", match.group(1)).lower()


def rom_sha256(env_id: str) -> str:
    """Resolve the configured ROM from ALE-Py and hash the ROM bytes."""
    rom_path = ale_py.roms.get_rom_path(_rom_id(env_id))
    if rom_path is None:
        raise FileNotFoundError(f"ALE-Py does not provide a ROM for {env_id}")
    return sha256_path(Path(rom_path))


def _campaign_sources(campaign: ResolvedCampaign) -> list[Path]:
    sources = [
        ROOT / "algorithms" / "ppo_sb3",
        ROOT / "baselines_common",
        campaign.manifest_path,
    ]
    protocol = ROOT / "docs" / "reports" / "ppo-experiments.md"
    if protocol.is_file():
        sources.append(protocol)
    sources.extend(Path(str(config["source_config"])) for config, _ in campaign.games)
    return sources


def capture_campaign_provenance(
    campaign_root: Path, campaign: ResolvedCampaign, command: list[str]
) -> dict[str, Any]:
    """Freeze campaign inputs and runtime facts before serial scheduling begins."""
    campaign_dir = campaign_root / "campaign"
    provenance_path = campaign_dir / "provenance.json"
    if provenance_path.is_file():
        return json.loads(provenance_path.read_text(encoding="utf-8"))

    snapshot_dir = campaign_dir / "source_snapshot"
    hashes = snapshot_sources(snapshot_dir, _campaign_sources(campaign))
    selected_hashes = {
        config["game"]["slug"]: sha256_path(Path(str(config["source_config"])))
        for config, _ in campaign.games
    }
    record = {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "campaign_id": campaign.controls.id,
        "command": list(command),
        "manifest": str(campaign.manifest_path),
        "manifest_sha256": sha256_path(campaign.manifest_path),
        "selected_game_yaml_sha256": selected_hashes,
        "resolved_config_sha256": {
            config["game"]["slug"]: config_hash for config, config_hash in campaign.games
        },
        "runtime": runtime_details(),
        "source_hashes": hashes,
    }
    atomic_write_json(campaign_dir / "source_hashes.json", hashes)
    atomic_write_json(provenance_path, record)
    return record


def capture_game_provenance(
    run_dir: Path, config: dict[str, Any], campaign_provenance: dict[str, Any]
) -> dict[str, Any]:
    """Persist game-specific configuration, ROM, and command facts before training."""
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    source_config = Path(str(config["source_config"]))
    record = {
        "schema_version": 1,
        "captured_at_unix": time.time(),
        "campaign_id": campaign_provenance.get("campaign_id"),
        "command": list(campaign_provenance.get("command", [])),
        "slug": config["game"]["slug"],
        "env_id": config["game"]["env_id"],
        "rom_sha256": rom_sha256(config["game"]["env_id"]),
        "source_config": str(source_config),
        "source_config_sha256": sha256_path(source_config),
        "game_config_sha256": config.get("game_config_sha256", config["resolved_config_sha256"]),
        "resolved_config_sha256": config["resolved_config_sha256"],
        "campaign_manifest_sha256": campaign_provenance.get("manifest_sha256"),
    }
    atomic_write_json(provenance_path, record)
    return record
