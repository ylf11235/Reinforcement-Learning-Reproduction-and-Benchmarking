"""Serial campaign scheduler and durable summaries for EADream."""

from __future__ import annotations

import json
import hashlib
import math
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Literal

from eadream.config import ResolvedCampaign, ResolvedRun


class CampaignCapacityError(RuntimeError):
    """The measured pilot footprint cannot fit the remaining campaign."""


class CampaignStateError(RuntimeError):
    """A persisted campaign state is malformed or incompatible."""


def free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(Path(path)).free)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _key(run: ResolvedRun) -> str:
    return f"{run.config['game']['slug']}/seed_{run.seed:03d}"


def _read_run_state(run: ResolvedRun) -> dict[str, Any] | None:
    path = run.run_dir / "run_state.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _compatible_completed(run: ResolvedRun) -> bool:
    state = _read_run_state(run)
    if not state or state.get("status") != "COMPLETED":
        return False
    if state.get("resolved_config_sha256", state.get("config_hash")) != run.config_hash:
        return False
    result = state.get("result")
    if not isinstance(result, dict):
        return False
    expected = int(run.config.get("budget", {}).get("agent_interactions", 100_000))
    interactions = result.get("agent_interactions", result.get("effective_timesteps"))
    if interactions is not None and int(interactions) < expected:
        return False
    checkpoint = run.run_dir / "checkpoints" / "final.pt"
    metadata = run.run_dir / "checkpoints" / "metadata.json"
    try:
        details = json.loads(metadata.read_text(encoding="utf-8"))
        if details["config_hash"] != run.config_hash or not checkpoint.is_file():
            return False
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != details["sha256"]:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    # If an artifact manifest is present, validate its declared files.  Older
    # runs without one remain valid when their terminal run record is complete.
    manifest = run.run_dir / "artifact_manifest.json"
    if manifest.is_file():
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            files = record.get("files", {})
            if not isinstance(files, dict):
                return False
            for relative, digest in files.items():
                path = run.run_dir / str(relative)
                if not path.is_file():
                    return False
                import hashlib
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != digest:
                    return False
            if record.get("resolved_config_sha256") not in (None, run.config_hash):
                return False
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
    return True


class CampaignLifecycle:
    """Small persistence wrapper; writes state before and after each runner."""

    def __init__(self, campaign: ResolvedCampaign) -> None:
        self.campaign = campaign
        self.root = campaign.run_root / "campaign"
        self.path = self.root / "campaign_state.json"
        self.state = self._load_or_create()

    def _load_or_create(self) -> dict[str, Any]:
        if self.path.is_file():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CampaignStateError("campaign state is unreadable") from error
            if not isinstance(state, dict):
                raise CampaignStateError("campaign state must be a mapping")
            if state.get("campaign_id") != self.campaign.campaign_id:
                raise CampaignStateError("campaign id differs from resolved campaign")
            if "runs" not in state and isinstance(state.get("games"), dict):
                state["runs"] = state["games"]
            if "games" not in state and isinstance(state.get("runs"), dict):
                state["games"] = state["runs"]
            return state
        now = __import__("time").time()
        state = {
            "schema_version": 1,
            "campaign_id": self.campaign.campaign_id,
            "status": "RUNNING",
            "selected_runs": [_key(run) for run in self.campaign.runs],
            "runs": {},
            "current_run": None,
            "created_at_unix": now,
            "updated_at_unix": now,
        }
        state["games"] = state["runs"]
        state["selected_slugs"] = [run.config["game"]["slug"] for run in self.campaign.runs]
        self._write(state)
        return state

    def _write(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at_unix"] = __import__("time").time()
        _atomic_json(self.path, self.state)

    def record(self, key: str, record: dict[str, Any]) -> None:
        self.state.setdefault("runs", {})[key] = record
        self._write()

    def current(self, key: str | None) -> None:
        self.state["current_run"] = key
        self._write()

    def complete(self) -> dict[str, Any]:
        records = self.state.get("runs", {})
        failures = [k for k in self.state.get("selected_runs", []) if records.get(k, {}).get("status") == "FAILED"]
        missing = [k for k in self.state.get("selected_runs", []) if records.get(k, {}).get("status") not in {"COMPLETED", "FAILED", "SKIPPED"}]
        if missing:
            raise CampaignStateError(f"campaign has missing run records: {missing}")
        self.state["status"] = "COMPLETED_WITH_FAILURES" if failures else "COMPLETED"
        self.state["current_run"] = None
        self._write()
        return self.state


def campaign_action_for_run(
    lifecycle: CampaignLifecycle, run: ResolvedRun | dict[str, Any], *, retry_failed: bool = False
) -> Literal["run", "skip_completed", "skip_failed"]:
    if isinstance(run, dict):
        config = run
        digest = str(config.get("resolved_config_sha256", config.get("config_hash", "")))
        seed = int(config.get("game", {}).get("seed", 0))
        output = config.get("output", {})
        root = Path(str(output.get("campaign_root", "")))
        run_dir = root / "games" / str(config.get("game", {}).get("slug", "unknown")) / f"seed_{seed:03d}"
        run = ResolvedRun(config, digest, seed, run_dir)
    key = _key(run)
    record = lifecycle.state.get("runs", {}).get(key, {})
    if _compatible_completed(run):
        return "skip_completed"
    if record.get("status") == "FAILED" and not retry_failed:
        return "skip_failed"
    return "run"


def run_campaign(
    campaign: ResolvedCampaign,
    *,
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    measured_run_bytes: int | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Execute resolved runs in order, with one active runner at a time."""
    lifecycle = CampaignLifecycle(campaign)
    if retry_failed and any(
        lifecycle.state.get("runs", {}).get(_key(run), {}).get("status") == "FAILED"
        for run in campaign.runs
    ):
        raise CampaignStateError(
            "retry_failed is unsupported without a periodic recovery checkpoint; use a new campaign output root"
        )
    pending = [run for run in campaign.runs if not _compatible_completed(run)]
    if measured_run_bytes is not None:
        if measured_run_bytes < 0:
            raise ValueError("measured_run_bytes must be non-negative")
        required = math.ceil(int(measured_run_bytes) * len(pending) * 1.20)
        available = free_bytes(campaign.run_root)
        if available < required:
            raise CampaignCapacityError(f"free space is below required campaign projection: {required} bytes (available {available})")
    if runner is None:
        from eadream.experiment import run_one
        runner = lambda config: run_one(config)
    for run in campaign.runs:
        key = _key(run)
        action = campaign_action_for_run(lifecycle, run, retry_failed=retry_failed)
        if action == "skip_completed":
            lifecycle.record(key, {"status": "SKIPPED", "config_hash": run.config_hash, "run_dir": str(run.run_dir), "resolved_config_sha256": run.config_hash})
            continue
        if action == "skip_failed":
            continue
        base = {"status": "RUNNING", "config_hash": run.config_hash, "resolved_config_sha256": run.config_hash, "run_dir": str(run.run_dir), "env_id": run.config["game"]["env_id"], "seed": run.seed}
        lifecycle.record(key, base)
        lifecycle.current(key)
        try:
            result = runner(run.config)
            if not isinstance(result, dict) or result.get("status", "COMPLETED") not in {"COMPLETED", "SKIPPED"}:
                raise CampaignStateError("runner did not return COMPLETED")
            lifecycle.record(key, {**base, "status": "COMPLETED", "result": result.get("result", result)})
        except Exception as error:
            lifecycle.record(key, {**base, "status": "FAILED", "failure_kind": "runner_exception", "error_type": type(error).__name__, "error_message": str(error), "traceback": traceback.format_exc()})
            if campaign.controls.on_failure == "stop":
                raise
        finally:
            lifecycle.current(None)
    state = lifecycle.complete()
    try:
        from eadream.archive import build_summary
        build_summary(campaign.run_root, state)
    except ImportError:
        pass
    return state


__all__ = ["CampaignCapacityError", "CampaignStateError", "CampaignLifecycle", "campaign_action_for_run", "run_campaign", "free_bytes", "build_summary"]


def build_summary(campaign_root: Path, campaign_state: dict[str, Any]) -> dict[str, Any]:
    from eadream.archive import build_summary as _build_summary
    return _build_summary(campaign_root, campaign_state)
