"""Persistent serial scheduling for the Phase-1 Atari PPO campaign."""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from baselines_common.utils import atomic_write_json, atomic_write_yaml

from ppo_sb3.archive import build_summary
from ppo_sb3.config import CampaignControls, ResolvedCampaign
from ppo_sb3.evaluation import checkpoint_path_from_metadata
from ppo_sb3.provenance import sha256_path


class CampaignStateError(RuntimeError):
    """Raised when a persisted campaign cannot advance safely."""


_IMPORT_SEMANTIC_FIELDS = (
    "game",
    "budget",
    "runtime",
    "environment",
    "algorithm",
    "evaluation",
    "checkpoint",
    "video",
)


def _training_semantics(config: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _IMPORT_SEMANTIC_FIELDS if field not in config]
    if missing:
        raise CampaignStateError(f"resolved config is missing training semantics: {missing}")
    return {field: config[field] for field in _IMPORT_SEMANTIC_FIELDS}


def _read_mapping(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignStateError(f"missing {description}: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignStateError(f"invalid {description}: {path}")
    return value


def _rewrite_paths(value: Any, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, source_root, target_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(item, source_root, target_root) for item in value]
    if isinstance(value, str):
        try:
            relative = Path(value).resolve().relative_to(source_root)
        except (OSError, ValueError):
            return value
        return str(target_root / relative)
    return value


def _rewrite_artifact_metadata(
    value: Any, source_root: Path, target_root: Path, target_config_hash: str
) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                target_config_hash
                if key in {"config_hash", "resolved_config_sha256"} and isinstance(item, str)
                else _rewrite_artifact_metadata(item, source_root, target_root, target_config_hash)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_artifact_metadata(item, source_root, target_root, target_config_hash)
            for item in value
        ]
    return _rewrite_paths(value, source_root, target_root)


def _rewrite_copied_json_metadata(
    target_run: Path, source_run: Path, target_config_hash: str
) -> None:
    target_owned = {"imported_from.json", "provenance.json", "run_state.json"}
    for path in target_run.rglob("*.json"):
        relative = path.relative_to(target_run)
        if "imported_source" in relative.parts or relative.as_posix() in target_owned:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        rewritten = _rewrite_artifact_metadata(
            value, source_run, target_run, target_config_hash
        )
        if rewritten != value:
            atomic_write_json(path, rewritten)


def _validate_target_checkpoint(target_run: Path, expected_hash: str, slug: str) -> None:
    target_model = checkpoint_path_from_metadata(target_run / "final_checkpoint")
    try:
        actual_hash = sha256_path(target_model)
    except FileNotFoundError as error:
        raise CampaignStateError(f"missing target final checkpoint for {slug}: {target_model}") from error
    if actual_hash != expected_hash:
        raise CampaignStateError(f"target final checkpoint hash mismatch for {slug}")


def _imported_from_record(
    *,
    source_campaign_root: Path,
    source_state: dict[str, Any],
    source_run: Path,
    source_run_state: dict[str, Any],
    source_snapshot: Path,
    target_config: dict[str, Any],
    expected_hash: str,
    imported_at_unix: float | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "imported_at_unix": imported_at_unix if imported_at_unix is not None else time.time(),
        "source_campaign_id": source_state.get("campaign_id"),
        "source_campaign_root": str(source_campaign_root),
        "source_run_dir": str(source_run),
        "source_config_hash": source_run_state.get("config_hash"),
        "source_run_state_sha256": sha256_path(source_snapshot / "run_state.json"),
        "source_provenance_sha256": sha256_path(source_snapshot / "provenance.json"),
        "target_config_hash": target_config["resolved_config_sha256"],
        "verified_semantic_fields": list(_IMPORT_SEMANTIC_FIELDS),
        "final_checkpoint_sha256": expected_hash,
    }


def _target_import_provenance(
    campaign: ResolvedCampaign,
    target_config: dict[str, Any],
    source_snapshot: Path,
    imported_from: dict[str, Any],
) -> dict[str, Any]:
    source_provenance = _read_mapping(
        source_snapshot / "provenance.json", "source provenance snapshot"
    )
    slug = target_config["game"]["slug"]
    env_id = target_config["game"]["env_id"]
    if source_provenance.get("slug") != slug or source_provenance.get("env_id") != env_id:
        raise CampaignStateError(f"source provenance identity mismatch for {slug}")
    source_config = Path(str(target_config["source_config"])).resolve()
    return {
        "schema_version": 1,
        "captured_at_unix": imported_from["imported_at_unix"],
        "campaign_id": campaign.controls.id,
        "execution": "imported_completed_run",
        "slug": slug,
        "env_id": env_id,
        "rom_sha256": source_provenance.get("rom_sha256"),
        "source_config": str(source_config),
        "source_config_sha256": sha256_path(source_config),
        "game_config_sha256": target_config.get(
            "game_config_sha256", target_config["resolved_config_sha256"]
        ),
        "resolved_config_sha256": target_config["resolved_config_sha256"],
        "campaign_manifest": str(campaign.manifest_path.resolve()),
        "campaign_manifest_sha256": sha256_path(campaign.manifest_path.resolve()),
        "source_provenance_sha256": imported_from["source_provenance_sha256"],
        "imported_from": imported_from,
    }


def _write_target_import_metadata(
    *,
    campaign: ResolvedCampaign,
    target_config: dict[str, Any],
    source_campaign_root: Path,
    source_state: dict[str, Any],
    source_run: Path,
    source_run_state: dict[str, Any],
    source_snapshot: Path,
    target_run: Path,
    source_result: dict[str, Any],
    expected_hash: str,
    imported_at_unix: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _rewrite_copied_json_metadata(
        target_run, source_run, target_config["resolved_config_sha256"]
    )
    target_input = _read_mapping(Path(str(target_config["source_config"])), "target input config")
    atomic_write_yaml(target_run / "input_config.yaml", target_input)
    atomic_write_yaml(target_run / "resolved_config.yaml", target_config)
    imported_from = _imported_from_record(
        source_campaign_root=source_campaign_root,
        source_state=source_state,
        source_run=source_run,
        source_run_state=source_run_state,
        source_snapshot=source_snapshot,
        target_config=target_config,
        expected_hash=expected_hash,
        imported_at_unix=imported_at_unix,
    )
    atomic_write_json(target_run / "imported_from.json", imported_from)

    target_result = _rewrite_paths(copy.deepcopy(source_result), source_run, target_run)
    target_run_state = copy.deepcopy(source_run_state)
    target_run_state["config_hash"] = target_config["resolved_config_sha256"]
    target_run_state["result"] = target_result
    target_run_state["imported_from"] = imported_from
    atomic_write_json(target_run / "run_state.json", target_run_state)
    atomic_write_json(
        target_run / "provenance.json",
        _target_import_provenance(campaign, target_config, source_snapshot, imported_from),
    )
    return imported_from, target_result


def import_completed_games(campaign: ResolvedCampaign, source_campaign_root: Path) -> list[str]:
    """Copy compatible completed runs and seed target campaign skip records."""
    source_campaign_root = source_campaign_root.resolve()
    source_state_path = source_campaign_root / "campaign" / "campaign_state.json"
    if not source_state_path.is_file():
        raise CampaignStateError(f"missing source campaign state: {source_state_path}")
    source_state = json.loads(source_state_path.read_text(encoding="utf-8"))
    source_records = source_state.get("games")
    if not isinstance(source_records, dict):
        raise CampaignStateError("source campaign games must be a mapping")

    lifecycle = CampaignLifecycle(campaign.controls.run_root, campaign.controls, campaign.manifest)
    lifecycle.validate_selected_slugs(_selected_slugs(campaign))
    current_game = lifecycle.state.get("current_game")
    if isinstance(current_game, str) and current_game:
        raise CampaignStateError(
            f"cannot import completed games while game {current_game} is running"
        )
    lifecycle.write_resolved_campaign(campaign)
    imported: list[str] = []
    for target_config, _ in campaign.games:
        slug = target_config["game"]["slug"]
        source_record = source_records.get(slug)
        if not isinstance(source_record, dict) or source_record.get("status") != "COMPLETED":
            continue

        source_run = (source_campaign_root / "games" / slug / "seed_000").resolve()
        recorded_run = Path(str(source_record.get("run_dir", ""))).resolve()
        if recorded_run != source_run:
            raise CampaignStateError(f"source run directory mismatch for {slug}: {recorded_run}")
        source_run_state = _read_mapping(source_run / "run_state.json", "source run state")
        if source_run_state.get("status") != "COMPLETED":
            raise CampaignStateError(f"source run is not completed for {slug}")
        source_config = _read_mapping(source_run / "resolved_config.yaml", "source resolved config")
        if _training_semantics(source_config) != _training_semantics(target_config):
            raise CampaignStateError(f"training semantics do not match for imported game {slug}")

        source_result = source_run_state.get("result")
        if not isinstance(source_result, dict):
            raise CampaignStateError(f"source run has no completed result for {slug}")
        recorded_result = source_record.get("result")
        if not isinstance(recorded_result, dict) or recorded_result != source_result:
            raise CampaignStateError(f"source campaign and run results differ for {slug}")
        final_model = checkpoint_path_from_metadata(source_run / "final_checkpoint")
        expected_hash = source_result.get("final_checkpoint_sha256")
        if not isinstance(expected_hash, str) or sha256_path(final_model) != expected_hash:
            raise CampaignStateError(f"source final checkpoint hash mismatch for {slug}")

        target_run = Path(str(target_config["output"]["run_root"])).resolve()
        if target_run.exists():
            existing = target_run / "imported_from.json"
            target_state = target_run / "run_state.json"
            if existing.is_file() and target_state.is_file():
                existing_record = json.loads(existing.read_text(encoding="utf-8"))
                state = json.loads(target_state.read_text(encoding="utf-8"))
                existing_source_root = Path(
                    str(existing_record.get("source_campaign_root", ""))
                ).resolve()
                if (
                    state.get("config_hash") == target_config["resolved_config_sha256"]
                    and existing_record.get("target_config_hash")
                    == target_config["resolved_config_sha256"]
                    and existing_source_root == source_campaign_root
                ):
                    _validate_target_checkpoint(target_run, expected_hash, slug)
                    imported_at = existing_record.get("imported_at_unix")
                    repaired_imported_from, repaired_result = _write_target_import_metadata(
                        campaign=campaign,
                        target_config=target_config,
                        source_campaign_root=source_campaign_root,
                        source_state=source_state,
                        source_run=source_run,
                        source_run_state=source_run_state,
                        source_snapshot=target_run / "imported_source",
                        target_run=target_run,
                        source_result=source_result,
                        expected_hash=expected_hash,
                        imported_at_unix=(
                            float(imported_at) if isinstance(imported_at, (int, float)) else None
                        ),
                    )
                    lifecycle.ensure_imported_game(
                        slug,
                        {
                            **_record_base(target_config),
                            "status": "COMPLETED",
                            "result": repaired_result,
                            "imported_from": repaired_imported_from,
                        },
                    )
                    imported.append(slug)
                    continue
            raise CampaignStateError(f"target run directory already exists: {target_run}")

        shutil.copytree(source_run, target_run)
        _validate_target_checkpoint(target_run, expected_hash, slug)
        source_snapshot = target_run / "imported_source"
        source_snapshot.mkdir()
        for name in ("run_state.json", "input_config.yaml", "resolved_config.yaml", "provenance.json"):
            copied = target_run / name
            if copied.is_file():
                shutil.copy2(copied, source_snapshot / name)
        imported_from, target_result = _write_target_import_metadata(
            campaign=campaign,
            target_config=target_config,
            source_campaign_root=source_campaign_root,
            source_state=source_state,
            source_run=source_run,
            source_run_state=source_run_state,
            source_snapshot=source_snapshot,
            target_run=target_run,
            source_result=source_result,
            expected_hash=expected_hash,
        )
        lifecycle.ensure_imported_game(
            slug,
            {
                **_record_base(target_config),
                "status": "COMPLETED",
                "result": target_result,
                "imported_from": imported_from,
            },
        )
        imported.append(slug)
    return imported


def _campaign_root(controls: CampaignControls) -> Path:
    return controls.run_root


def _selected_slugs(campaign: ResolvedCampaign) -> list[str]:
    return [config["game"]["slug"] for config, _ in campaign.games]


def _resolved_campaign_document(campaign: ResolvedCampaign) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign": {
            "id": campaign.controls.id,
            "run_root": str(campaign.controls.run_root),
            "archive_root": str(campaign.controls.archive_root),
            "scheduling": campaign.controls.scheduling,
            "on_failure": campaign.controls.on_failure,
            "automatic_resume": campaign.controls.automatic_resume,
            "max_retries_per_game": campaign.controls.max_retries_per_game,
            "skip_completed": campaign.controls.skip_completed,
        },
        "games": [config for config, _ in campaign.games],
    }


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def initial_campaign_state(campaign: ResolvedCampaign) -> dict[str, Any]:
    """Create the state persisted before any Phase-1 game is started."""
    return {
        "schema_version": 1,
        "campaign_id": campaign.controls.id,
        "status": "RUNNING",
        "selected_slugs": _selected_slugs(campaign),
        "games": {},
        "current_game": None,
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }


class CampaignLifecycle:
    """Own the durable campaign state and append-only game status log."""

    _TERMINAL_GAME_STATUSES = {"COMPLETED", "FAILED"}

    def __init__(self, campaign_root: Path, controls: CampaignControls, manifest: dict[str, Any]) -> None:
        self.campaign_root = campaign_root
        self.controls = controls
        self.manifest = manifest
        self.campaign_dir = campaign_root / "campaign"
        self.state_path = self.campaign_dir / "campaign_state.json"
        self.status_path = self.campaign_dir / "campaign_status.jsonl"
        self.campaign_dir.mkdir(parents=True, exist_ok=True)

        if self.state_path.is_file():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self._state.get("campaign_id") != controls.id:
                raise CampaignStateError("campaign state id does not match the requested campaign")
        else:
            atomic_write_yaml(self.campaign_dir / "input_manifest.yaml", manifest)
            self._state = initial_campaign_state_from_manifest(controls, manifest)
            self._write()

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def _write(self) -> None:
        self._state["updated_at_unix"] = time.time()
        atomic_write_json(self.state_path, self._state)

    def write_resolved_campaign(self, campaign: ResolvedCampaign) -> None:
        resolved_path = self.campaign_dir / "resolved_campaign.yaml"
        if not resolved_path.exists():
            atomic_write_yaml(resolved_path, _resolved_campaign_document(campaign))

    def mark_running(self) -> None:
        self._state["status"] = "RUNNING"
        self._state["current_game"] = None
        self._write()

    def validate_selected_slugs(self, selected_slugs: list[str]) -> None:
        if self._state.get("selected_slugs") != selected_slugs:
            raise CampaignStateError("campaign state selected games do not match the requested campaign")

    def record_game(self, slug: str, record: dict[str, Any]) -> None:
        stored = {**record, "updated_at_unix": time.time()}
        self._state.setdefault("games", {})[slug] = stored
        self._state["current_game"] = slug if stored["status"] == "RUNNING" else None
        self._write()
        _append_jsonl(self.status_path, {"slug": slug, **stored})

    def ensure_imported_game(self, slug: str, record: dict[str, Any]) -> None:
        if record.get("status") != "COMPLETED":
            raise CampaignStateError("an imported game record must be COMPLETED")
        current = self._state.setdefault("games", {}).get(slug)
        comparable = (
            {key: value for key, value in current.items() if key != "updated_at_unix"}
            if isinstance(current, dict)
            else None
        )
        if comparable == record:
            return
        stored = {**record, "updated_at_unix": time.time()}
        self._state["games"][slug] = stored
        self._write()
        _append_jsonl(self.status_path, {"slug": slug, **stored})

    def complete(self) -> dict[str, Any]:
        games = self._state.get("games", {})
        selected = self._state.get("selected_slugs", [])
        missing = [slug for slug in selected if slug not in games]
        nonterminal = [
            slug
            for slug in selected
            if slug in games and games[slug].get("status") not in self._TERMINAL_GAME_STATUSES
        ]
        if missing or nonterminal:
            raise CampaignStateError(
                f"cannot complete campaign with missing={missing} nonterminal={nonterminal}"
            )
        failures = [slug for slug in selected if games[slug]["status"] == "FAILED"]
        self._state["status"] = "COMPLETED_WITH_FAILURES" if failures else "COMPLETED"
        self._state["current_game"] = None
        self._write()
        _append_jsonl(
            self.status_path,
            {
                "event": "campaign_terminal",
                "status": self._state["status"],
                "failed_slugs": failures,
                "updated_at_unix": self._state["updated_at_unix"],
            },
        )
        return self._state


def initial_campaign_state_from_manifest(
    controls: CampaignControls, manifest: dict[str, Any]
) -> dict[str, Any]:
    games = manifest.get("games")
    if not isinstance(games, list):
        raise CampaignStateError("campaign manifest games must be a list")
    state = {
        "schema_version": 1,
        "campaign_id": controls.id,
        "status": "RUNNING",
        "selected_slugs": [game["slug"] for game in games],
        "games": {},
        "current_game": None,
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
    }
    return state


def _record_base(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PENDING",
        "env_id": config["game"]["env_id"],
        "config_hash": config["resolved_config_sha256"],
        "game_config_sha256": config.get("game_config_sha256"),
        "run_dir": str(config["output"]["run_root"]),
    }


def campaign_action_for_game(
    lifecycle: CampaignLifecycle, config: dict[str, Any], *, retry_failed: bool = False
) -> Literal["run", "skip_completed", "skip_failed", "mark_interrupted"]:
    """Choose the action for a resolved game using its persisted record."""
    slug = config["game"]["slug"]
    record = lifecycle.state.get("games", {}).get(slug)
    if not isinstance(record, dict) or record.get("config_hash") != config["resolved_config_sha256"]:
        return "run"
    if record.get("status") == "COMPLETED" and lifecycle.controls.skip_completed:
        return "skip_completed"
    if record.get("status") == "FAILED":
        return "run" if retry_failed else "skip_failed"
    if record.get("status") == "RUNNING" and not lifecycle.controls.automatic_resume:
        return "mark_interrupted"
    return "run"


def _last_checkpoint_record(run_dir: Path) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints" / "last"
    model_path = checkpoint_path_from_metadata(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    result: dict[str, Any] = {}
    if model_path.is_file():
        result["last_checkpoint"] = str(model_path)
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return result
        if "transitions" in metadata:
            result["last_known_timestep"] = metadata["transitions"]
        if "model_sha256" in metadata:
            result["last_checkpoint_sha256"] = metadata["model_sha256"]
    return result


def _failure_record(config: dict[str, Any], error: BaseException) -> dict[str, Any]:
    run_dir = Path(str(config["output"]["run_root"]))
    return {
        **_record_base(config),
        "status": "FAILED",
        "failure_kind": "runner_exception",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        **_last_checkpoint_record(run_dir),
    }


def run_phase_campaign(
    campaign: ResolvedCampaign,
    *,
    game_runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """Run the resolved Phase-1 games serially, recording failures and continuing."""
    lifecycle = CampaignLifecycle(_campaign_root(campaign.controls), campaign.controls, campaign.manifest)
    lifecycle.validate_selected_slugs(_selected_slugs(campaign))
    lifecycle.write_resolved_campaign(campaign)
    if retry_failed:
        lifecycle.mark_running()
    if game_runner is None:
        from ppo_sb3.experiment import run_game

        def game_runner(config: dict[str, Any]) -> dict[str, Any]:
            return run_game(config, retry_failed=retry_failed)

    for config, _ in campaign.games:
        slug = config["game"]["slug"]
        action = campaign_action_for_game(lifecycle, config, retry_failed=retry_failed)
        if action in {"skip_completed", "skip_failed"}:
            continue
        if action == "mark_interrupted":
            lifecycle.record_game(
                slug,
                {
                    **_record_base(config),
                    "status": "FAILED",
                    "failure_kind": "interrupted_without_resume",
                    "error_type": "CampaignStateError",
                    "error_message": "stale RUNNING game encountered with automatic resume disabled",
                    **_last_checkpoint_record(Path(str(config["output"]["run_root"]))),
                },
            )
            continue

        lifecycle.record_game(slug, {**_record_base(config), "status": "RUNNING"})
        try:
            run_state = game_runner(config)
            if run_state.get("status") != "COMPLETED":
                raise CampaignStateError(f"game runner returned {run_state.get('status')!r}, not COMPLETED")
            lifecycle.record_game(
                slug,
                {
                    **_record_base(config),
                    "status": "COMPLETED",
                    "result": run_state.get("result", {}),
                },
            )
        except Exception as error:
            lifecycle.record_game(slug, _failure_record(config, error))
            if campaign.controls.on_failure != "record_and_continue":
                raise
    terminal_state = lifecycle.complete()
    build_summary(_campaign_root(campaign.controls), terminal_state)
    return terminal_state
