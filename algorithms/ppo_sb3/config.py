"""Configuration loading and deterministic resolution for PPO-SB3 Atari runs."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]  # repository root
_START_PROTOCOLS = {
    "bank_heist": "move",
    "bowling": "fire",
    "breakout": "fire",
    "tennis": "fire",
}
_REQUIRED_SECTIONS = (
    "experiment",
    "suite",
    "game",
    "budget",
    "runtime",
    "environment",
    "algorithm",
    "evaluation",
    "checkpoint",
    "output",
)
_PHASE1_MANIFEST_FIELDS = {"schema_version", "experiment", "suite", "games", "campaign"}
_CAMPAIGN_CONTROL_FIELDS = {
    "id",
    "run_root",
    "archive_root",
    "scheduling",
    "on_failure",
    "automatic_resume",
    "max_retries_per_game",
    "skip_completed",
}


@dataclass(frozen=True)
class CampaignControls:
    """Campaign-owned controls that cannot alter game-training semantics."""

    id: str
    run_root: Path
    archive_root: Path
    scheduling: str
    on_failure: str
    automatic_resume: bool
    max_retries_per_game: int
    skip_completed: bool


@dataclass(frozen=True)
class ResolvedCampaign:
    """A Phase-1 manifest and its independently resolved game configurations."""

    manifest_path: Path
    controls: CampaignControls
    games: list[tuple[dict[str, Any], str]]
    manifest: dict[str, Any]


class ConfigError(ValueError):
    """Raised when a campaign or per-game configuration is invalid."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    return _mapping(raw, str(path))


def _resolve_reference(reference: str | Path, relative_to: Path) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path
    local = relative_to / path
    if local.exists():
        return local.resolve()
    return (ROOT / path).resolve()


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_versions() -> dict[str, str | None]:
    """Return the package versions that materially affect this protocol."""
    return {
        "gymnasium": _package_version("gymnasium"),
        "ale_py": _package_version("ale-py"),
        "stable_baselines3": _package_version("stable-baselines3"),
        "torch": _package_version("torch"),
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    """Read an ordered manifest or phase manifest and validate unique entries."""
    raw = _load_yaml(path.resolve())
    suite = _mapping(raw.get("suite"), "suite")
    games = raw.get("games")
    if not isinstance(games, list) or not games:
        raise ConfigError("manifest games must be a non-empty list")

    expected_count = suite.get("expected_game_count", suite.get("game_count"))
    if expected_count is not None and int(expected_count) != len(games):
        raise ConfigError(
            f"manifest expected {expected_count} games but contains {len(games)} entries"
        )

    entries: list[dict[str, str]] = []
    seen_slugs: set[str] = set()
    for index, value in enumerate(games):
        entry = _mapping(value, f"games[{index}]")
        slug = entry.get("slug")
        env_id = entry.get("env_id")
        if not isinstance(slug, str) or not slug:
            raise ConfigError(f"games[{index}].slug must be a non-empty string")
        if not isinstance(env_id, str) or not env_id:
            raise ConfigError(f"games[{index}].env_id must be a non-empty string")
        if slug in seen_slugs:
            raise ConfigError(f"manifest contains duplicate game slug: {slug}")
        seen_slugs.add(slug)
        entries.append({"slug": slug, "env_id": env_id})
    return entries


def load_game_config(path: Path) -> dict[str, Any]:
    """Load one self-contained official game YAML without applying defaults."""
    return copy.deepcopy(_load_yaml(path.resolve()))


def _validate_manifest_membership(config: dict[str, Any], config_path: Path, manifest_path: Path) -> None:
    game = _mapping(config["game"], "game")
    slug = game.get("slug")
    env_id = game.get("env_id")
    if not isinstance(slug, str) or not isinstance(env_id, str):
        raise ConfigError(f"{config_path}: game.slug and game.env_id must be strings")
    matches = {entry["slug"]: entry["env_id"] for entry in load_manifest(manifest_path)}
    if matches.get(slug) != env_id:
        raise ConfigError(
            f"{config_path}: game {slug}/{env_id} is not an exact member of {manifest_path}"
        )


def _validate_and_resolve(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    missing = [section for section in _REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"{config_path}: missing required sections: {', '.join(missing)}")
    for section in _REQUIRED_SECTIONS:
        _mapping(config[section], section)

    game = config["game"]
    budget = config["budget"]
    runtime = config["runtime"]
    environment = config["environment"]
    algorithm = config["algorithm"]
    output = config["output"]

    slug = game.get("slug")
    env_id = game.get("env_id")
    if not isinstance(slug, str) or not slug:
        raise ConfigError(f"{config_path}: game.slug must be a non-empty string")
    if not isinstance(env_id, str) or not env_id.startswith("ALE/") or not env_id.endswith("-v5"):
        raise ConfigError(f"{config_path}: game.env_id must be an ALE/*-v5 id")

    try:
        requested = int(budget["requested"])
        n_steps = int(algorithm["n_steps"])
        num_envs = int(runtime["num_envs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(
            f"{config_path}: budget.requested, algorithm.n_steps and runtime.num_envs are required integers"
        ) from error
    if requested <= 0 or n_steps <= 0 or num_envs <= 0:
        raise ConfigError(f"{config_path}: requested, n_steps and num_envs must be positive")

    rollout_size = n_steps * num_envs
    effective = int(math.ceil(requested / rollout_size) * rollout_size)
    declared_effective = budget.get("effective")
    if declared_effective is not None and int(declared_effective) != effective:
        raise ConfigError(
            f"{config_path}: effective budget {declared_effective} does not match rollout-aligned {effective}"
        )
    budget["requested"] = requested
    budget["effective"] = effective
    budget["rollout_size"] = rollout_size

    environment.setdefault("max_num_frames_per_episode", 108_000)
    environment.setdefault("start_protocol", _START_PROTOCOLS.get(slug, "none"))
    environment.setdefault("render_mode", "rgb_array")
    if int(environment["max_num_frames_per_episode"]) != 108_000:
        raise ConfigError(f"{config_path}: max_num_frames_per_episode must be 108000")
    if environment["start_protocol"] not in {"none", "fire", "move"}:
        raise ConfigError(f"{config_path}: invalid start_protocol {environment['start_protocol']!r}")
    if environment.get("terminal_on_life_loss") is not False:
        raise ConfigError(f"{config_path}: terminal_on_life_loss must be false")
    if environment.get("eval_clip_reward") is not False:
        raise ConfigError(f"{config_path}: eval_clip_reward must be false for raw-score evaluation")
    if not isinstance(environment.get("train_clip_reward"), bool):
        raise ConfigError(f"{config_path}: train_clip_reward must be a boolean")
    if environment["render_mode"] != "rgb_array":
        raise ConfigError(f"{config_path}: render_mode must be rgb_array")

    if runtime.get("vector_env") != "DummyVecEnv":
        raise ConfigError(f"{config_path}: runtime.vector_env must be DummyVecEnv")
    if not isinstance(runtime.get("require_device"), bool):
        raise ConfigError(f"{config_path}: runtime.require_device must be a boolean")
    if not isinstance(output.get("run_root"), str) or not output["run_root"]:
        raise ConfigError(f"{config_path}: output.run_root must be a non-empty string")

    resolved = copy.deepcopy(config)
    resolved["provenance"].setdefault("installed_versions", installed_versions())
    resolved["source_config"] = str(config_path)
    return resolved


def _hash_config(config: dict[str, Any]) -> str:
    payload = copy.deepcopy(config)
    payload.pop("resolved_config_sha256", None)
    # This field locates the input for local diagnostics; it is not experiment semantics.
    payload.pop("source_config", None)
    payload.pop("output", None)
    if "campaign_context" in payload:
        payload["campaign_context"].pop("run_root", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_config(input_path: Path, manifest_path: Path | None = None) -> tuple[dict[str, Any], str]:
    """Resolve a per-game YAML and return the canonical config plus SHA-256."""
    config_path = input_path.resolve()
    raw = load_game_config(config_path)
    suite = _mapping(raw.get("suite"), "suite")
    if manifest_path is None:
        manifest_reference = suite.get("manifest")
        if not isinstance(manifest_reference, str) or not manifest_reference:
            raise ConfigError(f"{config_path}: suite.manifest is required")
        manifest_path = _resolve_reference(manifest_reference, config_path.parent)
    _validate_manifest_membership(raw, config_path, manifest_path.resolve())
    resolved = _validate_and_resolve(raw, config_path)
    config_hash = _hash_config(resolved)
    resolved["resolved_config_sha256"] = config_hash
    return resolved, config_hash


def _game_yaml_path(source_manifest: Path, slug: str) -> Path:
    return source_manifest.parent / source_manifest.stem / "games" / f"{slug}.yaml"


def _campaign_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _parse_campaign_controls(raw: dict[str, Any], manifest_path: Path) -> CampaignControls:
    extra_fields = set(raw).difference(_PHASE1_MANIFEST_FIELDS)
    if extra_fields:
        names = ", ".join(sorted(extra_fields))
        raise ConfigError(f"{manifest_path}: Phase-1 manifest may contain only campaign controls; found {names}")

    campaign = _mapping(raw.get("campaign"), "campaign")
    extra_controls = set(campaign).difference(_CAMPAIGN_CONTROL_FIELDS)
    missing_controls = _CAMPAIGN_CONTROL_FIELDS.difference(campaign)
    if extra_controls or missing_controls:
        details: list[str] = []
        if extra_controls:
            details.append(f"unsupported {', '.join(sorted(extra_controls))}")
        if missing_controls:
            details.append(f"missing {', '.join(sorted(missing_controls))}")
        raise ConfigError(f"{manifest_path}: invalid campaign controls ({'; '.join(details)})")

    campaign_id = campaign["id"]
    run_root = campaign["run_root"]
    archive_root = campaign["archive_root"]
    if not all(isinstance(value, str) and value for value in (campaign_id, run_root, archive_root)):
        raise ConfigError(f"{manifest_path}: campaign id, run_root and archive_root must be non-empty strings")
    if campaign["scheduling"] != "sequential":
        raise ConfigError(f"{manifest_path}: campaign.scheduling must be sequential")
    if campaign["on_failure"] != "record_and_continue":
        raise ConfigError(f"{manifest_path}: campaign.on_failure must be record_and_continue")
    if campaign["automatic_resume"] is not False:
        raise ConfigError(f"{manifest_path}: campaign.automatic_resume must be false")
    if campaign["max_retries_per_game"] != 0:
        raise ConfigError(f"{manifest_path}: campaign.max_retries_per_game must be 0")
    if campaign["skip_completed"] is not True:
        raise ConfigError(f"{manifest_path}: campaign.skip_completed must be true")

    return CampaignControls(
        id=campaign_id,
        run_root=_campaign_output_root(run_root),
        archive_root=_campaign_output_root(archive_root),
        scheduling=campaign["scheduling"],
        on_failure=campaign["on_failure"],
        automatic_resume=campaign["automatic_resume"],
        max_retries_per_game=campaign["max_retries_per_game"],
        skip_completed=campaign["skip_completed"],
    )


def is_phase_manifest_with_campaign(path: Path) -> bool:
    """Return whether a manifest declares the strict Phase-1 campaign controls."""
    raw = _load_yaml(path.resolve())
    return "campaign" in raw and "game" not in raw


def resolve_phase1_campaign(path: Path) -> ResolvedCampaign:
    """Resolve a strict serial campaign without altering per-game semantics."""
    manifest_path = path.resolve()
    phase_raw = _load_yaml(manifest_path)
    controls = _parse_campaign_controls(phase_raw, manifest_path)
    phase_suite = _mapping(phase_raw.get("suite"), "suite")
    selected = load_manifest(manifest_path)

    source_reference = phase_suite.get("source_manifest")
    if not isinstance(source_reference, str) or not source_reference:
        raise ConfigError(f"{manifest_path}: suite.source_manifest is required")
    source_manifest = _resolve_reference(source_reference, manifest_path.parent)
    source_entries = {entry["slug"]: entry["env_id"] for entry in load_manifest(source_manifest)}

    games: list[tuple[dict[str, Any], str]] = []
    for selected_game in selected:
        slug = selected_game["slug"]
        if source_entries.get(slug) != selected_game["env_id"]:
            raise ConfigError(
                f"{slug} / {selected_game['env_id']} is not an exact source-manifest member"
            )
        game_path = _game_yaml_path(source_manifest, slug)
        if not game_path.is_file():
            raise ConfigError(f"missing independent game YAML for {slug}: {game_path}")
        resolved, game_config_hash = resolve_config(game_path, manifest_path=source_manifest)
        campaign_resolved = copy.deepcopy(resolved)
        campaign_resolved["output"]["run_root"] = str(controls.run_root / "games" / slug / "seed_000")
        campaign_resolved["output"]["archive_root"] = str(
            controls.archive_root / "games" / slug / "seed_000"
        )
        campaign_resolved["game_config_sha256"] = game_config_hash
        campaign_resolved["campaign_context"] = {
            "id": controls.id,
            "run_root": str(controls.run_root),
        }
        campaign_hash = _hash_config(campaign_resolved)
        campaign_resolved["resolved_config_sha256"] = campaign_hash
        games.append((campaign_resolved, campaign_hash))

    return ResolvedCampaign(
        manifest_path=manifest_path,
        controls=controls,
        games=games,
        manifest=copy.deepcopy(phase_raw),
    )


def resolve_campaign_configs(
    manifest_path: Path, *, game_slug: str | None = None
) -> list[tuple[dict[str, Any], str]]:
    """Resolve selected manifest games through their independent game YAMLs."""
    manifest_path = manifest_path.resolve()
    phase_raw = _load_yaml(manifest_path)
    phase_suite = _mapping(phase_raw.get("suite"), "suite")
    selected = load_manifest(manifest_path)
    if game_slug is not None:
        selected = [entry for entry in selected if entry["slug"] == game_slug]
        if not selected:
            raise ConfigError(f"game {game_slug!r} is not selected by {manifest_path}")

    source_reference = phase_suite.get("source_manifest")
    source_manifest = (
        _resolve_reference(source_reference, manifest_path.parent)
        if isinstance(source_reference, str)
        else manifest_path
    )
    source_entries = {entry["slug"]: entry["env_id"] for entry in load_manifest(source_manifest)}
    resolved_configs: list[tuple[dict[str, Any], str]] = []
    for selected_game in selected:
        game_path = _game_yaml_path(source_manifest, selected_game["slug"])
        if not game_path.is_file():
            raise ConfigError(
                f"missing independent game YAML for {selected_game['slug']}: {game_path}"
            )
        if source_entries.get(selected_game["slug"]) != selected_game["env_id"]:
            raise ConfigError(
                f"{selected_game['slug']} / {selected_game['env_id']} is not an exact source-manifest member"
            )
        resolved, config_hash = resolve_config(game_path, manifest_path=source_manifest)
        resolved_configs.append((resolved, config_hash))
    return resolved_configs
