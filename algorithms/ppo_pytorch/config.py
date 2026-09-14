"""Independent YAML resolution for the PPO-PyTorch Atari-10 campaign."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
EXPECTED_GAMES = [
    "alien",
    "asteroids",
    "bowling",
    "chopper_command",
    "enduro",
    "frostbite",
    "gopher",
    "kung_fu_master",
    "seaquest",
    "tennis",
]
DEFAULT_ALGORITHM = {
    "max_ep_len": 2000,
    "update_timestep": 1600,
    "K_epochs": 40,
    "eps_clip": 0.2,
    "gamma": 0.99,
    "lr_actor": 0.0003,
    "lr_critic": 0.001,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
}


class ConfigError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"configuration must be a mapping: {path}")
    value["source_config"] = str(path)
    return value


def load_manifest(path: Path) -> list[dict[str, str]]:
    raw = _load(path)
    suite = raw.get("suite")
    games = raw.get("games")
    if not isinstance(suite, dict) or not isinstance(games, list):
        raise ConfigError("manifest requires suite and games")
    expected = int(suite.get("expected_game_count", len(games)))
    if expected != len(games):
        raise ConfigError(f"manifest expected {expected} games, found {len(games)}")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(games):
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str) or not isinstance(entry.get("env_id"), str):
            raise ConfigError(f"games[{index}] requires slug and env_id")
        if entry["slug"] in seen:
            raise ConfigError(f"duplicate game slug: {entry['slug']}")
        seen.add(entry["slug"])
        result.append({"slug": entry["slug"], "env_id": entry["env_id"]})
    return result


def load_game_config(path: Path) -> dict[str, Any]:
    return copy.deepcopy(_load(path))


def _hash(value: dict[str, Any]) -> str:
    semantic = copy.deepcopy(value)
    semantic.pop("source_config", None)
    semantic.pop("output", None)
    if "campaign" in semantic:
        semantic["campaign"].pop("run_root", None)
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _campaign_run_root(value: str) -> Path:
    run_root = Path(value)
    return run_root if run_root.is_absolute() else (ROOT.parent.parent / run_root).resolve()


def resolve_game(path: Path) -> dict[str, Any]:
    config = load_game_config(path)
    for section in ("experiment", "suite", "game", "budget", "runtime", "environment", "algorithm", "output"):
        if not isinstance(config.get(section), dict):
            raise ConfigError(f"{path}: missing mapping section {section}")
    game = config["game"]
    if not isinstance(game.get("slug"), str) or not game["env_id"].startswith("ALE/"):
        raise ConfigError(f"{path}: invalid game identity")
    if int(config["budget"].get("requested", 0)) != 10_000_000:
        raise ConfigError(f"{path}: budget.requested must be 10000000")
    config["budget"]["effective"] = int(config["budget"]["requested"])
    if config["algorithm"] != DEFAULT_ALGORITHM:
        raise ConfigError(f"{path}: algorithm must equal the CartPole discrete defaults")
    environment = config["environment"]
    if environment.get("frame_skip") != 4 or environment.get("screen_size") != 84 or environment.get("stack") != 4:
        raise ConfigError(f"{path}: Atari preprocessing must be frame_skip=4, screen_size=84, stack=4")
    config["resolved_config_sha256"] = _hash({key: value for key, value in config.items() if key != "source_config"})
    return config


def resolve_campaign(path: Path) -> list[dict[str, Any]]:
    campaign = _load(path)
    manifest = load_manifest(ROOT / "configs" / "atari10.yaml")
    expected = [entry["slug"] for entry in manifest]
    if expected != EXPECTED_GAMES:
        raise ConfigError("Atari-10 manifest does not match the approved skill-game order")
    listed = campaign.get("games")
    if not isinstance(listed, list) or not listed:
        raise ConfigError("campaign games must be a non-empty list")
    listed_slugs: list[str] = []
    manifest_by_slug = {entry["slug"]: entry for entry in manifest}
    for index, entry in enumerate(listed):
        if not isinstance(entry, dict) or not isinstance(entry.get("slug"), str):
            raise ConfigError(f"campaign games[{index}] requires slug")
        slug = entry["slug"]
        if slug in listed_slugs:
            raise ConfigError(f"duplicate campaign game slug: {slug}")
        if slug not in manifest_by_slug:
            raise ConfigError(f"campaign game is not in the Atari-10 manifest: {slug}")
        if entry.get("env_id") != manifest_by_slug[slug]["env_id"]:
            raise ConfigError(f"campaign manifest mismatch for {slug}")
        listed_slugs.append(slug)
    expected_count = campaign.get("campaign", {}).get("expected_game_count")
    if expected_count is not None and int(expected_count) != len(listed_slugs):
        raise ConfigError(
            f"campaign expected_game_count={expected_count} but lists {len(listed_slugs)} games"
        )
    run_root = _campaign_run_root(str(campaign["campaign"]["run_root"]))
    resolved: list[dict[str, Any]] = []
    for entry in listed:
        game_path = ROOT / "configs" / "games" / f"{entry['slug']}.yaml"
        config = resolve_game(game_path)
        if config["game"]["env_id"] != manifest_by_slug[entry["slug"]]["env_id"]:
            raise ConfigError(f"game manifest mismatch for {entry['slug']}")
        config["output"]["run_root"] = str(run_root / "games" / entry["slug"] / "seed_000")
        config["output"]["archive_root"] = str(run_root / "archives" / entry["slug"] / "seed_000")
        config["campaign"] = {"id": campaign["campaign"]["id"], "run_root": str(run_root)}
        config["resolved_config_sha256"] = _hash({key: value for key, value in config.items() if key != "source_config"})
        resolved.append(config)
    return resolved


def resolve_any(path: Path) -> list[dict[str, Any]]:
    raw = _load(path)
    return resolve_campaign(path) if "campaign" in raw and "game" not in raw else [resolve_game(path)]
