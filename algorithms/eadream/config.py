"""Strict, hashable configuration records for the formal EADream suite."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from yaml.constructor import ConstructorError


class ConfigError(ValueError):
    """Raised when a configuration is not an exact formal EADream input."""


@dataclass(frozen=True)
class GameSpec:
    slug: str
    env_id: str
    rom_sha256: str


_FORMAL_GAME_SPECS = (
    GameSpec("alien", "ALE/Alien-v5", "9cd556c7d7f55aeb78fc5add45d74ce889c6e22cba6bb535ef1e8bd6b6c764c1"),
    GameSpec("asteroids", "ALE/Asteroids-v5", "5ba6f91851b2331a37a3fe6a950c9b464fd764cb035a3d25d4b4388e9b26768f"),
    GameSpec("bowling", "ALE/Bowling-v5", "dff44a85289f5e9f760254e4806d09038326cdfc42eba3e1cd7c64ed6e0b4dce"),
    GameSpec("chopper_command", "ALE/ChopperCommand-v5", "055637282252b5378a2b7df573726e85afa91264ffb7243d23fab9518b3c207d"),
    GameSpec("enduro", "ALE/Enduro-v5", "6045c8be78c7d0bec29040022543a8c0b9e3672b50005a94bf0166f0f73be3d9"),
    GameSpec("frostbite", "ALE/Frostbite-v5", "cbfdad89480def69d922adf16a1d89d55a6cb515929edf74e5bea884c9fb7834"),
    GameSpec("gopher", "ALE/Gopher-v5", "b4aff03aeb0fb1c8f4914b5a329f51430f00a2d9350aba24573ad78032dd4697"),
    GameSpec("kung_fu_master", "ALE/KungFuMaster-v5", "3f6501a649ad83e970a25827bd492c56128c36535ae2c96f94bab39b27f939ac"),
    GameSpec("montezuma_revenge", "ALE/MontezumaRevenge-v5", "69d363a747549817599cace65572700dab9ea33f3a5f91979454fdfbeaa837c0"),
    GameSpec("pitfall", "ALE/Pitfall-v5", "c56c99d9e00136a015a485851e0e925c6327ea2b20aa7d3daecfbd7f9afcfdf0"),
    GameSpec("private_eye", "ALE/PrivateEye-v5", "c8f9ce1b2e804b6778dbc20da0c935b2af7d7e998ad372d44df66f16eb4b98ce"),
    GameSpec("seaquest", "ALE/Seaquest-v5", "fbc29f4678f69ac27fe46da298c8c22f98cba2a3e5491e6b308fd9bf68d2ee43"),
    GameSpec("skiing", "ALE/Skiing-v5", "e8a96a74c05493c5394a097deb7dc339a0cf37fe373d93e4c19f9e6884eff36c"),
    GameSpec("solaris", "ALE/Solaris-v5", "0afa36e5f4d8d77e38716673c96cda3ad11d421283f5a5ae77ec7a01e7718a4e"),
    GameSpec("tennis", "ALE/Tennis-v5", "b829e42751fcbc93ad7be8c6bd41e00083aad74d1f9f869f40898d21fb34133b"),
    GameSpec("venture", "ALE/Venture-v5", "5cdd323b8f817cc5d953910db97c20d9b7203316a8996663fdc953f2e29ee80e"),
)


@dataclass(frozen=True)
class CampaignControls:
    scheduling: Literal["serial"]
    on_failure: Literal["record_and_continue", "stop"]
    automatic_resume: bool
    skip_completed: bool
    max_retries_per_run: int


@dataclass(frozen=True)
class ResolvedRun:
    config: dict[str, Any]
    config_hash: str
    seed: int
    run_dir: Path


@dataclass(frozen=True)
class ResolvedCampaign:
    campaign_id: str
    runs: tuple[ResolvedRun, ...]
    run_root: Path
    archive_root: Path
    controls: CampaignControls


_ROOT = Path(__file__).resolve().parents[1]
_FORMAL_SEEDS = (0, 1, 2, 3, 4)
_FIRE_GAMES = frozenset({"bowling", "tennis"})
_FORMAL_GAMES = frozenset(spec.slug for spec in _FORMAL_GAME_SPECS)
_TOP_LEVEL = frozenset({
    "schema_version", "experiment", "suite", "game", "budget", "runtime",
    "environment", "events", "model", "training", "evaluation", "checkpoint",
    "video", "output", "campaign", "provenance",
})
_SECTION_KEYS = {
    "experiment": frozenset({"id", "algorithm", "formal", "result_kind"}),
    "suite": frozenset({"id", "manifest", "seeds"}),
    "game": frozenset({"slug", "env_id", "rom_sha256"}),
    "budget": frozenset({"unit", "agent_interactions", "ale_frames_nominal"}),
    "runtime": frozenset({"device", "require_cuda", "precision", "compile", "num_envs", "python_version", "torch_version", "torch_cuda_version", "deterministic_algorithms"}),
    "environment": frozenset({"frameskip", "action_repeat", "screen_size", "grayscale", "frame_stack", "noop_max_exclusive", "repeat_action_probability", "terminal_on_life_loss", "minimal_action_set", "max_num_frames_per_episode", "max_agent_steps_per_episode", "reward_clip", "max_pool_semantics", "start_protocol"}),
    "events": frozenset({"history", "var_threshold", "detect_shadows", "learning_rate", "closing_kernel", "closing_shape", "event_pred_ratio"}),
    "model": frozenset({"dyn_stoch", "dyn_discrete", "dyn_deter", "dyn_hidden", "dyn_rec_depth", "units", "activation", "normalization", "layer_norm_eps", "unimix_ratio", "initial", "cnn_depth", "cnn_kernel", "cnn_min_resolution", "encoder_embed_dim", "image_input_scale", "encoder_input_offset", "reconstruction_target_range", "encoder_attention", "encoder_attention_kernel", "encoder_attention_ratio", "decoder_attention", "event_decoder_attention", "predict_from_prior", "first_frame_prediction", "previous_current_event_input", "mae_ratio", "multihead_rssm_input", "dense_event_filter", "harmony", "kl_free", "dyn_scale", "rep_scale", "event_loss_scale", "event_focal_alpha", "event_focal_gamma", "image_attention_weight", "head_layers", "actor_distribution", "scalar_distribution", "scalar_buckets", "scalar_low", "scalar_high", "decoder_output_offset", "event_output_sigmoid", "reward_outscale", "continuation_outscale", "actor_outscale", "value_outscale"}),
    "training": frozenset({"prefill", "pretrain_batches", "batch_size", "batch_length", "train_ratio", "imag_horizon", "discount", "lambda", "dataset_size", "sequence_sampler_seed", "reward_ema", "reward_ema_alpha", "imag_gradient", "imag_gradient_mix", "exploration_behavior", "exploration_until", "detect_anomaly_world_model", "actor_entropy", "slow_value_update", "slow_value_fraction", "optimizer", "optimizer_effective_eps", "optimizer_effective_betas", "optimizer_effective_weight_decay", "model_lr", "model_grad_clip", "actor_lr", "actor_grad_clip", "value_lr", "value_grad_clip"}),
    "evaluation": frozenset({"deterministic", "formal_episodes", "formal_seed_source", "eval_state_mean", "rssm_sampling", "evaluator_rng", "audit_seeds", "no_fire_diagnostic_games", "no_fire_diagnostic_steps", "periodic_anchor_agent_interactions", "periodic_every_agent_interactions", "periodic_nominal_ale_frames"}),
    "checkpoint": frozenset({"save_every_agent_interactions", "first_periodic_agent_interactions", "keep_periodic", "validate_after_write", "immutable_milestones"}),
    "video": frozenset({"enabled", "checkpoint", "world_model_prediction", "episode_seeds", "source"}),
    "output": frozenset({"campaign_root", "archive_root", "archive_scope"}),
    "campaign": frozenset({"scheduling", "on_failure", "automatic_resume", "skip_completed", "max_retries_per_run"}),
    "provenance": frozenset({"reference_commit", "reference_manifest", "declared_actor_value_eps", "declared_extra_weight_decay"}),
}


def canonical_hash(config: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(config))
    payload.pop("resolved_config_sha256", None)
    payload.pop("source_config", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError("while constructing a mapping", node.start_mark, f"duplicate key {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if type(document) is not dict:
        raise ConfigError(f"configuration {path} must be a mapping")
    return document


def _exact_fields(value: Any, expected: frozenset[str], label: str, *, optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict:
        raise ConfigError(f"{label} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ConfigError(f"{label} keys must be strings")
    fields = set(value)
    unknown = fields - expected - optional
    missing = expected - fields
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown fields {sorted(unknown)}")
        if missing:
            details.append(f"missing fields {sorted(missing)}")
        raise ConfigError(f"{label} has " + "; ".join(details))
    return value


def _same_type_and_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_type_and_value(item, wanted) for item, wanted in zip(actual, expected)
        )
    return actual == expected


def _require(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        if not _same_type_and_value(value[key], wanted):
            raise ConfigError(f"{label}.{key} must be {wanted!r}, got {value[key]!r}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def load_manifest(path: Path) -> tuple[GameSpec, ...]:
    document = _read_yaml(path)
    _exact_fields(document, frozenset({"schema_version", "suite", "games"}), "manifest")
    if not _same_type_and_value(document["schema_version"], 1):
        raise ConfigError("manifest.schema_version must be 1")
    suite = _exact_fields(document["suite"], frozenset({"id", "expected_game_count"}), "manifest.suite")
    _require(suite, {"id": "atari16", "expected_game_count": 16}, "manifest.suite")
    games = document["games"]
    if type(games) is not list or len(games) != 16:
        raise ConfigError("manifest must contain exactly 16 ordered games")
    specs = []
    seen = set()
    for index, game in enumerate(games):
        record = _exact_fields(game, frozenset({"slug", "env_id", "rom_sha256"}), f"manifest.games[{index}]")
        slug, env_id, rom_sha256 = record["slug"], record["env_id"], record["rom_sha256"]
        if type(slug) is not str or slug not in _FORMAL_GAMES or slug in seen:
            raise ConfigError(f"manifest game {slug!r} is not a unique Atari-16 entry")
        if type(env_id) is not str or not env_id.startswith("ALE/") or not env_id.endswith("-v5"):
            raise ConfigError(f"manifest game {slug!r} has an invalid ALE environment")
        if not _is_sha256(rom_sha256):
            raise ConfigError(f"manifest game {slug!r} must use a lowercase SHA256")
        specs.append(GameSpec(slug=slug, env_id=env_id, rom_sha256=rom_sha256))
        seen.add(slug)
    parsed_specs = tuple(specs)
    if parsed_specs != _FORMAL_GAME_SPECS:
        raise ConfigError("manifest does not exactly match the fixed Atari-16 matrix")
    return parsed_specs


def _validate_formal_config(config: dict[str, Any]) -> None:
    _exact_fields(config, _TOP_LEVEL, "config", optional=frozenset({"source_config"}))
    if not _same_type_and_value(config["schema_version"], 1):
        raise ConfigError("schema_version must be 1")
    for section, fields in _SECTION_KEYS.items():
        _exact_fields(config[section], fields, section)
    _require(config["experiment"], {"id": "eadream_atari16_100k_v1", "algorithm": "eadream", "formal": True, "result_kind": "atari16_100k"}, "experiment")
    _require(config["suite"], {"id": "atari16", "manifest": "eadream/configs/atari16.yaml", "seeds": list(_FORMAL_SEEDS)}, "suite")
    _require(config["budget"], {"unit": "agent_interactions", "agent_interactions": 100_000, "ale_frames_nominal": 400_000}, "budget")
    _require(config["runtime"], {"device": "cuda:0", "require_cuda": True, "precision": "float32", "compile": False, "num_envs": 1, "python_version": "3.10.20", "torch_version": "2.11.0+cu130", "torch_cuda_version": "13.0", "deterministic_algorithms": False}, "runtime")
    _require(config["environment"], {"frameskip": 1, "action_repeat": 4, "screen_size": 64, "grayscale": False, "frame_stack": 1, "noop_max_exclusive": 30, "repeat_action_probability": 0.0, "terminal_on_life_loss": False, "minimal_action_set": True, "max_num_frames_per_episode": 108_000, "max_agent_steps_per_episode": 27_000, "reward_clip": False, "max_pool_semantics": "source_fixed_two_buffer"}, "environment")
    _require(config["events"], {"history": 500, "var_threshold": 16.0, "detect_shadows": True, "learning_rate": -1.0, "closing_kernel": 3, "closing_shape": "ellipse", "event_pred_ratio": 0.05}, "events")
    _require(config["model"], {"dyn_stoch": 64, "dyn_discrete": 32, "dyn_deter": 512, "dyn_hidden": 512, "dyn_rec_depth": 1, "units": 512, "activation": "SiLU", "normalization": "LayerNorm", "layer_norm_eps": 1e-3, "unimix_ratio": 0.01, "initial": "learned", "cnn_depth": 32, "cnn_kernel": 4, "cnn_min_resolution": 4, "encoder_embed_dim": 4096, "image_input_scale": "divide_255", "encoder_input_offset": -0.5, "reconstruction_target_range": "zero_one", "encoder_attention": True, "encoder_attention_kernel": 3, "encoder_attention_ratio": 2, "decoder_attention": False, "event_decoder_attention": False, "predict_from_prior": True, "first_frame_prediction": False, "previous_current_event_input": False, "mae_ratio": 1.0, "multihead_rssm_input": False, "dense_event_filter": True, "harmony": True, "kl_free": 1.0, "dyn_scale": 0.5, "rep_scale": 0.1, "event_loss_scale": 0.5, "event_focal_alpha": 0.15, "event_focal_gamma": 4.0, "image_attention_weight": 0.5, "head_layers": 2, "actor_distribution": "onehot", "scalar_distribution": "symlog_disc", "scalar_buckets": 255, "scalar_low": -20.0, "scalar_high": 20.0, "decoder_output_offset": 0.5, "event_output_sigmoid": True, "reward_outscale": 0.0, "continuation_outscale": 1.0, "actor_outscale": 1.0, "value_outscale": 0.0}, "model")
    _require(config["training"], {"prefill": 2500, "pretrain_batches": 100, "batch_size": 16, "batch_length": 64, "train_ratio": 1024, "imag_horizon": 15, "discount": 0.997, "lambda": 0.95, "dataset_size": 1_000_000, "sequence_sampler_seed": 0, "reward_ema": True, "reward_ema_alpha": 0.01, "imag_gradient": "reinforce", "imag_gradient_mix": 0.0, "exploration_behavior": "greedy", "exploration_until": 0, "detect_anomaly_world_model": True, "actor_entropy": 3e-4, "slow_value_update": 1, "slow_value_fraction": 0.02, "optimizer": "adamw", "optimizer_effective_eps": 1e-8, "optimizer_effective_betas": [0.9, 0.999], "optimizer_effective_weight_decay": 0.01, "model_lr": 1e-4, "model_grad_clip": 1000.0, "actor_lr": 3e-5, "actor_grad_clip": 100.0, "value_lr": 3e-5, "value_grad_clip": 100.0}, "training")
    _require(config["evaluation"], {"deterministic": True, "formal_episodes": 100, "formal_seed_source": "train_seed_fresh_env", "eval_state_mean": False, "rssm_sampling": True, "evaluator_rng": "source_seed_before_model_construction", "audit_seeds": list(range(20000, 20030)), "no_fire_diagnostic_games": ["bowling", "tennis"], "no_fire_diagnostic_steps": 200, "periodic_anchor_agent_interactions": 2500, "periodic_every_agent_interactions": 7500, "periodic_nominal_ale_frames": 30000}, "evaluation")
    _require(config["checkpoint"], {"save_every_agent_interactions": 7500, "first_periodic_agent_interactions": 10000, "keep_periodic": 2, "validate_after_write": True, "immutable_milestones": [100000]}, "checkpoint")
    _require(config["video"], {"enabled": True, "checkpoint": "best", "world_model_prediction": True, "episode_seeds": [30000, 30001, 30002], "source": "native_ale_rgb"}, "video")
    _require(config["output"], {"campaign_root": "eadream/runs/atari16/eadream_atari16_100k_v1", "archive_root": "eadream/archives/atari16/eadream_atari16_100k_v1", "archive_scope": "result"}, "output")
    _require(config["campaign"], {"scheduling": "serial", "on_failure": "record_and_continue", "automatic_resume": True, "skip_completed": True, "max_retries_per_run": 1}, "campaign")
    _require(config["provenance"], {"reference_commit": "269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241", "reference_manifest": "eadream/reference_manifest.json", "declared_actor_value_eps": 1e-5, "declared_extra_weight_decay": 0.0}, "provenance")
    game = config["game"]
    if game["slug"] not in _FORMAL_GAMES or not isinstance(game["env_id"], str) or not _is_sha256(game["rom_sha256"]):
        raise ConfigError("game must be a fixed Atari-16 entry with a pinned ROM SHA256")
    wanted_start = "fire" if game["slug"] in _FIRE_GAMES else "none"
    if not _same_type_and_value(config["environment"]["start_protocol"], wanted_start):
        raise ConfigError(f"{game['slug']} must use start_protocol {wanted_start!r}")


def resolve_config(path: Path, *, seed: int) -> tuple[dict[str, Any], str]:
    if type(seed) is not int or seed not in _FORMAL_SEEDS:
        raise ConfigError("formal seed must be in 0..4")
    config = _read_yaml(path)
    _validate_formal_config(config)
    manifest_specs = {spec.slug: spec for spec in load_manifest(_repository_path(config["suite"]["manifest"]))}
    game = config["game"]
    expected_game = manifest_specs.get(game["slug"])
    if expected_game is None or (game["env_id"], game["rom_sha256"]) != (expected_game.env_id, expected_game.rom_sha256):
        raise ConfigError(f"game configuration {game['slug']!r} does not match its manifest record")
    config = copy.deepcopy(config)
    config["game"]["seed"] = seed
    config["source_config"] = str(path)
    digest = canonical_hash(config)
    config["resolved_config_sha256"] = digest
    return config, digest


def _repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _ROOT / path


def resolve_campaign(path: Path, *, game_slug: str | None = None, seed: int | None = None) -> ResolvedCampaign:
    document = _read_yaml(path)
    expected = frozenset({"schema_version", "campaign_id", "manifest", "game_directory", "games", "seeds", "campaign", "output"})
    _exact_fields(document, expected, "campaign")
    _require(document, {"schema_version": 1, "campaign_id": "eadream_atari16_100k_v1", "manifest": "eadream/configs/atari16.yaml", "game_directory": "eadream/configs/atari16/games"}, "campaign")
    specs = load_manifest(_repository_path(document["manifest"]))
    slugs = [spec.slug for spec in specs]
    if not _same_type_and_value(document["games"], slugs) or not _same_type_and_value(document["seeds"], list(_FORMAL_SEEDS)):
        raise ConfigError("campaign selectors must preserve the complete manifest order and formal seeds")
    controls_raw = _exact_fields(document["campaign"], _SECTION_KEYS["campaign"], "campaign.campaign")
    _require(controls_raw, {"scheduling": "serial", "on_failure": "record_and_continue", "automatic_resume": True, "skip_completed": True, "max_retries_per_run": 1}, "campaign.campaign")
    controls = CampaignControls(**controls_raw)
    output = _exact_fields(document["output"], frozenset({"campaign_root", "archive_root"}), "campaign.output")
    _require(output, {"campaign_root": "eadream/runs/atari16/eadream_atari16_100k_v1", "archive_root": "eadream/archives/atari16/eadream_atari16_100k_v1"}, "campaign.output")
    if game_slug is not None and (type(game_slug) is not str or game_slug not in slugs):
        raise ConfigError(f"unknown campaign game selector {game_slug!r}")
    if seed is not None and (type(seed) is not int or seed not in _FORMAL_SEEDS):
        raise ConfigError("formal seed selector must be in 0..4")
    selected_slugs = [game_slug] if game_slug else slugs
    selected_seeds = [seed] if seed is not None else list(_FORMAL_SEEDS)
    game_dir = _repository_path(document["game_directory"])
    run_root = _repository_path(output["campaign_root"])
    archive_root = _repository_path(output["archive_root"])
    by_slug = {spec.slug: spec for spec in specs}
    runs = []
    hashes = set()
    run_dirs = set()
    for slug in selected_slugs:
        for train_seed in selected_seeds:
            config_path = game_dir / f"{slug}.yaml"
            config, config_hash = resolve_config(config_path, seed=train_seed)
            spec = by_slug[slug]
            if (config["game"]["env_id"], config["game"]["rom_sha256"]) != (spec.env_id, spec.rom_sha256):
                raise ConfigError(f"game configuration {slug!r} does not match its manifest record")
            run_dir = run_root / "games" / slug / f"seed_{train_seed:03d}"
            if config_hash in hashes or run_dir in run_dirs:
                raise ConfigError("campaign has a resolved configuration/output collision")
            hashes.add(config_hash)
            run_dirs.add(run_dir)
            runs.append(ResolvedRun(config=config, config_hash=config_hash, seed=train_seed, run_dir=run_dir))
    return ResolvedCampaign(document["campaign_id"], tuple(runs), run_root, archive_root, controls)


def installed_distribution_versions() -> dict[str, str | None]:
    names = ("numpy", "opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless", "gymnasium", "ale-py", "torch")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_formal_versions() -> dict[str, str]:
    versions = installed_distribution_versions()
    expected = {"numpy": "1.26.4", "gymnasium": "1.2.2", "ale-py": "0.12.1", "torch": "2.11.0+cu130"}
    for package, wanted in expected.items():
        if versions.get(package) != wanted:
            raise ConfigError(f"{package} must be {wanted}, got {versions.get(package)!r}")
    if versions.get("opencv-python-headless") != "4.7.0.72":
        raise ConfigError("opencv-python-headless must be 4.7.0.72")
    for alternate in ("opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"):
        if versions.get(alternate) is not None:
            raise ConfigError(f"{alternate} conflicts with the formal OpenCV pin")
    return {name: version for name, version in versions.items() if version is not None}
