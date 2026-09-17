"""Configuration loading, validation, and semantic fingerprinting for MRQ.

Implements the frozen MRQ experiment contract (2026-09-14; see
README.md):
- two house campaign manifests (atari16_2m5 / atari16_10m) + regression;
- strict schema validation (unknown keys rejected, ranges checked);
- budget identity: raw_frames == 4 * steps_requested, variant-consistent;
- semantic fingerprint: sha256 over the resolved config minus output paths,
  source paths, and other machine-local identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

# Fixed suite order (atari-16.md). Never reorder.
GAME_SLUGS: tuple = (
    "alien", "asteroids", "bowling", "chopper_command", "enduro", "frostbite",
    "gopher", "kung_fu_master", "montezuma_revenge", "pitfall", "private_eye",
    "seaquest", "skiing", "solaris", "tennis", "venture",
)
REGRESSION_SLUGS: tuple = ("alien", "frostbite", "seaquest")
FIRE_GAMES = frozenset({"bowling", "tennis"})

VARIANTS = ("house", "regression", "dmc", "car_racing")
DMC_SLUG_TO_ENV: dict = {
    "humanoid_walk": "Dmc-humanoid-walk",
}
CAR_RACING_SLUG_TO_ENV: dict = {
    "car_racing": "CarRacing-v3",
}
CAMPAIGNS = {
    "atari16_2m5": {"variant": "house", "steps": 2_500_000},
    "atari16_10m": {"variant": "house", "steps": 10_000_000},
    # house protocol at reduced budget (1M) — single-game diagnostics.
    # sticky-off/noop-30 follow the house contract; training rewards are RAW
    # per the MRQ paper (the algorithm's own reward-scale mechanism replaces
    # sign-clipping) — recorded as a documented deviation from atari-16.md.
    "atari16_1m": {"variant": "house", "steps": 1_000_000},
    "regression": {"variant": "regression", "steps": 2_500_000},
    # budget-reduced regression diagnostics (1M) — parity gate at reduced
    # horizon; never mixed into the 2.5M regression evidence
    "regression_1m": {"variant": "regression", "steps": 1_000_000},
    # pilot campaigns: budget-frozen diagnostic runs (plan Task 8 Step 3);
    # never mixed into leaderboard or regression evidence
    "pilot_100k": {"variant": "house", "steps": 100_000},
    "pilot_250k": {"variant": "house", "steps": 250_000},
    # DMC continuous-control campaigns (env: Dmc-<domain>-<task>, vector
    # obs, Box actions, dense raw rewards — paper protocol)
    "dmc_humanoid_walk_500k": {"variant": "dmc", "steps": 500_000},
    # CarRacing continuous-control campaigns (pixel obs + Box actions per
    # docs/envs/car_racing.md): 600k agent steps = 1.2M env frames (action repeat 2)
    "car_racing_600k": {"variant": "car_racing", "steps": 600_000},
    # budget-reduced diagnostic pilot; never mixed into 600k evidence
    "car_racing_pilot_100k": {"variant": "car_racing", "steps": 100_000},
}

# Reference-faithful algorithm constants (PROTOCOL.md §4; plan Global
# Constraints). These are the *defaults*; every key is overridable per file
# and the resolved values are what count.
ALGORITHM_DEFAULTS: dict = {
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "discount": 0.99,
    "target_update_freq": 250,        # hard update + encoder burst cadence
    "encoder_burst_steps": 250,
    "buffer_size_before_training": 10_000,  # random-action warmup
    "exploration_noise": 0.2,         # continuous; halved for discrete
    "exploration_noise_discrete": 0.1,
    "target_policy_noise": 0.2,       # continuous smoothing
    "target_policy_noise_clip": 0.3,
    "target_policy_noise_discrete": 0.1,
    "target_policy_noise_clip_discrete": 0.15,
    "encoder_loss_weight_dyn": 1.0,
    "encoder_loss_weight_reward": 0.1,
    "encoder_loss_weight_done": 0.1,
    "priority_alpha": 0.4,
    "min_priority": 1.0,
    "enc_horizon": 5,
    "q_horizon": 3,
    "zs_dim": 512,
    "zsa_dim": 512,
    "za_dim": 256,
    "hdim": 512,
    "encoder_lr": 1e-4,
    "value_lr": 3e-4,
    "policy_lr": 3e-4,
    "weight_decay": 1e-4,
    "value_grad_clip": 20.0,
    "gumbel_tau": 10.0,
    "pre_activation_penalty": 1e-5,
    "twohot_num_bins": 65,
    "twohot_lower": -10.0,
    "twohot_upper": 10.0,
    "shift_augment_pad": 4,
}

ENVIRONMENT_DEFAULTS: dict = {
    "frame_skip": 4,                  # wrapper action-repeat (ALE frameskip 1)
    "screen_size": 84,
    "stack": 4,
    "max_num_frames_per_episode": 108_000,  # raw frames
}

EVALUATION_DEFAULTS: dict = {
    "periodic_freq": 100_000,
    "periodic_episodes": 10,
    "periodic_seed_base": 10_000,
    "audit_episodes": 30,
    "audit_seed_base": 20_000,
    "eval_eps_random": 1e-3,          # regression only
}

CHECKPOINT_DEFAULTS: dict = {
    "model_after_eval": True,
    "durable_every": 500_000,
}

RUNTIME_DEFAULTS: dict = {
    "device": "auto",
    "require_device": False,
    "seed": 0,
    "deterministic": True,
    "cudnn_benchmark": True,
}

OUTPUT_DEFAULTS: dict = {
    # repo-relative; resolved relative to this file's directory at load time
    "run_root": "runs",
}

# ---------------------------------------------------------------------------
# Schema: every allowed key at every level. Unknown keys are ConfigError.
# ---------------------------------------------------------------------------

_TOP_LEVEL = {
    "schema_version", "experiment", "suite", "game", "budget", "runtime",
    "environment", "algorithm", "evaluation", "checkpoint", "output",
    "provenance",
}
_EXPERIMENT = {"id", "algorithm", "variant", "campaign"}
_SUITE = {"id", "games", "expected_game_count"}
_GAME = {"slug", "env_id", "allow_custom_env_id", "allow_reduced_buffer"}
_BUDGET = {"unit", "steps_requested", "raw_frames"}
_RUNTIME = set(RUNTIME_DEFAULTS)
_ENVIRONMENT = set(ENVIRONMENT_DEFAULTS) | {
    "noop_max", "clip_reward", "repeat_action_probability", "start_protocol",
}
_ALGORITHM = set(ALGORITHM_DEFAULTS)
_EVALUATION = set(EVALUATION_DEFAULTS)
_CHECKPOINT = set(CHECKPOINT_DEFAULTS)
_OUTPUT = set(OUTPUT_DEFAULTS)
_PROVENANCE = {"record_pip_freeze", "record_rom_hash"}

_SLUG_TO_ENV: dict = {
    slug: "ALE/" + "".join(p.capitalize() for p in slug.split("_")) + "-v5"
    for slug in GAME_SLUGS
}


class ConfigError(Exception):
    """Raised on any invalid, unknown, or inconsistent config value."""


def _fail(field: str, message: str) -> None:
    raise ConfigError(f"{field}: {message}")


def _check_keys(name: str, data: dict, allowed: set) -> None:
    if not isinstance(data, dict):
        _fail(name, "must be a mapping")
    unknown = set(data) - allowed
    if unknown:
        _fail(name, f"unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


def _slug_to_env_id(slug: str) -> str:
    return "ALE/" + "".join(p.capitalize() for p in slug.split("_")) + "-v5"


def load_config(path: str | Path) -> dict:
    """Load and fully resolve a config file (game or suite manifest).

    Resolution rules:
    - per-game YAML merges suite defaults (``<campaign>/<campaign>.yaml``)
      under explicit per-game keys;
    - algorithm/environment/evaluation/checkpoint/runtime defaults fill in
      anything unsaid;
    - paths under ``output`` resolve relative to this file
      (algorithms/mrq/), so validation is cwd-independent.
    """
    path = Path(path).resolve()
    if not path.exists():
        _fail("path", f"config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(path.name, "top level must be a mapping")
    _check_keys("config", raw, _TOP_LEVEL)

    campaign_id = raw.get("experiment", {}).get("campaign")
    if campaign_id not in CAMPAIGNS:
        _fail("experiment.campaign",
              f"must be one of {sorted(CAMPAIGNS)} (got {campaign_id!r})")
    spec = CAMPAIGNS[campaign_id]

    # suite-level defaults from configs/<campaign>.yaml (one level above the
    # per-game directory); located relative to this file, not the cwd
    configs_dir = Path(__file__).resolve().parent / "configs"
    suite_file = configs_dir / f"{campaign_id}.yaml"
    merged: dict = {}
    if suite_file != path and suite_file.exists():
        suite_raw = yaml.safe_load(suite_file.read_text(encoding="utf-8"))
        if isinstance(suite_raw, dict):
            _check_keys(f"{suite_file.name}", suite_raw, _TOP_LEVEL)
            merged.update(suite_raw)
    for key, val in raw.items():
        if key in ("game",):
            continue
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    if "game" in raw:
        merged["game"] = dict(raw["game"])

    # ---- experiment ----
    exp = merged.get("experiment", {})
    _check_keys("experiment", exp, _EXPERIMENT)
    if exp.get("algorithm") != "mrq":
        _fail("experiment.algorithm", "must be 'mrq'")
    variant = exp.get("variant", spec["variant"])
    if variant != spec["variant"]:
        _fail("experiment.variant",
              f"campaign {campaign_id!r} requires variant {spec['variant']!r}")
    if variant not in VARIANTS:
        _fail("experiment.variant", f"must be one of {VARIANTS}")

    # ---- suite ----
    suite = merged.get("suite", {})
    _check_keys("suite", suite, _SUITE)
    games = tuple(suite.get("games", ()))
    expected = suite.get("expected_game_count", len(games))
    if not games:
        _fail("suite.games", "must not be empty")
    env_slugs = (DMC_SLUG_TO_ENV if variant == "dmc"
                 else CAR_RACING_SLUG_TO_ENV if variant == "car_racing"
                 else None)
    if env_slugs is not None:
        bad = [g for g in games if g not in env_slugs]
        if bad:
            _fail("suite.games", f"unknown {variant} slug(s): {bad} "
                  f"(known: {sorted(env_slugs)})")
    else:
        if any(g not in GAME_SLUGS for g in games):
            bad = [g for g in games if g not in GAME_SLUGS]
            _fail("suite.games", f"unknown game slug(s): {bad}")
    if len(set(games)) != len(games):
        _fail("suite.games", "duplicate slugs")
    if variant == "house":
        order = {g: i for i, g in enumerate(GAME_SLUGS)}
        idx = [order[g] for g in games]
        if idx != sorted(idx):
            _fail("suite.games", "house games must follow the fixed suite order")
        if expected != len(games):
            _fail("suite.expected_game_count",
                  f"must equal len(games) = {len(games)}")
    elif variant == "regression":
        allowed = set(REGRESSION_SLUGS)
        if any(g not in allowed for g in games):
            _fail("suite.games",
                  f"regression suite allows only {sorted(allowed)}")
        order = {g: i for i, g in enumerate(REGRESSION_SLUGS)}
        if [order[g] for g in games] != sorted(order[g] for g in games):
            _fail("suite.games", "regression games must keep the fixed order")
    elif variant in ("dmc", "car_racing"):
        if expected != len(games):
            _fail("suite.expected_game_count",
                  f"must equal len(games) = {len(games)}")

    # ---- game ----
    game = merged.get("game", {})
    _check_keys("game", game, _GAME)
    slug = game.get("slug")
    is_game_file = "game" in raw
    if is_game_file:
        if variant in ("dmc", "car_racing"):
            slug_map = (DMC_SLUG_TO_ENV if variant == "dmc"
                        else CAR_RACING_SLUG_TO_ENV)
            if slug not in slug_map:
                _fail("game.slug", f"unknown {variant} slug {slug!r}")
            if slug not in games:
                _fail("game.slug",
                      f"{slug!r} is not a member of suite {campaign_id!r}")
            want_env = slug_map[slug]
            if game.get("allow_custom_env_id"):
                # explicit test hook: env_id may point at a registered fake
                # env; recorded verbatim in the resolved config
                if "env_id" not in game:
                    _fail("game.env_id", "allow_custom_env_id requires env_id")
            elif "env_id" in game and game["env_id"] != want_env:
                _fail("game.env_id", f"slug {slug!r} maps to {want_env!r}")
            game.setdefault("env_id", want_env)
        else:
            if slug not in GAME_SLUGS:
                _fail("game.slug", f"unknown slug {slug!r}")
            if slug not in games:
                _fail("game.slug", f"{slug!r} is not a member of suite {campaign_id!r}")
            want_env = _SLUG_TO_ENV[slug]
            if game.get("allow_custom_env_id"):
                # explicit test hook: env_id may point at a registered fake env;
                # recorded verbatim in the resolved config (visible identity)
                if "env_id" not in game:
                    _fail("game.env_id", "allow_custom_env_id requires env_id")
            elif "env_id" in game and game["env_id"] != want_env:
                _fail("game.env_id", f"slug {slug!r} maps to {want_env!r}")
            game.setdefault("env_id", want_env)
    else:
        if slug is not None:
            _fail("game.slug", "only per-game files may set game.slug")

    # ---- budget ----
    budget = merged.get("budget", {})
    _check_keys("budget", budget, _BUDGET)
    steps = budget.get("steps_requested")
    if steps is None:
        steps = spec["steps"]
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        _fail("budget.steps_requested", "must be a positive integer")
    if steps != spec["steps"]:
        _fail("budget.steps_requested",
              f"campaign {campaign_id!r} is frozen at {spec['steps']} steps "
              f"(got {steps}); a different budget needs its own campaign")
    if budget.get("unit", "agent_steps") != "agent_steps":
        _fail("budget.unit", "must be 'agent_steps'")
    # conversion: Atari 1 agent step = 4 raw frames (frame-skip 4);
    # DMC 1 agent step = 1 control step = 1 env transition (physics
    # substeps are internal simulator accounting, not budget units)
    frame_mult = {"dmc": 1, "car_racing": 2}.get(variant, 4)
    raw_frames = budget.get("raw_frames")
    if raw_frames is not None and raw_frames != frame_mult * steps:
        _fail("budget.raw_frames",
              f"must equal {frame_mult} * steps_requested = {frame_mult * steps}")
    budget = dict(budget)
    budget.update(unit="agent_steps", steps_requested=steps,
                  raw_frames=frame_mult * steps)

    # ---- runtime ----
    runtime = {**RUNTIME_DEFAULTS, **merged.get("runtime", {})}
    _check_keys("runtime", runtime, _RUNTIME)
    if not isinstance(runtime["seed"], int) or runtime["seed"] < 0:
        _fail("runtime.seed", "must be a non-negative integer")

    # ---- environment ----
    env = {**ENVIRONMENT_DEFAULTS, **merged.get("environment", {})}
    _check_keys("environment", env, _ENVIRONMENT)
    if variant == "dmc":
        # DMC contract: 1 env step = 1 control step (env transition), no
        # frame stack, dense raw rewards, no sticky/noop/fire — frozen per
        # the paper protocol.
        dmc_expect = {
            "frame_skip": 1, "screen_size": 84, "stack": 1,
            "noop_max": 0, "repeat_action_probability": 0.0,
            "clip_reward": False, "max_num_frames_per_episode": 1_000,
            "start_protocol": "none",
        }
        for key, want in dmc_expect.items():
            if env.get(key) != want:
                _fail(f"environment.{key}",
                      f"dmc variant requires {key}={want!r} (got {env.get(key)!r})")
        # DMC budget: 1 agent step = 1 env step (no frame multiplication);
        # the Atari-derived 4x raw_frames is corrected after budget resolution
    elif variant == "car_racing":
        car_expect = {
            "frame_skip": 2, "screen_size": 96, "stack": 4,
            "noop_max": 0, "repeat_action_probability": 0.0,
            "clip_reward": False, "max_num_frames_per_episode": 1_000,
            "start_protocol": "none",
        }
        for key, want in car_expect.items():
            if env.get(key) != want:
                _fail(f"environment.{key}",
                      f"car_racing variant requires {key}={want!r} "
                      f"(got {env.get(key)!r})")
    elif variant == "house":
        house_expect = {
            "repeat_action_probability": 0.0, "noop_max": 30,
        }
        for key, want in house_expect.items():
            if env.get(key) != want:
                _fail(f"environment.{key}",
                      f"house variant requires {key}={want!r} (got {env.get(key)!r})")
        # clip_reward: house default True, but algorithm-faithful backends may
        # declare raw training rewards as an explicit documented deviation
        # (e.g. MRQ's internal reward-scale mechanism) — must be a real bool
        if not isinstance(env.get("clip_reward"), bool):
            _fail("environment.clip_reward", "must be a boolean")
    else:
        reg_expect = {
            "repeat_action_probability": 0.25, "noop_max": 0,
            "clip_reward": False,
        }
        for key, want in reg_expect.items():
            if env.get(key) != want:
                _fail(f"environment.{key}",
                      f"regression variant requires {key}={want!r} (got {env.get(key)!r})")
    if variant not in ("dmc", "car_racing"):
        if env["frame_skip"] != 4 or env["screen_size"] != 84 or env["stack"] != 4:
            _fail("environment",
                  "frame_skip=4, screen_size=84, stack=4 are frozen by the house contract")
        if env["max_num_frames_per_episode"] != 108_000:
            _fail("environment.max_num_frames_per_episode", "frozen at 108000 raw frames")

    if is_game_file:
        if variant in ("dmc", "car_racing"):
            if env.get("start_protocol") != "none":
                _fail("environment.start_protocol",
                      "dmc games require start_protocol='none'")
        else:
            want_start = "fire" if slug in FIRE_GAMES else "none"
            if env.get("start_protocol") != want_start:
                _fail("environment.start_protocol",
                      f"game {slug!r} requires start_protocol={want_start!r}")
    elif env.get("start_protocol", "none") != "none":
        _fail("environment.start_protocol", "suite files must not override start_protocol")

    # ---- algorithm ----
    algo = {**ALGORITHM_DEFAULTS, **merged.get("algorithm", {})}
    _check_keys("algorithm", algo, _ALGORITHM)
    if (variant == "car_racing" and algo["buffer_size"] != 200_000
            and not game.get("allow_reduced_buffer")):
        # explicit game-file test hook: allow_reduced_buffer lets smoke/CLI
        # tests shrink the replay (recorded verbatim in the resolved config,
        # same visibility rule as allow_custom_env_id)
        _fail("algorithm.buffer_size",
              "car_racing variant freezes buffer_size=200000 "
              "(docs/envs/car_racing.md house deviation)")
    gamma = algo["discount"]
    if not (0.0 < gamma < 1.0):
        _fail("algorithm.discount", "must be in (0, 1)")
    for key in ("batch_size", "buffer_size", "enc_horizon", "q_horizon",
                "twohot_num_bins"):
        if not isinstance(algo[key], int) or algo[key] <= 0:
            _fail(f"algorithm.{key}", "must be a positive integer")
    for key in ("priority_alpha", "encoder_lr", "value_lr", "policy_lr"):
        if not isinstance(algo[key], (int, float)) or algo[key] < 0:
            _fail(f"algorithm.{key}", "must be non-negative")

    # ---- evaluation ----
    evaluation = {**EVALUATION_DEFAULTS, **merged.get("evaluation", {})}
    _check_keys("evaluation", evaluation, _EVALUATION)
    for key in ("periodic_freq", "periodic_episodes", "periodic_seed_base",
                "audit_episodes", "audit_seed_base"):
        if not isinstance(evaluation[key], int) or evaluation[key] <= 0:
            _fail(f"evaluation.{key}", "must be a positive integer")

    # ---- checkpoint ----
    checkpoint = {**CHECKPOINT_DEFAULTS, **merged.get("checkpoint", {})}
    _check_keys("checkpoint", checkpoint, _CHECKPOINT)
    for key in ("durable_every",):
        if not isinstance(checkpoint[key], int) or checkpoint[key] <= 0:
            _fail(f"checkpoint.{key}", "must be a positive integer")

    # ---- output (paths resolved relative to this file) ----
    output = {**OUTPUT_DEFAULTS, **merged.get("output", {})}
    _check_keys("output", output, _OUTPUT)
    if Path(output["run_root"]).is_absolute() and not is_game_file:
        pass  # suite files may pin an absolute run root only for local ops; discouraged
    output = dict(output)
    if not Path(output["run_root"]).is_absolute():
        here = Path(__file__).resolve().parent
        output["run_root"] = str((here / output["run_root"]).resolve())

    # ---- provenance ----
    prov_defaults = {"record_pip_freeze": True, "record_rom_hash": True}
    provenance = {**prov_defaults, **merged.get("provenance", {})}
    _check_keys("provenance", provenance, _PROVENANCE)

    resolved = {
        "schema_version": raw.get("schema_version", SCHEMA_VERSION),
        "experiment": {"id": exp.get("id", campaign_id),
                       "algorithm": "mrq",
                       "variant": variant,
                       "campaign": campaign_id},
        "suite": {"id": suite.get("id", campaign_id),
                  "games": list(games),
                  "expected_game_count": expected},
        "game": game if game else {},
        "budget": budget,
        "runtime": runtime,
        "environment": env,
        "algorithm": algo,
        "evaluation": evaluation,
        "checkpoint": checkpoint,
        "output": output,
        "provenance": provenance,
    }
    if resolved["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"must be {SCHEMA_VERSION}")
    return resolved


def config_fingerprint(resolved: dict) -> str:
    """Semantic fingerprint: sha256 over score-affecting identity.

    Excludes machine-local identity (output paths, absolute run roots).
    Includes campaign id, variant, budget, game, seed, environment toggles,
    algorithm constants, and evaluation/checkpoint policy.
    """
    if not isinstance(resolved, dict):
        _fail("config_fingerprint", "expects the resolved config dict")
    sem = copy.deepcopy(resolved)
    sem.pop("output", None)       # machine-local paths
    sem.pop("provenance", None)   # recording toggles, not score-affecting
    sem.pop("schema_version", None)
    blob = json.dumps(sem, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
