"""Post-hoc formal + audit evaluation from a saved final checkpoint.

Loads the fast-variant run's final.pt, reconstructs the EADream policy, and
runs the formal 100-episode evaluation plus the 30-seed PPOSB3 audit using
the standard evaluation module. Results are written under evaluations/.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eadream.algorithm import EADream  # noqa: E402
from eadream.config import canonical_hash  # noqa: E402
from eadream.envs.atari import make_atari_env  # noqa: E402
from eadream.envs.wrappers import EventObservationWrapper  # noqa: E402
from eadream.evaluation import evaluate_audit, evaluate_formal, save_result  # noqa: E402
from eadream.scripts.run_game_fast import apply_compile_patch, build_fast_config  # noqa: E402
from eadream.experiment import _run_directory  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved_fast_run(run_dir: Path, game: str, seed: int, *, compiled: bool) -> tuple[dict, str, str]:
    import yaml

    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if config["game"]["slug"] != game or int(config["game"]["seed"]) != seed:
        raise ValueError("saved run game/seed do not match evaluator arguments")
    if config["experiment"]["id"] != "eadream_atari16_100k_fast" or canonical_hash(config) != config["resolved_config_sha256"]:
        raise ValueError("saved fast-run configuration is invalid")
    if config["runtime"]["compile"] and not compiled:
        raise ValueError("compiled checkpoint requires --compile")
    checkpoint = run_dir / "checkpoints" / "final.pt"
    metadata = json.loads((run_dir / "checkpoints" / "metadata.json").read_text(encoding="utf-8"))
    if not checkpoint.is_file() or sha256_of(checkpoint) != metadata["sha256"]:
        raise ValueError("final checkpoint SHA256 does not match metadata")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    config_hash = str(metadata["config_hash"])
    provenance_hash = str(metadata["provenance_hash"])
    if config_hash != config["resolved_config_sha256"]:
        historical = json.loads((run_dir / "evaluations" / "formal_100k.json").read_text(encoding="utf-8"))
        if historical["config_sha256"] != config_hash or provenance.get("source_provenance_sha256") != provenance_hash:
            raise ValueError("legacy checkpoint identities do not match source evidence")
    elif provenance["provenance_sha256"] != provenance_hash:
        raise ValueError("saved checkpoint provenance hash does not match")
    return config, config_hash, provenance_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="alien", help="game slug (default alien)")
    parser.add_argument("--run-dir", type=Path, default=None, help="override run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="checkpoint was saved with torch.compile")
    args = parser.parse_args()

    if args.run_dir is None:
        args.run_dir = _run_directory(build_fast_config(args.game, args.seed, compiled=args.compile))

    config, config_hash, provenance_hash = load_saved_fast_run(args.run_dir, args.game, args.seed, compiled=args.compile)

    if args.compile:
        apply_compile_patch()

    checkpoint_path = args.run_dir / "checkpoints" / "final.pt"
    # The driver composes the event wrapper at training time; evaluation envs
    # from make_atari_env are bare StartGuard stacks, so wrap them identically.
    env_factory = lambda: EventObservationWrapper(make_atari_env(config, mode="formal_eval"))

    # Construct a live agent bound to a training env shape, then restore.
    train_factory = lambda: make_atari_env(config, mode="train")
    agent = EADream(train_factory(), config, env_factory=train_factory)
    # run_one overrides these with the resolved-config identity; mirror it so
    # checkpoint metadata validation matches the training-time values.
    agent.config_hash = config_hash
    agent.provenance_hash = provenance_hash
    record = agent.load_checkpoint(checkpoint_path)
    print(f"restored checkpoint: {record.agent_interactions} interactions")

    policy = agent  # EADream exposes predict(observation, state=..., episode_start=..., deterministic=...)

    formal = evaluate_formal(
        policy,
        env_factory,
        train_seed=args.seed,
        episodes=int(config["evaluation"]["formal_episodes"]),
        config=config,
        checkpoint_hash=sha256_of(checkpoint_path),
        config_hash=config_hash,
    )
    formal_path = args.run_dir / "evaluations" / "formal_100k.json"
    save_result(formal_path, formal)

    audit = evaluate_audit(
        policy,
        env_factory,
        seeds=list(config["evaluation"]["audit_seeds"]),
        train_seed=args.seed,
        config=config,
    )
    audit_path = args.run_dir / "evaluations" / "pposb3_audit_100k.json"
    save_result(audit_path, audit)

    summary = {
        "formal_mean": formal.get("mean_return"),
        "formal_median": formal.get("median_return"),
        "formal_episodes": formal.get("episodes"),
        "audit_mean": audit.get("mean_return"),
        "audit_median": audit.get("median_return"),
        "audit_seeds": len(list(config["evaluation"]["audit_seeds"])),
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
