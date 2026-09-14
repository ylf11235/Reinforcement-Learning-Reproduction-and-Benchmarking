"""Launch a single non-formal EADream run for any Atari-16 game.

Fast variant for local experimentation:
- world-model updates skip ``torch.autograd.set_detect_anomaly`` bookkeeping
- ``experiment.formal`` is flipped to False (diagnostic variant, never formal)
- outputs go to the separate ``eadream_atari16_100k_fast_v1`` campaign root

Base config comes from the per-game formal YAML in
``eadream/configs/atari16/games/<slug>.yaml``; only the deviations above are
applied. Optionally applies torch.compile (mirrors upstream default).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eadream.config import canonical_hash, resolve_config  # noqa: E402
from eadream.engine.agent import EADreamAgent  # noqa: E402
from eadream.experiment import run_one  # noqa: E402

GAMES = (
    "alien", "asteroids", "bowling", "chopper_command", "enduro",
    "frostbite", "gopher", "kung_fu_master", "montezuma_revenge",
    "pitfall", "private_eye", "seaquest", "skiing", "solaris",
    "tennis", "venture",
)


def apply_compile_patch() -> None:
    """Compile world-model/behavior modules at engine construction time.

    Mirrors upstream dreamer.py (torch.compile(self._wm) etc.). Checkpoints
    written by a compiled run carry ``_orig_mod`` key prefixes, so do not mix
    compiled and eager checkpoints for resume.
    """

    original_init = EADreamAgent.__init__

    def patched_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        import torch

        self.world_model = torch.compile(self.world_model)
        self.behavior = torch.compile(self.behavior)

    EADreamAgent.__init__ = patched_init


def build_fast_config(slug: str, seed: int, *, compiled: bool = False) -> dict:
    formal_yaml = _ROOT / "eadream" / "configs" / "atari16" / "games" / f"{slug}.yaml"
    config, _ = resolve_config(formal_yaml, seed=seed)
    config = copy.deepcopy(config)
    config["training"]["detect_anomaly_world_model"] = False
    config["experiment"]["formal"] = False
    config["experiment"]["id"] = "eadream_atari16_100k_fast"
    config["runtime"]["compile"] = bool(compiled)
    config["output"]["campaign_root"] = "eadream/runs/atari16/eadream_atari16_100k_fast_v1"
    config["output"]["archive_root"] = "eadream/archives/atari16/eadream_atari16_100k_fast_v1"
    config.pop("resolved_config_sha256", None)
    config.pop("source_config", None)
    digest = canonical_hash(config)
    config["resolved_config_sha256"] = digest
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=GAMES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile world model and behavior")
    args = parser.parse_args()
    if args.compile:
        apply_compile_patch()
    config = build_fast_config(args.game, args.seed, compiled=args.compile)
    record = run_one(config, input_config=copy.deepcopy(config), resume=bool(args.resume))
    import json

    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
