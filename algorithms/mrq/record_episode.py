"""Render one deterministic episode from a saved MRQ checkpoint to MP4.

Usage:
  python record_episode.py --run-dir <run_dir> --seed 20000 --out video.mp4
Loads checkpoints/final, rebuilds the agent from resolved_config.yaml, plays
one episode with deterministic actions, and writes the RGB frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

MRQ_DIR = Path(__file__).resolve().parent
if str(MRQ_DIR) not in sys.path:
    sys.path.insert(0, str(MRQ_DIR))
REPO_ROOT = MRQ_DIR.parents[1]  # repository root (baselines_common)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import env_adapter  # noqa: E402
from agent import MRQAgent  # noqa: E402
from config import config_fingerprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="eval seed (episode identity)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())

    ckpt = run_dir / "checkpoints" / "final"
    manifest = json.loads((ckpt / "manifest.json").read_text())
    fp = config_fingerprint(cfg)
    assert manifest["config_fingerprint"] == fp, \
        f"fingerprint mismatch: run={manifest['config_fingerprint']} cfg={fp}"

    device = torch.device("cpu")  # single-episode playback: CPU is enough
    from gymnasium import spaces
    env = env_adapter.build_eval_env(cfg, eval_seed=args.seed,
                                     render_mode="rgb_array")
    act_space = env.action_space
    if isinstance(act_space, spaces.Box):
        action_dim, discrete = int(act_space.shape[0]), False
    else:
        action_dim, discrete = int(act_space.n), True

    obs_shape = env.observation_space.shape
    frame_shape = ((1, obs_shape[1], obs_shape[2]) if len(obs_shape) == 3
                   else obs_shape)
    agent = MRQAgent(frame_shape, action_dim, discrete=discrete,
                     device=device, history=cfg["environment"]["stack"],
                     algorithm=cfg["algorithm"])
    agent.load_checkpoint(str(ckpt), durable=False)

    import imageio.v2 as imageio
    frames = []
    raw_return = 0.0
    steps = 0
    obs, _ = env.reset(seed=args.seed)
    done = False
    while not done:
        # render BEFORE stepping (the frame the agent acts on)
        frames.append(env.render())
        action = agent.select_action(obs, evaluate=True)
        if action is None:
            action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        raw_return += float(info.get("raw_reward", r))
        steps += 1
        done = term or trunc
        if steps % 500 == 0:
            print(f"  step {steps} return {raw_return:.0f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=30, quality=8)
    print(f"[record] {steps} steps, raw return {raw_return:.0f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
