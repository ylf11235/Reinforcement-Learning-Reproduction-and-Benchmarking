# Runbook: PPO_SB3 on Atari-16

How to reproduce the `ppo_sb3_atari16_10m_seed0_v1` campaign (16 games 脳
10M steps, seed 0) 鈥?the run behind the
[Atari-16 leaderboard](../../README.md#leaderboard-atari-16-human-normalized-score).
Protocol details: [README](README.md) 路
[experiment report](../../docs/reports/ppo-experiments.md).

## 0. Prerequisites

- Python 3.10+, CUDA GPU (training fails fast without CUDA 鈥?silent CPU
  fallback is disabled by design)
- ROMs available through ALE-Py (`ale-py==0.12.1` ships them)

```bash
# from the repository root
# 1) PyTorch CUDA build matching YOUR GPU/driver 鈥?not in requirements.txt.
#    Example (cu130); pick yours at https://pytorch.org/get-started/locally/ :
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# 2) everything else:
pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available()"
```

## 1. Validate (no training)

Checks all 16 per-game configs resolve, versions pin, and paths are writable.

```bash
python algorithms/ppo_sb3/experiment.py validate \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml

# validate a single game only
python algorithms/ppo_sb3/experiment.py validate \
  --config algorithms/ppo_sb3/configs/atari16/games/alien.yaml
```

## 2. Run one game

```bash
# fresh run (鈮?鈥? GPU-hours per game; pitfall/solaris up to 鈮?)
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16/games/alien.yaml

# resume an interrupted run from the last atomic checkpoint
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16/games/alien.yaml --resume
```

Outputs land in
`algorithms/ppo_sb3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1/games/<slug>/seed_000/`
(train/eval JSONL logs, TensorBoard, checkpoints, best/final, videos).

## 3. Run the full campaign

`run` with a campaign config schedules all 16 games sequentially:
completed games are skipped automatically, failed games are recorded and
the campaign continues (the phase-1 campaign config allows no automatic retries).

```bash
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml

# re-invoke after interruption (skips completed games)
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml

# retry only the failed games
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml \
  --retry-failed
```

Total wall-clock 鈮?56 h on one RTX-class GPU (see
`results/ppo_sb3/.../summary/throughput.csv` for per-game FPS).

## 4. Inspect and summarize

```bash
# per-game status of a campaign run directory
python algorithms/ppo_sb3/experiment.py status \
  --run-dir algorithms/ppo_sb3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1

# rebuild summary/scores.* from the run state
python algorithms/ppo_sb3/experiment.py summarize \
  --run-dir algorithms/ppo_sb3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1
```

## 5. TensorBoard

```bash
tensorboard --logdir algorithms/ppo_sb3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1/games
```

(The published campaign's curves are also committed under
`results/ppo_sb3/ppo_sb3_atari16_10m_seed0_v1/games/`.)

## 6. Expected outcome

Per-game final scores should match
`results/ppo_sb3/ppo_sb3_atari16_10m_seed0_v1/summary/scores.csv` within
seed-noise (same seed 0 + deterministic audit 鈬?near-identical on identical
hardware/versions; different GPUs may drift slightly due to nondeterministic
CUDA kernels despite `torch_deterministic: true`).
