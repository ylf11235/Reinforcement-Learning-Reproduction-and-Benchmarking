# Runbook: PPO_PyTorch on Atari-10

How to run the CartPole-semantics PPO reference backend (single env,
full-batch, Monte-Carlo returns) on the Atari-10 suite. The completed
2-game campaign (`ppo_pytorch_atari2_10m_seed0_v2`, alien + frostbite) was
produced with these steps; the full 10-game campaign is ready but not yet
executed. Protocol details: [README](README.md) ·
[experiment report](../../docs/reports/ppo-experiments.md).

## 0. Prerequisites

- Python 3.10+, CUDA GPU
- Install a PyTorch **CUDA build matching your GPU/driver** first — it is
  deliberately not in `requirements.txt` (pick one at
  https://pytorch.org/get-started/locally/) — then
  `pip install -r requirements.txt` (from the repository root)

## 1. Validate (no training)

Enforces the upstream CartPole defaults (any deviation in the algorithm
section fails validation) and resolves all output paths.

```bash
python algorithms/ppo_pytorch/experiment.py validate \
  --config algorithms/ppo_pytorch/configs/atari10_campaign.yaml

# or the 2-game rerun campaign
python algorithms/ppo_pytorch/experiment.py validate \
  --config algorithms/ppo_pytorch/configs/atari2_campaign.yaml
```

## 2. Smoke test

Runs a handful of updates on one game to verify the stack end-to-end.

```bash
python algorithms/ppo_pytorch/experiment.py run \
  --config algorithms/ppo_pytorch/configs/games/alien.yaml --smoke
```

## 3. Run one game

≈7 GPU-hours per game at ~390–420 FPS (2.7–3× slower than PPO_SB3 due to
40-epoch full-batch updates and single-env collection).

```bash
python algorithms/ppo_pytorch/experiment.py run \
  --config algorithms/ppo_pytorch/configs/games/frostbite.yaml
```

The backend writes `last.pth` after every update, but this model-only file does
not contain optimizer, replay, or environment state. It is therefore not a
valid exact-resume checkpoint. An incomplete run is rejected on re-invocation;
use a new run root. A completed run is reused only when its config hash and
final checkpoint SHA256 both match.

Outputs land under
`algorithms/ppo_pytorch/runs/<campaign>/games/<slug>/seed_000/`
(`train_log.jsonl` with clipped + raw returns, TensorBoard scalars,
`checkpoints/{last,final}.pth`, `result.json`).

## 4. Run a campaign

```bash
# the completed 2-game rerun (alien, frostbite)
python algorithms/ppo_pytorch/experiment.py campaign \
  --config algorithms/ppo_pytorch/configs/atari2_campaign.yaml

# the full 10-game campaign (~70 GPU-hours total)
python algorithms/ppo_pytorch/experiment.py campaign \
  --config algorithms/ppo_pytorch/configs/atari10_campaign.yaml
```

Sequential scheduling, record-and-continue on failure, and completed games
skipped only after config/checkpoint verification.

## 5. Status

```bash
python algorithms/ppo_pytorch/experiment.py status \
  --run-dir algorithms/ppo_pytorch/runs/ppo10_v2
```

## 6. TensorBoard

```bash
tensorboard --logdir algorithms/ppo_pytorch/runs
```

(The published 2-game campaign's curves are committed under
`results/ppo_pytorch/ppo_pytorch_atari2_10m_seed0_v2/`.)

## 7. Expected outcome

alien ≈ 465, frostbite ≈ 199 final raw mean (30 deterministic audit
episodes, seeds 20000–20029). Note the evaluation step cap is
`max_ep_len=2000` — per-episode scores on other games may be truncated
if episodes run longer (see the report's cross-backend caveats).
