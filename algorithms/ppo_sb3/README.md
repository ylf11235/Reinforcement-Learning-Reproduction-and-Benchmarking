# PPO_SB3 — Stable-Baselines3 PPO on Atari-16

Atari-tuned PPO ([Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
`PPO("CnnPolicy")`) for the [Atari-16 suite](../../docs/envs/atari-16.md):
8 vectorized envs, GAE, minibatch updates, and the unified Atari contract
(`ALE/<Game>-v5`, noop 30, frame skip 4, grayscale 84×84, stack 4, sticky off).

This backend produced the repository's main Atari-16 campaign
(`ppo_sb3_atari16_10m_seed0_v1`, 16/16 games, 10M steps each, seed 0) —
see the [experiment report](../../docs/reports/ppo-experiments.md) and
[results](../../results/ppo_sb3/).

## Layout

- `train.py` — single-game training entry point (arg-driven)
- `experiment.py` — campaign runner: validate / run / campaign subcommands
- `campaign.py`, `archive.py`, `provenance.py`, `callbacks.py`,
  `evaluation.py`, `config.py` — campaign orchestration,
  immutable archiving, source/ROM provenance, periodic evaluation
- shared environment construction, episode recording, JSONL logging, and
  atomic-write/hash utilities live in the repository-level
  [baselines_common/](../../baselines_common/) package
- `configs/atari16.yaml` — the 16-game suite manifest
- `configs/atari16_campaign.yaml` — the published campaign definition
- `configs/atari16/games/<slug>.yaml` — per-game configs (field-identical
  except slug / env_id / output paths)
- `configs/atari16/game_choice.md` — per-game selection rationale and
  reference-score provenance

## Usage

From the repository root (install a PyTorch **CUDA build matching your
GPU/driver** first — see [RUNBOOK.md](RUNBOOK.md#0-prerequisites); other
dependencies in `requirements.txt`; CUDA required —
silent CPU fallback is disabled by `require_device: true`). Full walkthrough
with timings, resume/retry, and expected outcomes: **[RUNBOOK.md](RUNBOOK.md)**.

```bash
# Validate the campaign config without training
python algorithms/ppo_sb3/experiment.py validate \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml

# Train one game (e.g. alien)
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16/games/alien.yaml

# Full sequential campaign — 'run' with the campaign config schedules all
# 16 games (≈56 GPU-hours total); completed games are skipped on re-invocation
python algorithms/ppo_sb3/experiment.py run \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml
```

Runs and archives are written under `algorithms/ppo_sb3/runs/` and
`algorithms/ppo_sb3/archives/` (git-ignored). Summary artifacts of the
published campaign live in `results/ppo_sb3/`.

## Unified configuration (all 16 games)

| Category | Value |
| --- | --- |
| Algorithm | SB3 `PPO("CnnPolicy")` |
| Budget | 10,000,000 requested SB3 timesteps; rollout-aligned effective 10,000,384 (128 × 8) |
| Seed | 0 (single seed) |
| Vectorization | `DummyVecEnv`, 8 workers, serial |
| PPO hyperparameters | `n_steps=128`, `batch_size=256`, `n_epochs=4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`, `lr=2.5e-4` (constant), `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`, advantage normalization on |
| Rewards | training `sign(reward)` clipping; evaluation unclipped (raw is primary) |

## Run protocol (condensed)

1. **Primary score.** The headline number is always the **final** checkpoint's
   mean unclipped raw return over 30 deterministic episodes on fixed audit
   seeds 20000–20029. Best-checkpoint scores are used only for model selection
   and videos, never as the primary score — "the 10M score" must not benefit
   from picking the best of ~38 intermediate checkpoints.
2. **Periodic evaluation.** Every 262,144 timesteps, 10 episodes on seeds
   10000–10009, deterministic, raw return. Drives best-checkpoint selection
   (tiebreak: mean → median → std → earlier step) and milestone retention.
3. **Checkpoint retention.** `last` (atomic, per rollout, for resume),
   `periodic` (every 262,144, keep 3), `milestones` (every 1,048,576),
   `best` (copy, not symlink), `final` (immutable). All model writes are
   atomic; metadata records step, episode scores, config and model SHA256s.
4. **Provenance.** Every run freezes pip freeze, hardware info, ROM SHA256
   (via ALE-Py), and a source snapshot of `algorithms/ppo_sb3/` +
   `baselines_common/` before training starts; versions are pinned and
   verified (`gymnasium 1.2.2`, `ale-py 0.12.1`, `stable-baselines3 2.8.0`,
   `torch 2.11.0`). CUDA is mandatory.
5. **Videos.** Best checkpoint, seeds 30000–30002, deterministic, native ALE
   RGB at 15 FPS (H.264). Qualitative only — never part of scoring.
6. **Failure policy.** Campaign continues after per-game failure
   (record-and-continue, up to 2 retries); completed games are skipped on
   resume via `run_state.json` compatibility checks.
7. **Scoring files.** `summary/scores.csv|json|md` carry per-game
   final/best raw stats, checkpoint hashes, wall time, and FPS; the campaign
   report aggregates totals and lists every failure explicitly.

Random seeds: periodic 10000–10009, audit 20000–20029, video 30000–30002 —
fixed by protocol, never sampled.
