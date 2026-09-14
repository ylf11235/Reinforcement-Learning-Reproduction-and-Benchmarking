# EADream Runbook

Step-by-step instructions for reproducing the EADream Atari-16 runs from this
repository. All commands run from the repository root with a CUDA-capable
Python 3.10 environment (`pip install -r algorithms/eadream/requirements-formal.txt`
plus a CUDA torch build).

## 0. One-time setup

```bash
pip install -r algorithms/eadream/requirements-formal.txt
# PyTorch CUDA wheel is chosen per machine — pick the one matching YOUR
# GPU/driver (https://pytorch.org/get-started/locally/); cu130 example:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# ALE-Py 0.12.1 supplies the ROMs used by this pinned suite; preflight checks their SHA256.
```

## 1. Validate the configuration

```bash
python algorithms/eadream/experiment.py validate \
  --config algorithms/eadream/configs/alien_pilot_100k.yaml --seed 0
```

Prints the resolved config hash. The same hash is embedded in every artifact
of the run (run state, checkpoints, evaluations), so any later diff is
detectable.

## 2. Static + live preflight

```bash
# Gate 0-2: dependency pins, CUDA, manifest hashes, released-score fixtures
python algorithms/eadream/experiment.py preflight \
  --config algorithms/eadream/configs/atari16_campaign.yaml

# Gate 0-3: live ROM/render/event/inference checks for all 16 games
# (writes eadream/runs/preflight/preflight_summary.json)
```

Expected: `failed: 0` and all 16 games `PASSED`.

## 3A. Formal run (protocol-faithful)

```bash
python algorithms/eadream/experiment.py run \
  --config algorithms/eadream/configs/alien_pilot_100k.yaml --seed 0
```

- Cold start from random parameters; prefill 2,500 interactions; 100
  pretraining batches; then one gradient batch per interaction to 100K.
- Outputs: `algorithms/eadream/runs/atari16/eadream_atari16_100k_v1/games/<slug>/seed_<NNN>/`
  (`run_state.json`, `logs/metrics.jsonl`, `tensorboard/`, `checkpoints/final.pt`,
  durable `replay/`).
- The formal entry point currently saves a terminal checkpoint only; interrupted
  runs are marked `FAILED` and cannot be resumed exactly. Start a new output root.
- Expect ~2.3 s/batch (RTX 5080): **~60+ h per 100K run** — the formal
  protocol keeps upstream's `torch.autograd.set_detect_anomaly(True)`, whose
  stack-bookkeeping overhead dominates wall-clock (profiled: >50%).

## 3B. Fast run (published-variant used for the leaderboard row)

```bash
python algorithms/eadream/scripts/run_game_fast.py \
  --game frostbite --seed 0 --compile
```

Deviations from formal (recorded in the run's resolved config):
`training.detect_anomaly_world_model=false`, `experiment.formal=false`,
`runtime.compile` effectively on, separate `eadream_atari16_100k_fast_v1`
campaign root. Expect ~0.4 s/batch (~7–11 h per 100K run on an RTX 5080).

The fast launcher also saves only a terminal checkpoint. It rejects `--resume`
for incomplete runs. The `--compile` flag is part of the resolved config hash,
so evaluation must use the same flag as training.

## 4. Post-hoc evaluation from the final checkpoint

```bash
python algorithms/eadream/scripts/evaluate_final.py \
  --game frostbite --seed 0 --compile
```

Restores `final.pt`, runs the **100-episode deterministic formal evaluation**
(fresh env from the train seed) and the **30-seed PPOSB3 audit**
(seeds 20000–20029), and writes
`evaluations/formal_100k.json` and `evaluations/pposb3_audit_100k.json`
(both with SHA256 self-hashes). Unclipped raw returns throughout.

## 5. Status / summarize / archive

```bash
python algorithms/eadream/experiment.py status  --run-dir <run_dir>
python algorithms/eadream/experiment.py archive --run-dir <run_dir> \
  --archive-root <archive_dir>
```

`archive` validates and publishes an immutable, no-overwrite copy of the run
(result scope: configs, provenance, terminal state, final checkpoint,
evaluations, logs, replay index — not replay payloads).

## 6. Full campaign (16 games × 5 seeds)

```bash
python algorithms/eadream/experiment.py run \
  --config algorithms/eadream/configs/atari16_campaign.yaml
```

Serial scheduler with durable completed-run skipping; incomplete runs are not
resumed because no periodic recovery checkpoint is saved. Failures are recorded
and the campaign continues. At formal speed
budget **~200 GPU-days**; at fast speed and one game at a time via the
launcher, ~7–11 h per run.

## Known implementation gaps (as of this revision)

- `run_one` saves only the final checkpoint; periodic 10K/17.5K/… checkpoints
  and in-run evaluation calls are not wired yet. Use
  `scripts/evaluate_final.py` after a run completes (step 4).
- The fast launcher is the supported entry point for the published variant;
  the formal CLI is fully functional for single runs and campaigns.
