# MRQ Runbook

Reproduction steps for the MRQ backend from a clean checkout. Commands are
written for the **repository root**. The implementation itself is
working-directory independent (every path resolves from `__file__`; verified
from multiple working directories in the research workspace before
publication), but the relative paths shown below only spell out from the
root. From `algorithms/mrq/`, use `experiment.py` and `configs/...`
directly; from anywhere else, anchor both paths to the repository, e.g.

```bash
python <repo-root>/algorithms/mrq/experiment.py validate \
  --config <repo-root>/algorithms/mrq/configs/atari16_1m.yaml
```

## 1. Prerequisites

- Python 3.10 (the published runs used 3.10.20).
- A CUDA-enabled PyTorch wheel matching **your** GPU and driver — install it
  first, exactly as in the repository quick start; `requirements.txt`
  deliberately does not pin a torch build. The published runs used
  `torch 2.11.0+cu130` on an RTX 5080.
- `pip install -r requirements.txt`
- DMC experiments additionally need `dm-control` (pinned in
  `requirements.txt`; pulls `mujoco`); CarRacing experiments need `Box2D`
  and `pygame` (also pinned there). Atari experiments need neither.
- Optional: `tensorboard` for curve inspection (not needed for training).

Hardware assumption: one GPU with ≥16 GB VRAM and ~32 GB system RAM. The 1M
-transition replay buffer (84×84 uint8 frames + indices) is ≈7–8 GB and is
placed on GPU when free memory allows (auto-probed), otherwise host RAM.

## 2. Validate and preflight

```bash
python algorithms/mrq/experiment.py validate \
  --config algorithms/mrq/configs/atari16_1m.yaml
python algorithms/mrq/experiment.py validate \
  --config algorithms/mrq/configs/dmc_humanoid_walk_500k.yaml
python algorithms/mrq/experiment.py validate \
  --config algorithms/mrq/configs/car_racing_600k.yaml
python algorithms/mrq/experiment.py preflight \
  --config algorithms/mrq/configs/dmc_humanoid_walk_500k/games/humanoid_walk.yaml
```

`validate` resolves a manifest and prints its semantic fingerprint.
`preflight` additionally builds the real environment once and checks free
disk (≥20 GB).

## 3. Real-stack smoke (~1–2 min)

```bash
python algorithms/mrq/experiment.py run \
  --config algorithms/mrq/configs/dmc_humanoid_walk_500k/games/humanoid_walk.yaml \
  --smoke --run-dir <Your Workdir>/mrq_smoke/humanoid_walk
```

Expected: `[run] COMPLETED steps=800 mean_raw=... wall=...s`, a
`run_state.json` with `"state": "COMPLETED"`, and `summary/scores.json`
with `audit_episodes: 2`. The smoke path exercises the full lifecycle
(training, periodic eval, checkpointing, audit, provenance) at 800 steps.

## 4. Single game and full campaign

```bash
# one game (writes to output.run_root/<experiment id>/<game slug>)
python algorithms/mrq/experiment.py run \
  --config algorithms/mrq/configs/atari16_1m/games/alien.yaml

# whole campaign (idempotent: verified completed games are skipped)
python algorithms/mrq/experiment.py campaign \
  --config algorithms/mrq/configs/atari16_1m.yaml
```

Override the run root with `--run-root <path>` (campaign) or `--run-dir
<path>` (single run). Seeds live in the YAML (frozen experiment identity);
`--seed` is rejected with an explanation — edit the config instead.

## 5. Status, summary, archive, curves

```bash
python algorithms/mrq/experiment.py status   --run-dir <run>
python algorithms/mrq/experiment.py summarize --run-dir <run>
python algorithms/mrq/experiment.py archive  --run-dir <run> --archive-root <dir>
tensorboard --logdir <run>/tensorboard
```

`summarize` prints the audit mean ± std over episodes and the step budget.
`archive` copies a `COMPLETED` run into an immutable, hash-manifested
archive and refuses to overwrite. To render a deterministic episode video
from the final checkpoint (requires `imageio`):

```bash
python algorithms/mrq/record_episode.py --run-dir <run> --seed 20000 --out episode.mp4
```

## 6. Resume behavior

Resume is **rejected by design** (`run --resume` exits with status 2): the
checkpoints contain model + optimizer + RNG + replay state but not the
environment emulator state, so exact continuation is impossible. Interrupted
runs are marked `INTERRUPTED` and must restart fresh under the same
experiment id.

## 7. Output locations and scoring

Per run directory: `input_config.yaml`, `resolved_config.yaml`,
`run_state.json`, `provenance.json`, `train_log.jsonl`,
`tensorboard/`, `checkpoints/<generation>/manifest.json` (+ weights), and
`summary/scores.{json,csv,md}`.

Primary score: mean **unclipped raw return** over 30 deterministic episodes
(seeds 20000–20029) of the **final** checkpoint. Budget unit is agent steps;
Atari 1 step = 4 raw ALE frames, DMC 1 step = 1 dm_control control step (see
[docs/envs/dmc.md](../../docs/envs/dmc.md)).

## 8. Expected wall-clock (reference GPU: RTX 5080, torch 2.11 cu130)

| Workload | Measured throughput | Wall-clock |
| --- | --- | --- |
| Atari 1M steps (alien / frostbite) | 32.9 / 33.1 train FPS | 8.49 / 8.47 h |
| DMC humanoid-walk 500K steps | 55.1 train FPS | 2.55 h |
| CarRacing 600K steps | 14.2 train FPS | 11.95 h |

Extrapolations at the same throughput: Atari 2.5M ≈ 21 h, 10M ≈ 84 h per
game; the full 16-game suite at both budgets is a multi-week serial
undertaking on one GPU. Evaluation adds ~2–4 min per run; checkpoint I/O is
reported separately in `provenance.json`.

## 9. Known failure modes

- `validate FAILED: ...`: unknown key, wrong variant toggle, or budget that
  does not match the frozen campaign — the message names the field.
- `preflight FAIL env construction failed`: missing ROMs (Atari) or missing
  `dm-control`/MuJoCo (DMC).
- OOM during replay allocation: the buffer needs ≈7–8 GB; let the auto-probe
  place it on host RAM by freeing VRAM or reducing `algorithm.buffer_size`
  (this changes the semantic fingerprint — record it as a new variant).
- Windows path length limits: keep run roots short.
