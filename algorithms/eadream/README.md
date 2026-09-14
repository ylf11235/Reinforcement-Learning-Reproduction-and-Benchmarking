# EADream (Event-Aware Dreamer) — Atari-16 100K

A native PyTorch reimplementation of **EADream**, the DreamerV3-style
event-aware world model from *"From Observations to Events: Event-Aware World
Models for Reinforcement Learning"* (Peng et al., ICLR 2026 [[1]](#references)),
adapted from the official GPL-3.0 repository at commit
`269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241` ([[2]](#references), see
[NOTICE.md](NOTICE.md) and [LICENSE](LICENSE) for provenance).

EADream augments a DreamerV3-style agent with an **event prediction** auxiliary
task: a MOG2-based event segmentor produces binary event masks from frame
motion, the world model predicts those masks alongside image reconstruction,
and the event loss is harmonized with the other world-model objectives. The
paper reports state-of-the-art results on Atari 100K among non-search methods.

## What is in this directory

- `algorithm.py` — public `EADream` API (`learn` / `predict` / `save_checkpoint`
  / `load_checkpoint` / `close`)
- `engine/` — world model, imagination-based behavior learning, RSSM,
  distributions, schedules, checkpointing
- `networks/` — encoder, decoder, attention blocks, heads, harmonizers
- `envs/` — Gymnasium/ALE-Py adapter with the pinned observation contract
  (64×64 RGB + 64×64 event mask), fixed two-buffer max-pool, FIRE start guard
  for Bowling/Tennis
- `events/` — stateful MOG2 event extractor
- `replay/` — durable episode store and length-weighted sequence sampling
- `experiment.py` — CLI and per-run durable lifecycle
  (`validate` / `preflight` / `run` / `status` / `summarize` / `archive`)
- `evaluation.py` — 100-episode formal evaluation and 30-seed audit protocols
- `campaign.py`, `archive.py` — serial campaign scheduling and immutable archives
- `configs/` — the formal Atari-16 manifests (16 games × seeds 0–4)
- `scripts/run_game_fast.py`, `scripts/evaluate_final.py` — the fast-variant
  launcher and post-hoc evaluator for the published protocol (see RUNBOOK;
  the archived runs record their exact at-the-time launchers in
  `provenance.json`)
- `reference_manifest.json`, `reference_data/` — pinned upstream file hashes
  and released paper scores used by the parity gates

## Two run protocols

| Protocol | Command flavor | Notes |
| --- | --- | --- |
| **Formal** | `python algorithms/eadream/experiment.py run --config ... --seed N` | Byte-pinned to the upstream semantics, including `torch.autograd.set_detect_anomaly(True)` around world-model updates. ~2.3 s/batch on an RTX 5080 (the anomaly bookkeeping alone costs >50% wall-clock; the upstream paper's official numbers include it too). |
| **Fast** | `python algorithms/eadream/scripts/run_game_fast.py --game <slug> --seed N --compile` | Diagnostic variant: anomaly detection off, `torch.compile` on (the upstream default), `experiment.formal=false`. ~0.4 s/batch. **All scores currently published in this repository come from this variant** and are marked as such. |

Both protocols share the same environment stack, model, training loop, and
evaluation code; they differ only in the two runtime switches above and write
to separate campaign roots (`runs/atari16/eadream_atari16_100k_v1` vs
`..._fast_v1`), so formal and fast results never mix.

## Results snapshot (fast variant, seed 0)

| Game | Formal eval (100 eps) | Audit (30 seeds) | Paper released mean (5 seeds) |
| --- | ---: | ---: | ---: |
| alien | 631.2 | 636.3 | 775.6 |
| frostbite | 269.7 | 270.0 | 692.5* |
| montezuma_revenge | 0.0 | 0.0 | not released |

\* The paper's frostbite five-seed spread is [328.6, 2296.3, 267.2, 276.2,
294.8]; our seed-0 result sits inside the four-seed cluster (267–294).
montezuma_revenge is a hard-exploration game where PPO at 10M steps also
scores 0 — see the negative-result discussion in
[docs/reports/EADream.md](../../docs/reports/EADream.md#3-finding-no-exploration-rescue-from-event-awareness).

See the repository leaderboard and
[docs/reports/EADream.md](../../docs/reports/EADream.md) for
cross-method HNS comparison; the summary artifacts of these runs are archived
under [results/eadream/](../../results/eadream/).

## Setup

```bash
pip install -r algorithms/eadream/requirements-formal.txt   # from repository root
# PyTorch CUDA build is installed separately — pick the wheel matching YOUR
# GPU and driver (https://pytorch.org/get-started/locally/); example:
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

The pinned ALE-Py 0.12.1 installation provides the ROMs used in this suite.
Each game config pins the ROM SHA256 that preflight verifies. If your ALE-Py
distribution lacks ROMs, supply legally obtained copies before running.

## References

1. Z.-H. Peng, S. Li, Z. Li, S. Ruan, Y. Liu, Y. He. *From Observations to
   Events: Event-Aware World Models for Reinforcement Learning.* ICLR 2026.
   arXiv:2601.19336.
2. MarquisDarwin/EAWM (upstream implementation, GPL-3.0), pinned commit
   `269f71af5b3510fbcdb2c7d3ebeb22b1a15f5241`.
