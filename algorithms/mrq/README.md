# MRQ — Clean-Room Reimplementation

MRQ ("Towards General-Purpose Model-Free Reinforcement Learning", Fujimoto
et al., Meta FAIR, [arXiv:2501.16142](https://arxiv.org/abs/2501.16142)) is a
single-hyperparameter model-free actor-critic that covers discrete
(Atari-57), continuous-state (MuJoCo / DeepMind Control), and pixel-based
control. It combines TD7-style model-free Q-learning with a TD-MPC2-style
self-supervised latent encoder (reward / dynamics / termination prediction),
n-step returns, and LAP replay prioritization — no world model, no planner.

This directory is an **independent clean-room reimplementation** (upstream
license is CC BY-NC 4.0; nothing is imported or vendored — see
[THIRD_PARTY.md](THIRD_PARTY.md) and [NOTICE.md](NOTICE.md)). It is the
repository's first backend spanning both discrete and continuous control.

## Variants

All protocol values live in version-controlled YAML under `configs/`; the
loader (`config.py`) strictly validates every key, the suite membership and
order, and the budget identity.

| Item | `house` (Atari) | `dmc` (continuous) | `car_racing` (pixel continuous) | `regression` (pending) |
| --- | --- | --- | --- | --- |
| environment contract | [atari-16.md](../../docs/envs/atari-16.md) (`ALE/<Game>-v5`, noop 30, sticky 0, frame-skip 4, sign-clip training / raw eval, fire start for bowling+tennis) | [dmc.md](../../docs/envs/dmc.md) (`Dmc-<domain>-<task>`, flattened state vector, Box actions, dense raw rewards, 1000-step episodes) | [car_racing.md](../../docs/envs/car_racing.md) (`CarRacing-v3`, BT.601 grayscale 96×96 + 4-frame stack, Box(3) actions, action repeat 2, raw rewards) | near-paper Atari toggles: sticky 0.25, no noop/fire, raw rewards |
| training reward | sign-clip by default; raw allowed as a documented variant (see below) | raw | raw (buffer_size frozen at 200K) | raw |
| evaluation | 10 periodic episodes / 100K steps (seeds 10000+); final audit 30 deterministic episodes, seeds 20000–20029, unclipped raw return | same cadence (50K periodic on the published run) | same audit; 100K periodic (20K on the pilot) | adds ε=1e-3 random-action noise in the evaluator |
| budget unit | agent steps (1 step = 4 raw ALE frames) | agent steps = env transitions (1 dm_control control step) | agent steps (1 step = 2 env frames) | agent steps |
| status | **2/16 games completed @ 1M** (`alien`, `frostbite`) | **humanoid-walk completed @ 500K** | **car_racing completed @ 600K** + frozen 100K diagnostic | not yet run; parity gate against the official learning curves is performed in the research workspace |

### The published Atari campaign: `mrq_atari16_1m_raw`

The completed Atari runs use the **house environment contract with raw
(unclipped) training rewards at a 1M-step budget** — MRQ's internal running
reward-scale replaces sign-clipping. This is a deliberate deviation from the
default house reward policy, so it carries its own experiment identity
(`mrq_atari16_1m_raw`, campaign `atari16_1m`) and is never mixed with
sign-clip results. The frozen primary campaigns `atari16_2m5` /
`atari16_10m` (sign-clip, paper-aligned 2.5M and PPO-comparable 10M budgets)
are declared in `configs/` but not yet run; the 1M results are labelled a
sample-efficiency diagnostic.

## Layout

```text
algorithms/mrq/
  experiment.py     # CLI: validate / preflight / run / campaign / status /
                    #       summarize / archive
  config.py         # YAML schema, resolution, strict validation, semantic
                    #       fingerprint (excludes output/provenance sections)
  agent.py          # MRQAgent: select_action / train_step / checkpoints
  networks.py       # encoder, policy, twin Q, two-hot reward, shift augment
  buffer.py         # LAP replay buffer with history/horizon indexing
  env_adapter.py    # train/eval env factories over the shared
                    #       baselines_common contracts (Atari / DMC /
                    #       CarRacing)
  evaluation.py     # deterministic periodic + audit episode loops
  record_episode.py # render one deterministic episode from a checkpoint
  configs/          # campaign manifests + per-game YAML (5 campaigns)
```

Runs write below a configurable run root (`output.run_root`, resolved
relative to this directory; override with `--run-dir` / `--run-root`).

## Algorithm constants (reference-faithful)

Frozen for all runs and recorded in every resolved config: batch 256, replay
1M transitions, γ = 0.99, hard target update every 250 train steps followed
by a 250-step encoder-only burst (amortized exactly 1 encoder + 1 RL update
per env step), uniform random actions until 10,000 transitions, exploration
noise σ 0.2 continuous / 0.1 discrete, target-policy smoothing σ 0.2 / clip
0.3 continuous and σ 0.1 / clip 0.15 discrete, encoder loss weights dyn 1 /
reward 0.1 / done 0.1, LAP α 0.4 (min priority 1, no IS correction), encoder
horizon 5, Q horizon 3, latents zs 512 / zsa 512 / za 256, hidden 512,
AdamW (encoder 1e-4, value 3e-4, policy 3e-4; weight decay 1e-4), value
grad-clip 20, Gumbel-Softmax τ 10 (discrete), pre-activation penalty 1e-5,
two-hot reward head 65 symexp bins in [−10, 10], random-shift augmentation
(pad 4) for pixel observations, reward scale = running mean |r| refreshed
from the buffer at each target update.

## Current coverage

| Run | Steps | Audit mean raw return (30 episodes, seeds 20000–20029) |
| --- | --- | --- |
| alien (`mrq_atari16_1m_raw`) | 1,000,000 | **1510.0 ± 693.2** |
| frostbite (`mrq_atari16_1m_raw`) | 1,000,000 | **8009.7 ± 953.8** |
| humanoid_walk (`mrq_dmc_humanoid_walk_500k`) | 500,000 | **500.7 ± 26.1** |
| car_racing (`mrq_car_racing_600k`) | 600,000 | **886.7 ± 24.4** |
| car_racing (`mrq_car_racing_pilot_100k`, frozen diagnostic) | 100,000 | 902.1 ± 27.2 (reported separately, never merged) |

Full report with cost accounting and cross-protocol context:
[docs/reports/MRQ.md](../../docs/reports/MRQ.md). Summary artifacts:
[results/mrq/](../../results/mrq/).

## Source snapshot policy

The code in this tree exists in **two attested generations**, each pinned
to the runs that executed it:

- **Generation 1 — Atari + DMC runs** (alien, frostbite, humanoid_walk):
  their provenance `source_sha256` records the bytes with the DMC adapter
  still in-package inside `env_adapter.py`. Those recorded hashes are kept
  verbatim in the published results; they attest what actually ran.
- **Generation 2 — CarRacing runs** (car_racing 600K + pilot): these
  executed the current tree — the `image_size`-parameterized encoder
  (Atari behavior unchanged at the 84×84 default) and the environment
  classes living in `baselines_common/envs/` (`car_racing.py`, `dmc.py`).
  The "re-sync lands with CarRacing" note from the first migration is
  resolved here: this migration is that landing.

From generation 2 on, every run's provenance additionally records
`env_source_sha256` — the hashes of the consumed
`baselines_common/envs/*.py` — so shared environment code is part of the
audit chain even though it lives outside this directory. Future source
evolution follows the same rule: publish the code that produced the
artifacts, never silently re-attribute an old run to new bytes.

## Lifecycle and guarantees

- Resolved config + semantic fingerprint are written before training;
  checkpoints are atomically promoted with hash manifests; a run flips to
  `COMPLETED` only after the final checkpoint, audit, scores, and provenance
  verify.
- **Resume is rejected by design**: model+buffer checkpoints do not contain
  the environment emulator state, so exact continuation is impossible;
  interrupted runs restart fresh (recorded `INTERRUPTED`).
- Campaign scheduling is idempotent: a completed game is skipped only after
  its config fingerprint and checkpoint hashes verify; failures are recorded
  in `campaign_report.json`.
- `archive` validates a `COMPLETED` run and refuses to overwrite an existing
  archive.

## Known deviations and limitations

1. Environments are never pickled (the reference reuses one env object);
   separate train/eval factories with explicit seeds are used instead.
2. Noop/fire/sticky toggles are config-driven rather than hardcoded.
3. Frame pooling uses Gymnasium `AtariPreprocessing` (equivalent by
   construction to the reference's manual pooling).
4. The DMC adapter seeds physics once at load; episode resets re-randomize
   via dm_control's task RNG. Exact reset-seed rebinding is unsupported —
   evaluation determinism comes from the deterministic action path
   ([dmc.md](../../docs/envs/dmc.md)).
5. The CarRacing adapter exposes no `render()`; episode video recording
   (`record_episode.py`) is unsupported for CarRacing runs.
6. Reporting-layer fix applied at migration: `raw_frames` in run summaries
   and provenance is taken from the resolved config budget (Atari ×4 frame
   skip, DMC ×1). The originally published DMC run recorded the Atari ×4
   value by mistake; the derived result copies in `results/mrq/` correct
   this and record the change in their publication manifest.
7. Single-seed coverage so far (seed 0); Atari coverage is 2/16 games at a
   1M diagnostic budget — partial aggregates are never ranked as full-suite
   results.

## Verification

This repository ships no test suite for this backend — algorithm packages
here are test-free by design, and the pre-migration test gates run outside
this repository before anything is published. Before each migration, the
development side runs the unit suites (config schema/fingerprints, buffer
indexing, network math, env-factory wiring with faked constructors, CLI
lifecycle with a fake environment, publication-manifest verification) plus
real-stack smoke runs; the outcomes are recorded in the migration plan and
summarized in the experiment report. Anyone can still verify the published
artifacts offline using only this repository: recompute the
`published_sha256` values of each campaign's `publication_manifest.json`
and re-derive the semantic fingerprint from the published
`resolved_config.yaml`.

## License

Apache-2.0 (repository-wide). Upstream attribution and the clean-room
statement: [THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE.md](NOTICE.md).
