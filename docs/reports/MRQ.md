# MRQ — Clean-Room Reproduction Report

**MRQ** ("Towards General-Purpose Model-Free Reinforcement Learning",
Fujimoto, D'Oro, Zhang, Tian, Rabbat, Meta FAIR,
[arXiv:2501.16142](https://arxiv.org/abs/2501.16142)) is a single-hyperparameter
model-free actor-critic for discrete, continuous-state, and pixel control. This
report covers the first published runs of our clean-room reimplementation
([algorithms/mrq](../../algorithms/mrq/)): the first backend in this repository
to span both discrete (Atari) and continuous (DMC) control.

Published coverage (all seed 0, single machine, RTX 5080 / torch 2.11 cu130):

| Campaign | Suite coverage | Budget | Status |
| --- | --- | --- | --- |
| `mrq_atari16_1m_raw` | Atari-16: alien, frostbite | 1M agent steps (4M raw frames) / game | COMPLETED, 2/16 |
| `mrq_dmc_humanoid_walk_500k` | DMC: humanoid-walk | 500K agent steps (500K env transitions) | COMPLETED, 1/1 |
| `mrq_car_racing_600k` | CarRacing: car_racing | 600K agent steps (1.2M env frames) | COMPLETED, 1/1 |
| `mrq_car_racing_pilot_100k` | CarRacing diagnostic | 100K agent steps (200K env frames) | COMPLETED, frozen diagnostic |

## 1. Provenance and licensing

The implementation is a **clean-room reimplementation**: written from the
paper, with the pinned upstream snapshot
(`facebookresearch/MRQ` @ `073adcd6b30f84694c241e00bd684db50242a215`) used
for behavioral verification only. The upstream license (CC BY-NC 4.0)
forbids incorporation, so **no upstream file is imported, copied, or
vendored**; see [THIRD_PARTY.md](../../algorithms/mrq/THIRD_PARTY.md). The
backend therefore falls under the repository's Apache-2.0 license.

## 2. Protocol

The full variant table, frozen algorithm constants, lifecycle guarantees,
and known deviations are documented in
[algorithms/mrq/README.md](../../algorithms/mrq/README.md). Summary:

- **Audit discipline (primary score):** final checkpoint, 30 deterministic
  episodes, env seeds 20000–20029, **unclipped raw return** — identical to
  the PPO_SB3/PPO_PyTorch house audits, making the per-game head-to-head
  below an equal-audit-seed comparison (budgets differ).
- **Atari environment:** the [atari-16.md](../envs/atari-16.md) contract
  (sticky off, noop 30, frame-skip 4, grayscale 84×84, stack 4, 108k-frame
  episode cap) with one deliberate deviation for this campaign: **raw
  (unclipped) training rewards** — MRQ's internal running reward-scale
  replaces sign-clipping. The variant carries its own experiment identity
  (`mrq_atari16_1m_raw`) and is never compared against sign-clip results
  without labelling.
- **DMC environment:** state input (67-dim), Box(21) actions, dense raw
  rewards, 1000-step episodes — [dmc.md](../envs/dmc.md).
- **CarRacing environment:** pixel input (BT.601 grayscale 96×96, 4-frame
  stack), Box(3) actions, raw rewards, action repeat 2, 1000-frame episode
  cap — [car_racing.md](../envs/car_racing.md).
- **Resume is rejected by design** (checkpoints lack emulator state);
  campaign skips require fingerprint + checkpoint-hash verification.

Protocol note (manifest history): the alien run predates the addition of
frostbite to the `atari16_1m` manifest; its archived suite section lists
one game (fingerprint `b95eba7a…`), frostbite's lists two
(`3c09883b…`). Both identities are self-consistent with their archived
resolved configs (see [results/mrq](../../results/mrq/)).

## 3. Atari-16 results @ 1M (house env, raw training rewards)

Audit = 30 deterministic episodes, seeds 20000–20029, final checkpoint.
HNS uses the Agent57 H.4 baselines [[2]](../../README.md#references) with
the same convention as the PPO report
([ppo-experiments.md](ppo-experiments.md)).

| Game | Audit mean raw return | HNS | Wall-clock | Train FPS |
| --- | ---: | ---: | ---: | ---: |
| alien | **1510.0 ± 693.2** | 18.6% | 30,567 s (8.49 h) | 32.9 |
| frostbite | **8009.7 ± 953.8** | 186.1% | 30,493 s (8.47 h) | 33.1 |

Partial aggregate over the two completed games (not an Atari-16 suite
aggregate): median HNS 102.3%, mean HNS 102.3%.

### Head-to-head with PPO_SB3 (same audit seeds, different budget)

PPO_SB3 numbers are from `ppo_sb3_atari16_10m_seed0_v1`
([ppo-experiments.md](ppo-experiments.md)) — the same deterministic
30-episode audit on seeds 20000–20029, but at **10M** agent steps and with
sign-clip training rewards.

| Game | MRQ @ 1M | PPO_SB3 @ 10M | Raw ratio | HNS ratio |
| --- | ---: | ---: | ---: | ---: |
| alien | 1510.0 (18.6%) | 1001.3 (11.2%) | 1.51× | 1.66× |
| frostbite | 8009.7 (186.1%) | 2044.3 (46.4%) | 3.92× | 4.01× |

MRQ exceeds the PPO_SB3 final scores on both games using **one tenth of the
environment interactions** (1M vs 10M agent steps). The trade is learner
compute: at ~33 train FPS (1 RL + 1 amortized encoder update per env step),
a 1M-step MRQ game costs ≈ 8.5 h versus PPO_SB3's mean 3.51 h per game —
more wall-clock per game, an order of magnitude fewer env transitions.
For context, EADream on the same audit reached alien 631.2 at 100K
interactions (7.03 h) — a different regime again
([EADream.md](EADream.md)).

### Cross-protocol reference: official MRQ curves

Official released learning curves (10 training seeds, sticky 0.25, no noop,
ε=1e-3 eval noise, final @ 2.5M agent steps): alien 2834.7, frostbite
4561.8. These are **not** same-protocol numbers: our house environment is
easier (sticky off, noop 30) and our audit is deterministic. Readings, with
that discount in mind: our frostbite @ 1M (8009.7) exceeds the official
@ 2.5M value; our alien @ 1M (1510.0) is ~53% of the official @ 2.5M value
at 40% of the budget. The `regression` variant (near-paper toggles) that
would give a same-stack parity signal is declared in
`algorithms/mrq/configs/` but has not been run to completion yet.

## 4. DMC humanoid-walk @ 500K

| Metric | Value |
| --- | --- |
| Audit mean raw return | **500.7 ± 26.1** (30 episodes, seeds 20000–20029) |
| Wall-clock | 9,195 s (2.55 h) |
| Train FPS | 55.1 |
| Official MRQ @ 500K (10-seed mean, cross-protocol) | 662 |

The single-seed audit reaches ~76% of the official 10-seed mean at the same
**agent-step** budget — but not the same environment budget: the paper's DMC
stack applies action repeat 2 (its 500K steps = 1M native control steps,
per the paper's Table 5 caption), while this contract uses no action repeat
(500K control steps). At half the environment interactions, ~76% of the
official mean is the honest reading. The official number also averages
training seeds under its own evaluation protocol; it is a reference
magnitude, not a parity claim.

## 5. CarRacing @ 600K (pixel continuous control)

Environment contract: [car_racing.md](../envs/car_racing.md) (`CarRacing-v3`,
BT.601 grayscale 96×96 with a 4-frame stack, Box(3) actions, action repeat
2, raw rewards, 1000-frame episode cap). Same audit discipline as every
other MRQ campaign: final checkpoint, 30 deterministic episodes, env seeds
20000–20029, unclipped raw return.

| Run | Steps | Raw frames | Audit mean raw return | Wall-clock | Train FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mrq_car_racing_600k` (headline) | 600,000 | 1,200,000 | **886.7 ± 24.4** | 43,028 s (11.95 h) | 14.2 |
| `mrq_car_racing_pilot_100k` (frozen diagnostic) | 100,000 | 200,000 | 902.1 ± 27.2 | 7,935 s (2.20 h) | 14.1 |

Readings, stated plainly:

- Both audits sit near the top of CarRacing's reward scale — values around
  900 per episode indicate consistently fast laps under the 95%
  lap-completion criterion — and the 600K policy is tight across the fixed
  seed set (σ = 24.4).
- The 100K pilot audits slightly **higher** than the 600K headline (902.1
  vs 886.7). That is a diagnostic variance point on a single seed: the two
  budgets are reported side by side and never merged; the pilot campaign is
  frozen as a diagnostic by its manifest and excluded from headline claims.
- To our knowledge this is the first published benchmark number for a
  TD-lineage agent on CarRacing under a deterministic fixed-seed audit. No
  same-protocol literature baseline exists, so no cross-method table is
  given for this environment.

## 6. Budget-ladder alignment

The campaigns land on or between the canonical reporting checkpoints of the
repository ladder (`100K / 200K / 500K / 1M / …`): Atari at 1M, DMC at
500K, CarRacing at 600K (between the 500K and 1M rungs — runs report their
actual budgets) with the diagnostic pilot at 100K. The declared primary
Atari campaigns (`atari16_2m5` paper-aligned and PPO-comparable
`atari16_10m`) are frozen in `algorithms/mrq/configs/` and not yet run; at
the measured 33 FPS they project to ≈ 21 h (2.5M) and ≈ 84 h (10M) per
game.

## 7. Limitations and next steps

- **Single seed, partial suite.** All results are seed 0; Atari coverage is
  2/16 games. The 102.3% partial HNS aggregate covers alien + frostbite
  only and must not be read as an Atari-16 suite result.
- **1M diagnostic, not the frozen primary budget.** The 2.5M / 10M
  sign-clip campaigns remain the intended headline comparisons; until they
  run, MRQ's leaderboard row is a partial, regime-noted entry.
- **Regression parity campaign pending** — clean-room validation against the
  official curves is planned under the near-paper variant.
- **DMC coverage is one task** (state input); the pixel-based DMC-vision
  line and the wider task suite are future work, as is any multi-seed DMC
  campaign.
- **CarRacing is one task, single seed**: the 600K headline and the frozen
  100K diagnostic pilot are reported separately; multi-seed campaigns and
  video recording (unsupported by the adapter) are future work.

## 8. Artifacts

Per-run summary artifacts, provenance, resolved configs, and the dual-hash
publication manifests: [results/mrq/](../../results/mrq/). Excluded as
bulky: checkpoint weights and replay payloads (~67 MB per generation),
`train_log.jsonl`, TensorBoard curves, rollout videos.
