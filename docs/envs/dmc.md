# The DeepMind Control Suite (DMC) Environment Line

The DMC line is this repository's continuous-control benchmark family,
first exercised by the `MRQ` backend (`mrq_dmc_humanoid_walk_500k`, manifest
`algorithms/mrq/configs/dmc_humanoid_walk_500k.yaml`); experiment scores are
reported in [`../reports/MRQ.md`](../reports/MRQ.md). It complements the
Atari-16 suite: same audit discipline, different observation/action spaces.

## 1. What DMC is

The [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
(Tassa et al., [*DeepMind Control Suite*](https://arxiv.org/abs/1801.00690),
arXiv:1801.00690) is a set of continuous-control tasks built on MuJoCo,
addressed through the `dm_control` Python API. Published runs use
`dm-control 1.0.45` / `mujoco 3.12.0` (pinned in `requirements.txt`;
installed only when running DMC experiments).

## 2. Task line

- **Current task: `humanoid-walk`** (env id `Dmc-humanoid-walk`). A
  21-degree-of-freedom humanoid must walk forward as fast as possible with
  smooth, upright locomotion; the reward is a dense mix of forward velocity
  and upright-orientation terms. Chosen because the MRQ paper benchmarks it
  directly (official 500K-step, 10-seed mean ≈ 662 return), making it the
  cleanest continuous-control parity reference for this backend.
- Planned expansion follows the MRQ paper's DMC suite (cheetah, walker, dog
  families). New tasks are added as their own frozen campaign manifests —
  a completed run is never reused across tasks or budgets.

## 3. Environment contract

One identical contract for every DMC task in this repository, implemented by
the shared adapter in [`baselines_common/envs/dmc.py`](../../baselines_common/envs/dmc.py)
(`DMCGymAdapter`; no `shimmy` dependency):

| Item | Value |
| --- | --- |
| Env id convention | `Dmc-<domain>-<task>` (mapped from the game slug by `config.py`) |
| Observation | flattened float32 state vector — humanoid-walk: 67 dims (21 joint angles, 1 head height, 12 extremities, 3 torso-vertical, 3 center-of-mass velocity, 27 joint velocities) |
| Action space | continuous `Box(21,)` in [−1, 1] |
| Step semantics | 1 agent step = 1 dm_control **control step** (0.025 s for humanoid-walk); dm_control internally advances 5 MuJoCo physics steps (0.005 s each) per control step — physics substeps are simulator internals, not part of the budget unit |
| Frame stack | 1 (state input; no frame stacking) |
| Episode cap | 1000 agent steps |
| Rewards | dense raw reward from the task, unclipped in training **and** evaluation |
| Termination | `terminated` when the task discount is 0; `truncated` at task end without discount-0 or the 1000-step cap |
| Stochasticity | none injected (no sticky actions, no noop starts, no start protocol) |
| Rendering | default tracking camera, 320×240 RGB (video capture only) |

## 4. Adapter design notes

- The adapter (`DMCGymAdapter`) exposes dm_control through the Gymnasium
  `Env` interface with flattened observations, `raw_reward` in step info,
  and a render method. The `dm_control` import is lazy, so importing the
  module does not require `mujoco` installed.
- **Determinism caveat:** physics is seeded once at environment load (the
  training seed or the evaluation seed); episode resets re-randomize through
  dm_control's own task RNG, and exact reset-seed rebinding is not
  supported. Evaluation determinism comes from the deterministic action
  path (argmax/tanh policy), not from environment re-seeding. This is a
  documented deviation from the Gymnasium seeding convention.

## 5. Budget and scoring

- **Budget unit: agent steps**, where 1 agent step = 1 control step = 1
  environment transition. `budget.raw_frames` records the same count (×1
  multiplication) — unlike Atari, there is no frame-skip multiplier in the
  budget. For context only: humanoid-walk's 500K control steps correspond
  to 2.5M MuJoCo physics substeps internally.
- Conversion rule (must appear wherever budgets are compared): Atari
  1 agent step = 4 raw ALE frames; DMC 1 agent step = 1 env transition.
  Agent-step counts across the two families are directly comparable; raw
  frame counts are not.
- **Primary score:** mean raw return over the 30-episode deterministic
  final-checkpoint audit (env seeds 20000–20029) — the same rule as the
  Atari-16 house protocol.
- Official-paper comparisons (10 seeds, periodic evaluation) are
  cross-protocol reference values and are always labelled as such. Note the
  budget conversion difference: the MRQ paper's DMC stack applies action
  repeat 2 (its 500K steps = 1M native control steps), whereas this
  repository's contract uses no action repeat — agent-step counts match,
  environment-transition counts do not.

## 6. Known deviations from the paper protocol

1. State input only — the MRQ paper also reports pixel-based DMC-vision
   tasks; those are not implemented in this backend yet.
2. The determinism caveat of §4 (reset-seed rebinding unsupported).
3. Our audit uses 30 deterministic episodes at fixed seeds; the official
   curves report means over 10 training seeds with stochastic-periodic
   evaluation — numbers are not interchangeable.
