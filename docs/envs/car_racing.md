# The CarRacing Environment (Pixel Continuous Control)

CarRacing is this repository's pixel-based continuous-control benchmark,
first exercised by the `MRQ` backend (`mrq_car_racing_600k`, manifest
`algorithms/mrq/configs/car_racing_600k.yaml`); experiment scores are
reported in [`../reports/MRQ.md`](../reports/MRQ.md). It completes the
action/observation quadrant matrix: discrete-actions Atari-16,
state-continuous DMC ([dmc.md](dmc.md)), and pixel+continuous CarRacing.

## 1. Environment identity

[gymnasium](https://arxiv.org/abs/2407.17032) `CarRacing-v3` (Box2D
top-down racer; `box2d` extra, `pygame` windowing): a car must follow a
procedurally generated track as fast as possible. Constructed with
`continuous=True`, `domain_randomize=False`, `lap_complete_percent=0.95`
— deterministic visuals, continuous control, an episode counts as a
completed lap after 95% track coverage. Pinned dependency versions live in
the repository `requirements.txt` (`Box2D==2.3.10`, `pygame==2.6.1`;
installed only when running CarRacing experiments).

## 2. Environment contract

Implemented by the shared adapter in
[`baselines_common/envs/car_racing.py`](../../baselines_common/envs/car_racing.py)
(`CarRacingAdapter` + `make_car_racing_env`):

| Item | Value |
| --- | --- |
| Env id | `CarRacing-v3` (continuous, no domain randomization, 95% lap completion) |
| Observation | BT.601 integer luma `(299·R + 587·G + 114·B + 500) // 1000`, uint8, CHW `(1, 96, 96)`; the adapter returns the last 4 frames stacked oldest-first `(4, 96, 96)` |
| Action space | continuous `Box(-1, 1, (3,))` (steering, gas, brake), rescaled to the native `Box([-1, 0, 0], [1, 1, 1])` via `low + (a+1)/2 · (high−low)` |
| Action repeat | 2 native env steps per agent step; rewards summed, termination/truncation OR-ed, repeat stops early on episode end |
| Episode cap | 1000 raw env frames (native truncation may fire earlier) |
| Rewards | raw, unclipped in training and evaluation (per-frame reward dominated by progress along the track, penalized by fuel and off-track; higher is better) |
| Stochasticity | none injected (no sticky actions, no noop starts, no start protocol) |
| Seeding | a constructor seed is applied to the first `reset()` that does not pass one (and to the action space), matching the seeded-at-construction behavior of the DMC adapter |
| Frame accounting | **1 agent step = 2 env frames** (action repeat 2): 600K steps = 1.2M frames, 100K steps = 200K frames |

## 3. Budget and scoring

- **Budget unit: agent steps** (1 step = 2 raw env frames, recorded as
  `budget.raw_frames`). The conversion differs per family in this
  repository — Atari ×4, CarRacing ×2, DMC ×1 — so agent-step counts are
  the comparable quantity across families.
- **Primary score:** mean raw return over the 30-episode deterministic
  final-checkpoint audit (env seeds 20000–20029), identical to the Atari-16
  and DMC house audits.
- Frozen house deviation for this variant: `algorithm.buffer_size = 200000`
  (reduced from the 1M default; pixel frames are 96×96). The game-file test
  hook `allow_reduced_buffer` exists solely so smoke/CLI checks can shrink
  the replay — it is recorded verbatim in the resolved config and never
  used by published runs.

## 4. Known deviations and limitations

1. The adapter exposes no `render()`: episode video recording
   (`record_episode.py`) is not supported for CarRacing runs.
2. Grayscale conversion uses integer BT.601 luma (not `cv2`/`gymnasium`
   wrappers) so the preprocessing is dependency-free and exactly
   reproducible from the recorded formula.
3. No literature baseline runs under this exact protocol; cross-method
   comparisons for CarRacing use explicitly labelled reference numbers
   from the cited sources (see the report).
