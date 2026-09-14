<div align="center">
  <img src="docs/icon.svg" width="900" alt="RL-Reproduction-Benchmarking wordmark">
</div>

# Reinforcement-Learning-Reproduction-and-Benchmarking

An open-source project primarily focused on **reproducing, understanding, and benchmarking modern reinforcement learning algorithms**.

## Project Purpose

This repository is intended to support practical reinforcement-learning
technology selection, not only paper-score reproduction. Comparing methods by
environment interaction steps alone is incomplete: the same step budget can
require very different learner compute, wall-clock time, memory, and hardware
utilization. Every published experiment therefore records both the achieved
return and the cost of obtaining it.

The default comparison setting is **all on one GPU**: one training run owns one
GPU, runs are scheduled serially, and the hardware/software configuration is
archived with the result. The primary practical budget is a 6-12 hour
wall-clock window per run. Within that window, results are reported at the
nearest completed Env-Interact checkpoint from this canonical ladder:

`100K / 200K / 500K / 1M / 5M / 10M / 500M / 1B`.

The ladder is a reporting grid, not a claim that every method reaches every
checkpoint. A comparison must always state the actual wall-clock time and the
actual Env-Interact steps reached; missing runtime data is shown as `--` rather
than estimated.

The current work is organized along several complementary dimensions:

- **Action spaces:** discrete and continuous control
- **Agent settings:** single-agent and multi-agent learning
- **Learning paradigms:** model-free and model-based reinforcement learning
- **Practical focus:** academic baselines, training heuristics, reproducibility techniques, and production-oriented engineering practices

The project documents both algorithmic details and practical techniques that affect real-world performance, including exploration, replay-buffer design, normalization, reward and advantage estimation, target-network updates, entropy regularization, parallel data collection, evaluation across random seeds, and training efficiency.

Broader domain-specific coverage is a **future direction**, rather than the primary focus of the current work. Planned domains may include games, robotics, locomotion, navigation, manipulation, and general control tasks.

Representative and planned methods include **PPO, SAC, TD7, Agent57, BBF, BTR, multi-agent reinforcement learning methods, and World-Action Models** (EADream is the first world-model entry, see [algorithms/eadream](algorithms/eadream)).

---

## Leaderboard: Atari-16 (Human-Normalized Score)

Human-normalized score (HNS) on the [Atari-16 suite](docs/envs/atari-16.md):
`HNS = 100% × (Agent − Random) / (Human − Random)`, with Human / Random baselines
from the Agent57 reference tables [[2]](#references). Aggregated across games as
median / mean of per-game HNS, sorted by median.

| # | Method | Reproduced | Median HNS | Mean HNS | Training wall-clock / game (mean) | Env-Interact steps / game | Games | Remark |
| ---: | --- | :---: | ---: | ---: | ---: | ---: | :---: | --- |
| 1 | Agent57 † | ✗ | 314.6% | 2706.7% | -- | -- | 16/16 | |
| 2 | PPO (Schulman et al., 2017) † | ✗ | 9.1% | 32.9% | -- | 10M | 14/16 | |
| 3 | **PPO_SB3** | ✓ | **3.7%** | **16.6%** | **3.51 h** | **10M** | 16/16 | |
| -- | EADream | ✓ | 4.8% (partial) | 3.5% (partial) | 10.47 h | 100K | 3/16 | World model; fast variant |
| -- | PPO_PyTorch | ✓ | 3.3% (partial) | 3.3% (partial) | 6.86 h | 10M | 2/16 | 2/10 of its Atari-10 target |

**†** Literature reference scores quoted from the cited papers. The **Reproduced**
column marks whether this repository contains runnable source code and archived
artifacts for the method: ✓ = trained, audited, and archived here (re-runnable);
✗ = paper-only reference (no code here, numbers cannot be re-run or audited).

Partial medians and means describe only the completed games. They are not
ranked against the full-suite rows and must not be read as Atari-16 aggregates.

Leaderboard notes:

- **Agent57**: per-game HNS values taken from Badia et al., Appendix H.4 [[2]](#references),
  aggregated here over the 16 games of this suite (their headline 57-game result is higher).
- **PPO (paper)**: mean final score of the last 100 training episodes, 3 seeds, 40M game
  frames per run, converted to HNS [[1]](#references). The paper omits skiing and solaris,
  so the aggregate covers the remaining 14 games.
- **PPO_SB3**: this repository's run — 10M agent steps per game, single seed 0, final
  checkpoint audited over 30 deterministic episodes on fixed seeds (20000–20029),
  unclipped raw returns. Per-game HNS uses the audit mean. The wall-clock value is
  reported per game as a 1.17–8.53 h range (mean 3.51 h); the 16-game sum is
  202,225 s. Individual game times are retained in
  `results/ppo_sb3/.../summary/throughput.csv`.
- **PPO_PyTorch**: this repository's reference backend with upstream CartPole semantics
  [[6]](#references); only alien and frostbite (2/10 Atari-10 target games,
  also 2/16 common-suite games) have completed 10M runs so
  far. Mean HNS over these two games: 3.3% (alien 3.4%, frostbite 3.1%) — a partial
  aggregate, pending the remaining games. Wall-clock is reported per game as a
  6.57–7.16 h range (mean 6.86 h) across the two completed runs; their combined
  time is 49,414 s.
- `--` means that no directly comparable wall-clock or Env-Interact value is
  available in the cited artifact. Literature rows are not assigned a runtime
  from another implementation or hardware platform.
- `*` The PPO paper reports 40M game frames, equivalent to approximately 10M
  agent interaction steps under the action-repeat convention used here; its
  training wall-clock is not reported in the cited result.
- `‡` PPO_SB3 per-game training time ranges from 4,223 s to 30,715 s
  (mean 12,639 s); each run reached 10,000,384 effective SB3 timesteps after
  rollout alignment. The 16-game total is 202,225 s.
- `§` PPO_PyTorch per-game training time ranges from 23,637 s to 25,777 s
  (mean 24,707 s); each run reports 10,000,000 transitions. The 49,414 s sum
  covers the two completed games only, not a 16-game total.
- `‖` EADream per-game training time: alien 25,320 s (7.03 h), frostbite
  38,760 s (10.77 h), montezuma_revenge 49,032 s (13.62 h); mean 10.47 h over
  the three completed runs (combined 113,112 s). Post-hoc evaluation
  (~5–9 min/game) is excluded from the training interval, as recorded in the
  per-run `evaluations/` artifacts.
- `¶` **EADream rows are a different regime**: a sample-efficient world-model
  method trained at **100K interactions/game** (the Atari-100K protocol),
  versus 10M for the PPO rows. Its scores therefore come from 1% of the PPO
  interaction budget, at ~7–14 h of learner compute per game. The published
  runs use the **fast variant** (anomaly detection off, `torch.compile` on,
  `experiment.formal=false`); see
  [algorithms/eadream/README.md](algorithms/eadream/README.md#two-run-protocols).
  Per-game: alien 631.2 (5.9% HNS), frostbite 269.7 (4.8% HNS),
  montezuma_revenge 0.0 (0.0% HNS — hard-exploration game where PPO at 10M
  also scores 0), from the 100-episode deterministic formal evaluation of the
  final checkpoint. Full report: [docs/reports/EADream.md](docs/reports/EADream.md).
- The existing PPO campaigns were originally step-budgeted. Their recorded
  runtimes are shown as historical evidence and are not retroactively normalized
  to the 6-12 hour protocol; new campaigns should use the wall-clock window and
  checkpoint ladder above.
- The rows above are **cross-protocol reference values**, not a strict same-protocol
  ranking: they differ in environment stack (2017 ALE vs Gymnasium/ALE-Py), stochasticity,
  action sets, budget accounting, and score definitions. Per-game raw scores, HNS, and
  protocol caveats: [full report](docs/reports/ppo-experiments.md#2-ppo-paper-vs-pposb3-on-atari-10-hns).

Headline findings from the first campaign (PPO on Atari-16, 16/16 completed):

- 10M-step PPO **exceeds the human baseline on enduro (111.4% HNS)** and is near it on
  gopher (108.6%), but **fails to explore** the six mandatory hard-exploration games
  (montezuma_revenge, pitfall, venture, solaris at 0; skiing at −100.9%; private_eye at −4.3%).
- Against the PPO paper under equal 10M-step budget, PPO_SB3 wins on frostbite (7.95x HNS),
  seaquest (1.48x), and enduro (1.26x), and reaches 0.41x–0.88x of the paper elsewhere
  ([per-game table](docs/reports/ppo-experiments.md#2-ppo-paper-vs-pposb3-on-atari-10-hns)).

## Documentation

- **[docs/envs/atari-16.md](docs/envs/atari-16.md)** — the Atari-16 environment suite:
  why these 16 games (skill coverage + the six MuZero-below-human games [[3]](#references)),
  the fixed game order, the unified `ALE/<Game>-v5` environment contract, and where
  reference scores come from.
- **[docs/reports/ppo-experiments.md](docs/reports/ppo-experiments.md)** — experiment
  report: PPOSB3 (Stable-Baselines3 PPO, Atari-tuned) and a PPO-PyTorch reference
  backend with upstream CartPole semantics; unified configurations, run results,
  completion status, and cross-backend protocol notes.
- **[docs/reports/EADream.md](docs/reports/EADream.md)** — EADream (event-aware
  world model) Atari-16 100K campaign: protocol and fast-variant definition,
  per-game results vs the paper's released scores, and the hard-exploration
  negative finding (montezuma_revenge 0 at 100K, matching PPO at 10M).

## Running the Experiments

| Backend | Code & protocol | Step-by-step runbook |
| --- | --- | --- |
| PPO_SB3 (Atari-16) | [algorithms/ppo_sb3](algorithms/ppo_sb3/) | [RUNBOOK.md](algorithms/ppo_sb3/RUNBOOK.md) |
| PPO_PyTorch (Atari-10) | [algorithms/ppo_pytorch](algorithms/ppo_pytorch/) | [RUNBOOK.md](algorithms/ppo_pytorch/RUNBOOK.md) |
| EADream (Atari-16, 100K) | [algorithms/eadream](algorithms/eadream/) | [RUNBOOK.md](algorithms/eadream/RUNBOOK.md) |

### Experiment Environment

Published runs used Windows 11 (build 22631), an NVIDIA RTX 5080 with 16 GB
VRAM, 32 GB system RAM, and Python 3.10.20 (`py310`). Individual run provenance
records the software stack and GPU driver; historical OpenCV wheel CI build
metadata is not the operating system of the training host. Use a CUDA-enabled
PyTorch wheel compatible with your own GPU and driver.

### Quick Start

From the repository root (CUDA required):

```bash
# 1) Install a PyTorch CUDA build matching YOUR GPU and driver —
#    requirements.txt deliberately does not pin one.
#    Example (cu130, the index used for the published runs);
#    pick yours at https://pytorch.org/get-started/locally/ :
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
# 2) Then everything else:
pip install -r requirements.txt
python algorithms/ppo_sb3/experiment.py validate \
  --config algorithms/ppo_sb3/configs/atari16_campaign.yaml
```

Also:
- **[results/](results/)** — summary artifacts of the published campaigns
  (scores, hashes, evaluations, provenance). Bulky training curves and full
  rollout logs are kept out of git.
- **[baselines_common/](baselines_common/)** — shared code used by the model-free
  backends: Gymnasium/ALE environment construction (`envs/atari.py`), episode
  recording (`envs/video.py`), JSONL logging, and atomic-write/hash utilities.

## Repository Layout

```text
algorithms/ppo_sb3/      # SB3 PPO backend: code, Atari-16 configs, README + RUNBOOK
algorithms/ppo_pytorch/  # CartPole-semantics PPO reference backend (Atari-10)
algorithms/eadream/      # event-aware DreamerV3-style world model (Atari-16, 100K) — self-contained (GPL-3.0)
baselines_common/        # shared envs (Atari + video), logging, and IO utilities
docs/                    # environment suite docs + experiment reports
results/                 # summary artifacts + TensorBoard curves of published runs
```

Setup: first install a **PyTorch CUDA build matching your GPU and driver**
(not covered by `requirements.txt` — choose at
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/);
the published runs used `torch 2.11.0+cu130`), then
`pip install -r requirements.txt` (pinned: gymnasium 1.2.2, ale-py 0.12.1,
stable-baselines3 2.8.0). CUDA required for training runs.

## Work in Progress

| Work stream | Method family | Status | Published output |
| --- | --- | --- | --- |
| **PPO_SB3** — Atari-16 @ 10M steps | model-free PPO (Stable-Baselines3, Atari-tuned) | ✓ Completed — 16/16 games, seed 0 | [report](docs/reports/ppo-experiments.md) · [results](results/ppo_sb3/) |
| **PPO_PyTorch** — Atari-10 @ 10M steps | model-free PPO (upstream-faithful reference backend) | Partial — 2-game rerun archived (2/10 target games; 2/16 shared games) | [report](docs/reports/ppo-experiments.md) · [results](results/ppo_pytorch/) |
| **EADream** — Atari-16 @ 100K steps | event-aware world model (DreamerV3-style) | Partial — fast-variant runs completed for 3/16 games; formal-protocol runs pending | [report](docs/reports/EADream.md) · [results](results/eadream/) |
| **BTR** [[8]](#references) — Atari-16 @ 10M / 50M steps | value-based, Rainbow-DQN lineage (Impala/IQN, Munchausen, NoisyLinear, PER) | In progress — native backend under implementation, not yet merged | — |
| **MARL** | multi-agent reinforcement learning | In progress — work stream started, no published runs yet | — |

---

## Methodology in Brief

- **Budget accounting:** one policy action decision = 1 step; 10M steps ≈ 40M raw ALE
  frames per game. Published campaigns also report the nearest completed checkpoint
  from the `100K / 200K / 500K / 1M / 5M / 10M / 500M / 1B` ladder.
- **Wall-clock accounting:** record training wall-clock from the first training
  update to the final checkpoint, in seconds and hours. Evaluation, video export,
  and checkpoint I/O are reported separately when they are not part of the training
  interval. A run must include the exact GPU, driver/runtime, precision, number of
  environments, and parallelism settings; changing these creates a different
  runtime comparison.
- **All-on-one-GPU constraint:** one run owns one GPU and campaigns are scheduled
  serially. The target run window is 6-12 hours; methods that finish earlier or
  fail to reach a checkpoint remain valid observations, but their actual time and
  Env-Interact steps must be reported together.
- **Environment (PPO runs):** Gymnasium 1.2.2 / ALE-Py 0.12.1, `ALE/<Game>-v5` minimal action sets,
  noop 30, frame skip 4, grayscale 84×84, stack 4, sticky actions off, 108k-frame
  episode cap; training rewards sign-clipped, evaluation unclipped.
- **Scoring (PPO runs):** the primary score is the **final** checkpoint's mean raw return
  over 30 deterministic episodes on fixed seeds — best-checkpoint scores are used only
  for selection and videos, never as the headline number.
- **EADream protocol:** the event-aware model uses 64x64 RGB plus an event mask,
  and its headline scores are final-checkpoint means over 100 formal evaluation
  episodes. Its 30-seed PPO-comparable audit is reported separately; it is not
  interchangeable with the PPO training environment or its 10M-step budget.
- **Time-versus-quality reporting:** when periodic evaluations are available, publish
  both reward-vs-Env-Interact and reward-vs-wall-clock curves. For cross-game
  aggregation, use per-game HNS before taking a median or mean; raw scores are not
  comparable across games.
- **Provenance:** per-run configs, ROM SHA256s, checkpoint hashes, per-episode scores,
  and hardware/pip-freeze snapshots are archived with every campaign. Summary
  artifacts of published runs are committed under [results/](results/); bulky
  training curves and full rollout logs stay out of git.

## References

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. *Proximal Policy
   Optimization Algorithms.* arXiv:1707.06347, 2017. https://arxiv.org/abs/1707.06347
2. C. Badia et al. *Agent57: Outperforming the Atari Human Benchmark.*
   arXiv:2003.13350, 2020. https://arxiv.org/abs/2003.13350
3. J. Schrittwieser et al. *Mastering Atari, Go, Chess and Shogi by Planning with a
   Learned Model.* arXiv:1911.08265, 2019. https://arxiv.org/abs/1911.08265
4. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, N. Dormann. *Stable-Baselines3:
   Reliable Reinforcement Learning Implementations.* JMLR 22(268), 2021.
   https://jmlr.org/papers/v22/20-1364.html
5. M. Towers et al. *Gymnasium: A Standard Interface for Reinforcement Learning
   Environments.* arXiv:2407.17032, 2024. https://arxiv.org/abs/2407.17032
6. N. Barhate. *PPO-PyTorch: Minimal PyTorch implementation of PPO.*
   https://github.com/nikhilbarhate99/PPO-PyTorch (MIT).
7. Z.-H. Peng, S. Li, Z. Li, S. Ruan, Y. Liu, Y. He. *From Observations to
   Events: Event-Aware World Models for Reinforcement Learning.* ICLR 2026.
   arXiv:2601.19336. Upstream code: MarquisDarwin/EAWM (GPL-3.0), commit
   `269f71af`.
8. T. Clark, M. Towers, C. Evers, J. Hare. *Beyond The Rainbow: High
   Performance Deep Reinforcement Learning on a Desktop PC.* arXiv:2411.03820,
   2024. https://arxiv.org/abs/2411.03820. Official implementation:
   VIPTankz/BTR.

## License

This repository is released under the [Apache License 2.0](LICENSE), with two
directory-scoped exceptions that carry their own license files:

- [algorithms/eadream](algorithms/eadream/) — **GPL-3.0** ([LICENSE](algorithms/eadream/LICENSE),
  [NOTICE](algorithms/eadream/NOTICE.md)): a native reimplementation adapted
  from the GPL-3.0 upstream EADream/EAWM repository, so the directory remains
  GPL-3.0 as a whole.
- [algorithms/ppo_pytorch](algorithms/ppo_pytorch/) — retains the upstream
  **MIT** [LICENSE](algorithms/ppo_pytorch/LICENSE) of
  `nikhilbarhate99/PPO-PyTorch` for the vendored third-party core
  ([THIRD_PARTY.md](algorithms/ppo_pytorch/THIRD_PARTY.md)).
