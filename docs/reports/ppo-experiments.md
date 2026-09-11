# PPO Experiments: PPOSB3 Atari-16 and PPO-PyTorch

## Contents

1. [Cross-Backend Score Comparison (Shared Games)](#1-cross-backend-score-comparison-shared-games)
2. [PPO Paper vs PPOSB3 on Atari-10 (HNS)](#2-ppo-paper-vs-pposb3-on-atari-10-hns)
3. [Atari-16 Environment Selection Logic](#3-atari-16-environment-selection-logic)
4. [PPOSB3 Unified Configuration (ppo_sb3_atari16_10m_seed0_v1)](#4-pposb3-unified-configuration-ppo_sb3_atari16_10m_seed0_v1)
5. [PPO-PyTorch Configuration (Upstream CartPole Defaults on Atari)](#5-ppo-pytorch-configuration-upstream-cartpole-defaults-on-atari)
6. [Run Results and Completion Status](#6-run-results-and-completion-status)
7. [Additional Notes](#7-additional-notes)

## 0. Document Status

This document summarizes the final state and configuration baselines of both finished PPO
experiment lines:

- `PPOSB3` Atari-16 × 10M × seed-0 campaign: 16/16 `COMPLETED`, summary artifacts written
  2026-09-10 05:29, run directory `PPOSB3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1/`.
- `PPO-PyTorch` (upstream CartPole-style implementation) 10M × seed-0: the completed
  campaign on disk is the v2 rerun (Alien, Frostbite, with TensorBoard logging),
  `COMPLETED` 2026-09-11 03:54, run directory `PPO-PyTorch/runs/ppo10_v2/`.

This document does not repeat the protocol details of the PPOSB3 protocol document
(`ppo_sb3_experiments.md`); it only references its conclusions.

---

## 1. Cross-Backend Score Comparison (Shared Games)

Shared games = Alien and Frostbite, the two games with a completed `PPO-PyTorch` run.
PPO-paper values are the mean final scores from Schulman et al., *Proximal Policy
Optimization Algorithms*, Table 6 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347);
per-game transcription also in `PPOSB3/configs/atari16/game_choice.md`). PPOSB3 and
PPO-PyTorch values are both final@10M: 30 audit episodes on seeds 20000–20029,
deterministic policy, unclipped raw return, seed 0.

| Game | PPO-paper | PPOSB3 | PPO-PyTorch |
| --- | ---: | ---: | ---: |
| alien | 1850.3 | 1001.33 | 465.33 |
| frostbite | 314.2 | 2044.33 | 199.00 |

On both games PPOSB3 (GAE + minibatch + 8-vectorized envs + Atari-tuned hyperparameters)
clearly beats PPO-PyTorch, which keeps the upstream CartPole semantics. Note that the
PPO-paper column is a cross-protocol reference value (see Section 2 caveats), not a
same-protocol score: the paper measures the mean of the last 100 training episodes under
the stochastic behavior policy over 3 seeds, while both local backends audit the final
checkpoint deterministically on fixed seeds. The two local backends also differ slightly
in protocol (Sections 7.1–7.2), so this is an equal-budget, equal-audit-seed approximate
comparison, not a strict same-protocol ranking.

---

## 2. PPO Paper vs PPOSB3 on Atari-10 (HNS)

Full comparison on the ten Atari-10 games (`PPO-PyTorch/configs/atari10.yaml`), all of
which are covered by PPO-paper Table 6. Raw scores are converted to human-normalized
scores (HNS) as `HNS = (Agent − Random) / (Human − Random)`; each raw score is followed
by its HNS in parentheses. Human and random baselines come from the Agent57 reference
tables (Badia et al., *Agent57: Outperforming the Atari Human Benchmark*, Appendix H.4,
[arXiv:2003.13350](https://arxiv.org/abs/2003.13350)). PPOSB3 scores are from
`ppo_sb3_atari16_10m_seed0_v1` (final@10M audit, seed 0).

| # | Game | Human | PPO-paper<br>(HNS) | PPOSB3 final@10M<br>(HNS) | final ÷ paper<br>(HNS) |
| ---: | --- | ---: | --- | --- | ---: |
| 1 | alien | 7127.7 | 1850.3 (23.5%) | 1001.33 (11.2%) | 0.48 |
| 2 | asteroids | 47388.7 | 2097.5 (3.0%) | 1377.00 (1.4%) | 0.48 |
| 3 | bowling | 160.7 | 40.1 (12.4%) | 30.00 (5.0%) | 0.41 |
| 4 | chopper_command | 7387.8 | 3516.3 (41.1%) | 2156.67 (20.5%) | 0.50 |
| 5 | enduro | 860.5 | 758.3 (88.1%) | 958.73 (111.4%) | 1.26 |
| 6 | frostbite | 4334.7 | 314.2 (5.8%) | 2044.33 (46.4%) | 7.95 |
| 7 | gopher | 2412.5 | 2932.9 (124.1%) | 2598.67 (108.6%) | 0.88 |
| 8 | kung_fu_master | 22736.3 | 23310.3 (102.6%) | 16326.67 (71.5%) | 0.70 |
| 9 | seaquest | 42054.7 | 1204.5 (2.7%) | 1746.00 (4.0%) | 1.48 |
| 10 | tennis | -8.3 | -14.8 (58.1%) | -24.00 (-1.3%) | -0.02 |

Observations:

- By HNS ratio, PPOSB3 exceeds the paper on 3/10 games: frostbite (7.95x), seaquest
  (1.48x), and enduro (1.26x).
- PPOSB3 reaches roughly half the paper's HNS on alien, asteroids, bowling, and
  chopper_command (0.41x–0.50x) and 0.70x–0.88x on kung_fu_master and gopher.
- tennis is the one negative outlier: PPOSB3's final policy scores -24.0, below even
  the random baseline (-23.8), for a -1.3% HNS versus the paper's 58.1%; its
  periodic-best checkpoint had reached -1.0, so the final policy regressed rather than
  the system being incapable.
- HNS rescales each game to the human gap, so this table weights games very differently
  from the raw-score comparison in Section 1: frostbite's raw 6.5x becomes 7.95x, while
  enduro's modest raw lead (1.26x) crosses the 100% human line (88.1% → 111.4%).
- PPOSB3 exceeds the human baseline (HNS > 100%) on enduro (111.4%) and is near it on
  gopher (108.6%); the paper exceeds it on enduro, gopher, and kung_fu_master.

Protocol caveats (these numbers are a cross-paper reference, not a strict same-protocol
ranking):

- **Score definition.** PPO-paper reports the mean final score over the last 100
  training episodes under the stochastic behavior policy, averaged over 3 seeds
  (40M game frames = 10M agent steps per run). PPOSB3 audits the immutable final
  checkpoint deterministically for 30 fixed-seed episodes, single seed 0. Both count
  the same 10M policy decisions and score unclipped raw returns.
- **Environment stack.** The paper used the 2017-era Arcade Learning Environment; this
  repository uses Gymnasium 1.2.2 / ALE-Py 0.12.1 with `ALE/<Game>-v5`, minimal action
  sets, sticky actions off, and a FIRE start guard for bowling/tennis (Section 4).
- **Single seed.** PPOSB3 is one seed; per-game deltas within roughly ±25% of the paper
  are not meaningful evidence of a systematic difference.

The PPO paper did not report skiing or solaris, so a full 16-game Atari-16 paper
comparison is impossible; 14/16 games have paper values. PPO-paper values for the six
mandatory hard-exploration games that are in Table 6 (montezuma_revenge 42.0,
pitfall -32.9, private_eye 69.5, tennis -14.8, venture 0.0) are consistent with this
repository's finding that 10M-step PPO does not explore effectively on them.

---

## 3. Atari-16 Environment Selection Logic

The Atari-16 environment description (selection logic, fixed game list, environment
stack, reference score provenance, and the Atari-10 subset) lives in the standalone
document **[`../envs/atari-16.md`](../envs/atari-16.md)**. Key facts retained here for
context:

- The suite is a fixed 16-game manifest defined in `PPOSB3/configs/atari16.yaml`,
  with per-game rationale and score provenance in `PPOSB3/configs/atari16/game_choice.md`.
- It combines ten skill-coverage representatives (chosen to stress PPO) with the six
  mandatory MuZero-below-human games (montezuma_revenge, pitfall, private_eye, skiing,
  solaris, venture) kept as the hard-exploration core.
- The PPO-PyTorch line uses the Atari-10 subset (the ten skill representatives with
  denser rewards, dropping the six hard-exploration games, `PPO-PyTorch/configs/atari10.yaml`).

---

## 4. PPOSB3 Unified Configuration (ppo_sb3_atari16_10m_seed0_v1)

All 16 games share one identical configuration; the per-game YAMLs
(`PPOSB3/configs/atari16/games/<slug>.yaml`) are field-for-field identical except
slug / env_id / output paths (verified by diff). Key points:

| Category | Unified value |
| --- | --- |
| Algorithm | SB3 `PPO("CnnPolicy")` |
| Budget | `10,000,000` requested SB3 timesteps per game; rollout-aligned effective `10,000,384` (`128 x 8 = 1024` alignment) |
| Seed | `seed=0` only |
| Vectorization | `DummyVecEnv`, 8 workers, serial scheduling |
| Environment | `ALE/<Game>-v5`, ALE frameskip 1; wrappers: noop 30, frame skip 4, max pooling, grayscale 84x84, stack 4 (channel-first), no life-loss termination, `repeat_action_probability=0.0`, 108,000 raw frames per-episode cap |
| Start protocol | `start_protocol` injected by config.py: bowling and tennis (and breakout, not in this suite) get `fire`; all others `none` |
| Rewards | Training uses `sign(reward)` clipping; evaluation is unclipped (raw is the primary score) |
| PPO hyperparameters | `n_steps=128`, `batch_size=256`, `n_epochs=4`, `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2` (no vf clip), constant `lr=2.5e-4`, `ent_coef=0.01`, `vf_coef=0.5`, `max_grad_norm=0.5`, advantage normalization on |
| Periodic evaluation | Every 262,144 timesteps, 10 episodes, seeds 10000–10009, deterministic, raw return |
| Final audit | Final checkpoint, 30 episodes, seeds 20000–20029, deterministic, unclipped raw return as the primary score |
| Checkpoints | periodic every 262,144 (keep 3) + milestones every 1,048,576 + best (periodic selection) + final |
| Videos | Best checkpoint, seeds 30000–30002, deterministic, native ALE RGB, H.264 / 15 FPS |
| Provenance | pip freeze, hardware, ROM SHA256, source snapshot; versions pinned gymnasium 1.2.2 / ale-py 0.12.1 / SB3 2.8.0 / torch 2.11.0+cu130; CUDA mandatory, silent CPU fallback forbidden |

Scoring follows Sections 2 and 6 of the PPOSB3 protocol document
(`ppo_sb3_experiments.md`): the primary score is the raw-return mean of final@10M over
30 fixed audit seeds; best-checkpoint scores are used only for selection and videos,
never as a replacement for the primary score.

---

## 5. PPO-PyTorch Configuration (Upstream CartPole Defaults on Atari)

### 5.1 Origin and shape

- Algorithm source: [`nikhilbarhate99/PPO-PyTorch`](https://github.com/nikhilbarhate99/PPO-PyTorch)
  (MIT, commit `728cce8`, 2023-12-08), preserving the upstream discrete PPO update
  semantics.
- Fully independent of PPOSB3: its own `PPO.py` / `config.py` / `atari_env.py` /
  `evaluation.py` / `experiment.py`, never reading PPOSB3 configuration; environment
  construction reuses the same `baselines_common.atari` wrapper contract.

### 5.2 Algorithm configuration (unified across the ten games)

| Parameter | Value | Notes |
| --- | --- | --- |
| `max_ep_len` | **2000** | Upstream CartPole official default is 400; because the Atari environment differs from the official CartPole environment (single pixel-based environment, longer episodes), it is set to 2000 here |
| `update_timestep` | 1600 | One full PPO update every 1600 transitions |
| `K_epochs` | 40 | Each update iterates 40 epochs over the full 1600-sample batch (full-batch, no mini-batches) |
| `eps_clip` | 0.2 | |
| `gamma` | 0.99 | |
| `lr_actor` / `lr_critic` | 3e-4 / 1e-3 | Single Adam optimizer with per-group learning rates |
| `entropy_coef` / `value_coef` | 0.01 / 0.5 | |
| Budget | `10,000,000` transitions per game | Single environment, no rollout-alignment overshoot, exactly 10,000,000 |

Key semantic differences from PPOSB3: no GAE (advantage = normalized Monte-Carlo return
minus value), no mini-batching, no gradient clipping, no vectorization, and separate
Nature-CNN encoders for actor and critic (obs / 255 normalization). `max_ep_len=2000`
applies to both training and evaluation (see Section 7).

### 5.3 Environment and evaluation configuration

- The environment contract matches PPOSB3: `ALE/<Game>-v5`, noop 30, frame skip 4,
  grayscale 84x84, stack 4, 108,000-raw-frames cap, sticky off, sign-clipped training /
  raw evaluation.
- bowling and tennis likewise use `start_protocol: fire`.
- Evaluation: final checkpoint, 30 episodes, seeds 20000–20029, deterministic (argmax),
  unclipped raw return — the seed range matches the PPOSB3 audit, but the per-episode
  step cap is `max_ep_len=2000`.
- Training logs both clipped and raw episode returns (`reward` / `raw_reward` in
  `train_log.jsonl`); the v2 rerun added TensorBoard scalars (`train/loss`, `time/fps`,
  `rollout/ep_rew`, `rollout/ep_raw_rew`, `rollout/ep_len`).

---

## 6. Run Results and Completion Status

### 6.1 PPOSB3 Atari-16 (16/16 COMPLETED)

final@10M = mean deterministic raw return over 30 audit seeds (20000–20029);
best = best periodic evaluation.

| # | Game | final raw mean | best periodic | training time (s) |
| ---: | --- | ---: | ---: | ---: |
| 1 | alien | 1001.33 | 2666.0 | 8,548 |
| 2 | asteroids | 1377.00 | 1529.0 | 7,123 |
| 3 | bowling | 30.00 | 67.7 | 9,538 |
| 4 | chopper_command | 2156.67 | 4840.0 | 8,847 |
| 5 | enduro | 958.73 | 1320.7 | 14,036 |
| 6 | frostbite | 2044.33 | 2596.0 | 8,478 |
| 7 | gopher | 2598.67 | 2200.0 | 8,364 |
| 8 | kung_fu_master | 16326.67 | 33840.0 | 4,223 |
| 9 | montezuma_revenge | 0.00 | 0.0 | 10,986 |
| 10 | pitfall | 0.00 | 0.0 | 24,814 |
| 11 | private_eye | 100.00 | 100.0 | 11,571 |
| 12 | seaquest | 1746.00 | 1808.0 | 10,314 |
| 13 | skiing | -29971.77 | -29973.9 | 13,200 |
| 14 | solaris | 0.00 | 902.0 | 30,715 |
| 15 | tennis | -24.00 | -1.0 | 20,125 |
| 16 | venture | 0.00 | 220.0 | 11,341 |

Total training wall-clock is about 202,225 s (≈ 56.2 h). The six mandatory
MuZero-below-human games (montezuma_revenge, pitfall, skiing, solaris, venture at 0 or
very poor; private_eye only 100) match the expectation set when the suite was selected:
10M-step PPO does not explore effectively on these environments.

Full per-episode scores, checkpoint SHA256s, and video SHA256s are in
`PPOSB3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1/summary/`
(`scores.csv/json/md`, `throughput.csv`, `campaign_report.json`).

### 6.2 PPO-PyTorch (Alien, Frostbite — COMPLETED 2026-09-11)

The completed campaign is `ppo_pytorch_atari2_10m_seed0_v2` (a rerun of Alien and
Frostbite from the atari10 manifest, adding TensorBoard), run directory
`PPO-PyTorch/runs/ppo10_v2/`.

| Game | transitions | episodes | final raw mean / median / std | wall time (s) | effective FPS |
| --- | ---: | ---: | --- | ---: | ---: |
| alien | 10,000,000 | 14,337 | 465.33 / 450.0 / 72.2 | 25,777 | ~388 |
| frostbite | 10,000,000 | 20,733 | 199.00 / 230.0 / 68.5 | 23,637 | ~423 |

Final checkpoint SHA256s: alien `ddca74fd…31d24c`, frostbite `d2e9a8d4…22369e` (full
values in `PPO-PyTorch/runs/ppo10_v2/campaign/campaign_result.json`).

---

## 7. Additional Notes

1. **Different evaluation step caps.** PPOSB3 evaluation episodes can run to the ALE
   108,000-raw-frame limit (≈ 27,000 agent steps); PPO-PyTorch evaluation reuses
   `max_ep_len=2000` as the per-episode step cap. For the two completed games, measured
   evaluation episode lengths (alien 474–1061, frostbite 412–668) are well below 2000,
   so the cap did not truncate and the scores are comparable; but if policies on other
   games later approach 2000 steps, PPO-PyTorch scores would be systematically
   underestimated, and cross-backend comparisons must state this.
2. **The training-time `max_ep_len=2000` is an artificial truncation.** At 2000 steps
   the environment resets and the boundary is written to the buffer as a terminal (MC
   return reset to zero, no bootstrapping), matching the upstream CartPole `max_ep_len`
   semantics. 2000 agent steps ≈ 8,000 raw frames, far below the 108,000 ALE cap, so
   this artificial boundary almost always fires first.
3. **Same budget unit, different effective totals.** Both count "one policy action
   decision = 1 step": PPOSB3 aligns to 10,000,384 via 8-environment rollouts;
   PPO-PyTorch is exactly 10,000,000 on a single environment. Both correspond to about
   40M raw ALE frames per game.
4. **Throughput difference.** PPOSB3 ≈ 1,170–1,440 FPS (alien 1,170 / frostbite 1,180);
   PPO-PyTorch ≈ 388–423 FPS (40-epoch full-batch updates + single-env collection),
   about 2.7–3.0x slower on the same game.
5. **PPO-PyTorch has no best-checkpoint mechanism.** Only the final checkpoint (plus a
   `last.pth` auto-resume checkpoint written after every update, without hash gating);
   periodic selection, videos, and immutable archiving exist only in PPOSB3.
6. **Identical training rewards on both sides**: sign-clipped training, raw evaluation;
   PPO-PyTorch additionally logs both episode-return conventions.
7. **Randomness differences**: PPO-PyTorch derives a deterministic seed per reset via
   `seed + episode`; PPOSB3's periodic/audit/video seed system is documented in its
   protocol document.
8. **Result file entry points**:
   - PPOSB3: `PPOSB3/runs/ppo16/ppo_sb3_atari16_10m_seed0_v1/summary/`
   - PPO-PyTorch: `PPO-PyTorch/runs/ppo10_v2/campaign/campaign_result.json` and each
     `games/<slug>/seed_000/` (`result.json`, `train_log.jsonl`, `tensorboard/`,
     `checkpoints/{last,final}.pth`).
9. **The full 10-game PPO-PyTorch campaign config is ready**
   (`PPO-PyTorch/configs/atari10_campaign.yaml`, run root
   `PPO-PyTorch/runs/ppo10/ppo_pytorch_atari10_10m_seed0_v1`) but has not been
   executed; only the 2-game v2 rerun above exists on disk. If launched later, results
   must be reported under the conventions of this document.
