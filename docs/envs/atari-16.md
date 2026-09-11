# The Atari-16 Environment Suite

The Atari-16 suite is a fixed 16-game manifest used as the primary benchmarking
environment of this repository. It was first exercised by the `PPOSB3`
Atari-16 × 10M × seed-0 campaign (`ppo_sb3_atari16_10m_seed0_v1`, manifest
`PPOSB3/configs/atari16.yaml`); experiment scores and per-backend configurations
are reported in [`../reports/ppo-experiments.md`](../reports/ppo-experiments.md).

## 1. Selection Logic

1. **Skill coverage (primary reason).** Ten complementary skill representatives
   were selected so that the suite covers the full spectrum of capabilities
   required by Atari games: position coverage and collection, dodging, jumping
   and climbing, blocking and interception, vehicle control, racing, melee
   combat, ranged combat, planning and puzzle solving, navigation and
   exploration, defense and rescue, precise timing, and resource management.
   Within each skill family, games with lower PPO human-normalized performance
   were preferred, so the suite stresses PPO rather than flattering it. A
   focused single-skill game was also retained wherever relying only on
   multi-skill games could hide a missing capability.
2. **Difficulty against the human baseline (secondary reason).** All six Atari
   games on which MuZero did not exceed the human baseline (Schrittwieser et
   al., 2020) are kept as a mandatory core: Montezuma's Revenge, Pitfall,
   Private Eye, Skiing, Solaris, and Venture. They are retained unconditionally
   because they represent the hardest exploration / long-horizon planning /
   sparse-reward challenges that motivated this study in the first place.

## 2. Game List (Fixed Campaign Order)

```text
alien, asteroids, bowling, chopper_command, enduro, frostbite, gopher,
kung_fu_master, montezuma_revenge, pitfall, private_eye, seaquest, skiing,
solaris, tennis, venture
```

Of the 16 games, the six mandatory hard-exploration core games are
`montezuma_revenge`, `pitfall`, `private_eye`, `skiing`, `solaris`, and
`venture`; the remaining ten are the skill representatives.

## 3. Environment Stack

All 16 games share one identical environment contract (unified PPOSB3
configuration):

| Item | Value |
| --- | --- |
| Env id | `ALE/<Game>-v5`, minimal action sets (Gymnasium 1.2.2 / ALE-Py 0.12.1) |
| Frameskip | ALE frameskip 1 + frame-skip-4 wrapper |
| Wrappers | noop 30, frame skip 4, max pooling, grayscale 84x84, stack 4 (channel-first), no life-loss termination |
| Stochasticity | sticky actions off (`repeat_action_probability=0.0`) |
| Start protocol | `fire` for bowling and tennis (breakout, not in this suite, also gets `fire`); `none` for all others |
| Episode cap | 108,000 raw frames per episode |
| Rewards | training uses `sign(reward)` clipping; evaluation is unclipped (raw is the primary score) |

The PPO-PyTorch reference backend reuses the same environment contract for its
Atari-10 subset: noop 30, frame skip 4, grayscale 84x84, stack 4,
108,000-raw-frames cap, sticky off, sign-clipped training / raw evaluation,
and `fire` start for bowling and tennis.

## 4. Reference Score Provenance

Reference scores come from three protocol-different sources and are not a
strict same-protocol ranking:

- **Agent57 means** — Badia et al., *Agent57: Outperforming the Atari Human
  Benchmark*, Appendix H.4 ([arXiv:2003.13350](https://arxiv.org/abs/2003.13350)).
  The Agent57 reference tables also provide the random and human raw-score
  baselines used for human-normalized scores across this repository.
- **Original PPO means** — Schulman et al., *Proximal Policy Optimization
  Algorithms*, Table 6 ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)).
- **This repository's own runs** — deterministic 30-episode final raw-return
  audits under the 10M-step, seed-0 protocol (see
  [`../reports/ppo-experiments.md`](../reports/ppo-experiments.md)).

Note: the PPO paper did not report skiing or solaris, so only 14/16 games have
paper values; a full 16-game paper comparison is impossible.

## 5. Atari-10 Subset

The PPO-PyTorch reference backend uses **Atari-10**, the subset of Atari-16
that drops the six mandatory hard-exploration games (where PPO reliably scores
~0) and keeps the ten skill representatives with denser rewards: alien,
asteroids, bowling, chopper_command, enduro, frostbite, gopher, kung_fu_master,
seaquest, tennis (`PPO-PyTorch/configs/atari10.yaml`).
