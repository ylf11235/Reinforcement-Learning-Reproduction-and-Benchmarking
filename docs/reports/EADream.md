# EADream on Atari-16 (100K): Experiment Report

Native reproduction of **EADream** (Peng et al., ICLR 2026 [[1]](#references)),
the event-aware DreamerV3-style world model, adapted from the official GPL-3.0
repository at commit `269f71af` (see [algorithms/eadream/NOTICE.md](../../algorithms/eadream/NOTICE.md)).
Status: **3 / 16 games completed** (alien, frostbite, montezuma_revenge), single
seed each, fast variant. The summary artifacts of all three runs are archived
under [results/eadream/](../../results/eadream/) (per-run configs, provenance,
logs, and evaluations; checkpoints, replay payloads, and TensorBoard curves are
excluded by size).

## 1. Protocol

### 1.1 Training contract (both variants)

- Budget: **100,000 agent interactions** per game (Atari-100K regime;
  action-repeat 4 → ~400K raw ALE frames), single environment, cold start
  from random parameters, seed 0.
- Environment: Gymnasium 1.2.2 / ALE-Py 0.12.1, `ALE/<Game>-v5` minimal
  action sets, 64×64 RGB + 64×64 MOG2 event-mask observation contract,
  fixed two-buffer max-pool, noop 30, sticky actions off, 108K-frame episode
  cap. Bowling/Tennis would use the FIRE start guard (not exercised yet).
- Training: prefill 2,500 random interactions → 100 pretraining batches →
  one gradient batch (16×64 sequences) per interaction to 100K
  (97,600 updates). Raw unclipped rewards throughout.
- Scoring: final checkpoint, **100-episode deterministic formal evaluation**
  (fresh env from the train seed) + **30-seed audit** (seeds 20000–20029),
  unclipped raw returns.

### 1.2 The fast variant (deviations from formal)

The formal protocol pins `torch.autograd.set_detect_anomaly(True)` around
world-model updates (upstream parity). Profiling on an RTX 5080 showed >50%
of wall-clock spent in anomaly stack bookkeeping (linecache/traceback), at
2.29 s/batch. The published runs use a **fast variant**:

| Switch | Formal | Fast | Rationale |
| --- | --- | --- | --- |
| `detect_anomaly_world_model` | true | **false** | removes the >50% bookkeeping tax |
| `torch.compile` (wm + behavior) | false | **true** | upstream's own default (`compile: True`) |
| `experiment.formal` | true | **false** | run recorded as diagnostic, never formal |
| output root | `..._100k_v1` | `..._100k_fast_v1` | formal tree stays untouched |

Net effect: 0.41 s/batch (vs 0.52 eager, 2.29 formal) — a 3.9× end-to-end
speedup. Learning semantics are unchanged; anomaly detection is a runtime
switch, while the compile setting is included in the resolved configuration
identity. Checkpoints from compiled runs carry `_orig_mod` key prefixes and
must be evaluated with compile on; incomplete runs cannot be resumed because
only the terminal checkpoint is saved. Hardware: RTX 5080 16 GB, PyTorch 2.11.0+cu130,
Python 3.10.20, Windows (compile via triton-windows 3.8.0).

## 2. Results

| Game | Formal eval (100 eps)<br>mean / std | Audit (30 seeds)<br>mean | Training<br>time | Segments with<br>nonzero return |
| --- | ---: | ---: | ---: | ---: |
| alien | **631.2** / 26.9 | 636.3 | 7.03 h | 12/13 |
| frostbite | **269.7** / 3.0 | 270.0 | 10.77 h | 13/13 |
| montezuma_revenge | **0.0** / 0.0 | 0.0 | 13.62 h | 0/13 |

Training-return trajectories (per-7,500-interaction segment boundary):

- **alien**: 150 (random prefill) → 190 → 340 → 410 → 450 → 910 (peak @40K)
  → 370 → … → 480–630 late. Noisy but clearly ascending.
- **frostbite**: 130 (prefill) → monotone climb into the 240–270 band by
  mid-training; final policy extremely stable (std 3 across 100 episodes).
- **montezuma_revenge**: **0.0 at every boundary from 2.5K to 100K**. The
  policy never obtained the first key; episode lengths vary (249–1,040),
  i.e. the agent moves but no scoring event ever occurs.

### 2.1 Comparison with the paper's released scores

Paper released 5-seed means (upstream `results/atari/EADream.json`, pinned
SHA256 in the repo): alien 775.6, frostbite 692.5, montezuma_revenge —
**not released** (no hard-exploration game is).

- **alien**: our 631.2 = 81% of the paper's 5-seed mean from a single seed.
- **frostbite**: paper spread is [328.6, 2296.3, 267.2, 276.2, 294.8] — the
  mean is dominated by one outlier seed. Our 269.7 sits inside the four-seed
  cluster (267–294), i.e. the same tier as the paper's typical seeds.
- **montezuma_revenge**: 0.0. See §3.

## 3. Finding: no exploration rescue from event awareness

A natural question for this benchmark suite: is there a game where PPO/MuZero
fail but the event-aware world model learns? Cross-referencing three evidence
layers:

1. **PPO_SB3 (10M steps, this repo)**: montezuma_revenge, pitfall, solaris,
   venture all at 0; skiing −29,972; tennis −24; private_eye 100 (≈0.1% HNS).
2. **EADream released scores** cover only 7 games — all event-dense; every
   hard-exploration game is absent.
3. **Our montezuma run**: 0.0 across training and both evaluation protocols,
   matching PPO_SB3's 0 at 1% of its interaction budget.

The only overlap between "PPO fails" and "EADream released" is private_eye,
where the paper's own seeds are [-387.5, -422.0, -672.1, 103.1, -984.0]
(mean −472, below random ≈30) — also not learned.

**Conclusion**: EADream's event-prediction auxiliary task improves sample
efficiency on games with dense event structure; it does **not** provide
exploration. Sparse-reward hard exploration remains unsolved for both
model-free PPO and this world-model family at these budgets. The negative
result is reported deliberately: it closes the question for this suite.

## 4. Wall-clock and throughput

| Metric | Value |
| --- | --- |
| Batch time (16×64), fast compile | 0.39–0.41 s |
| Batch time, eager (anomaly off) | 0.52 s |
| Batch time, formal (anomaly on) | 2.29 s |
| Throughput (steady, fast) | ~1.9 interactions/s |
| GPU compute utilization (fast) | 17–29% |
| Peak CUDA allocated | ~4.0 GB |
| Per-game totals | alien 25,320 s · frostbite 38,760 s · montezuma 49,032 s |

For context, the paper reports ~9.6 h/environment on a V100 32 GB with three
tasks per GPU in parallel (Appendix M), including the anomaly-detection cost.
Our fast-variant per-game times are comparable despite the consumer GPU and
Windows, by dropping the anomaly tax and enabling compile.

HNS conversion (Agent57 baselines): alien 631.2 → 5.85%, frostbite 269.7 →
4.79%, montezuma 0 → 0.00% (random 227.8/65.2/0; human 7127.7/4334.7/4753.3).

## 5. Reproduction

See [algorithms/eadream/RUNBOOK.md](../../algorithms/eadream/RUNBOOK.md) for
the full step-by-step protocol (validate → preflight → fast run → post-hoc
evaluation). The equivalent invocation for the published runs is:

```bash
python algorithms/eadream/scripts/run_game_fast.py --game <slug> --seed 0 --compile
python algorithms/eadream/scripts/evaluate_final.py --game <slug> --seed 0 --compile
```

Launcher provenance, as recorded in each run's `provenance.json`: the
frostbite and montezuma_revenge runs were launched by the at-the-time versions
of `run_game_fast.py` / `evaluate_final.py` (the checked-in scripts are their
continued evolution); the alien run was launched by a one-off per-game script
setting the same fast-variant switches. The recorded per-file implementation
hashes and command lines are archived with each run under
[results/eadream/](../../results/eadream/).

Per-run artifacts (archived under
`results/eadream/eadream_atari16_100k_fast_v1/games/<slug>/seed_000/`):
`run_state.json` (lifecycle + counters), `resolved_config.yaml` /
`input_config.yaml`, `provenance.json`, `logs/metrics.jsonl`,
`logs/transitions.jsonl`, `evaluations/formal_100k.json` and
`evaluations/pposb3_audit_100k.json`, and `checkpoints/metadata.json` (ties
the size-excluded `final.pt` to its SHA256). TensorBoard curves are likewise
excluded by size.

Known implementation gaps (being fixed before the formal campaign): periodic
10K/17.5K checkpoints and in-run evaluation calls are not yet wired into
`run_one`; the post-hoc evaluator covers scoring.

## References

1. Z.-H. Peng, S. Li, Z. Li, S. Ruan, Y. Liu, Y. He. *From Observations to
   Events: Event-Aware World Models for Reinforcement Learning.* ICLR 2026.
   arXiv:2601.19336.
2. Upstream implementation: MarquisDarwin/EAWM, commit `269f71af` (GPL-3.0).
3. C. Badia et al. *Agent57.* arXiv:2003.13350, 2020 (Human/Random baselines).
