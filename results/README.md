# Results

Summary-level artifacts of the published campaigns (checkpoints, videos, replay
payloads, and full immutable archives are too large for git and are kept out of
the repository; every score in the [leaderboard](../README.md) is traceable to
the files below).

## Archive conventions

The public artifacts are publication copies, not byte-for-byte originals.
Each campaign carries a publication manifest (e.g. the EADream
[manifest](eadream/eadream_atari16_100k_fast_v1/publication_manifest.json))
that records both the original file SHA256 and the SHA256 of each published
file, so the published bytes and their historical identities can be audited
offline.

- **Absolute paths were relativized.** Paths inside archived JSON now start at
  the producing workspace's runs root (`ppo16\...`, `ppo10_v2\...`) or use the
  `<site-packages>\` placeholder; source-file prefixes inside recorded
  tracebacks point at this repository's modules (`ppo_sb3\...`).
- **A localized Windows error message was translated to English**
  (`[WinError 5] Access denied.`) in the recorded failure of the runs that
  crashed once and resumed.
- **Bulky artifacts are not published:** TensorBoard event files and the full
  per-rollout `train_log.jsonl` stay out of git (blocked in `.gitignore`);
  they remain in the source run archives.
- EADream derived configs, states, and provenance have publication-specific
  hashes. Their checkpoint metadata and unchanged evaluation files retain the
  original run identity. The manifest records both identities explicitly;
  original hashes must not be used to verify the rewritten file bytes.
- MRQ publication copies additionally drop local editable-install entries
  from the recorded `pip_freeze`, and the DMC run's `raw_frames` is corrected
  to the DMC budget definition (the source recorded the Atari ×4 value; see
  [docs/envs/dmc.md](../docs/envs/dmc.md)). MRQ `resolved_config.yaml` copies
  carry repository-relative run roots and still recompute to the fingerprint
  recorded in their `run_state.json` / checkpoint manifests.

The third-party wheel's recorded build metadata contained external CI paths;
those logs were excluded from the public EADream provenance copy.

The published provenance uses `algorithms/eadream/` paths; `source_implementation_sha256`
records hashes measured for the original implementation, not necessarily for
the code as currently published. Source-only scripts are not published.

## ppo_sb3/ppo_sb3_atari16_10m_seed0_v1 — Atari-16, 16/16 COMPLETED

- `summary/` — `scores.csv|json|md` (per-game final/best raw stats, checkpoint
  and video SHA256s), `campaign_report.json` (aggregate + per-episode scores),
  `throughput.csv`, `failures.csv`
- `games/<slug>/seed_000/` — per-game `eval_log.jsonl` (periodic-evaluation
  history) and `run_state.json`; the per-rollout `train_log.jsonl` and
  TensorBoard curves are not published

## ppo_pytorch/ppo_pytorch_atari2_10m_seed0_v2 — Atari-10 (2-game rerun), COMPLETED

- `campaign/` — `campaign_result.json`, `campaign_state.json`
- `games/{alien,frostbite}/seed_000/` — `result.json`; `train_log.jsonl` and
  TensorBoard curves are not published

## eadream/eadream_atari16_100k_fast_v1 — Atari-16 @ 100K, fast variant, 3/16 COMPLETED

Summary artifacts of the three published fast-variant runs (alien, frostbite,
montezuma_revenge, seed 0). Full report: [docs/reports/EADream.md](../docs/reports/EADream.md).

- `games/<slug>/seed_000/` — `run_state.json`, `resolved_config.yaml` +
  `input_config.yaml`, `provenance.json` (incl. per-file implementation
  hashes), `logs/metrics.jsonl`, `logs/transitions.jsonl`,
  `evaluations/formal_100k.json` + `evaluations/pposb3_audit_100k.json`,
  `checkpoints/metadata.json` (ties the excluded `final.pt` to its SHA256)
- Excluded as bulky: `checkpoints/final.pt` (~0.5 GB per run), `replay/`
  payloads, exported videos, TensorBoard curves

## mrq/mrq_atari16_1m_raw — Atari-16 house/raw-reward variant @ 1M, 2/16 COMPLETED

Summary artifacts of the two completed runs (alien, frostbite, seed 0) under
the house environment contract with **raw (unclipped) training rewards** at a
1M-step diagnostic budget. Full report: [docs/reports/MRQ.md](../docs/reports/MRQ.md).

- `<game>/` — `run_state.json`, `resolved_config.yaml` + `input_config.yaml`,
  `provenance.json`, `summary/scores.{csv,json,md}`,
  `checkpoints/final/manifest.json` (ties the excluded checkpoint weights to
  their SHA256s)
- Excluded as bulky: checkpoint weights and replay payloads (~67 MB per
  checkpoint generation, 13 generations per run), `train_log.jsonl`,
  TensorBoard curves
- Protocol note: the alien run predates the addition of frostbite to the
  campaign manifest, so its archived suite section lists one game
  (fingerprint `b95eba7a…`); frostbite ran under the current two-game
  manifest (fingerprint `3c09883b…`). Both identities are self-consistent
  with their archived resolved configs; re-running the campaign under the
  current manifest would conservatively re-run alien (fingerprint mismatch
  refuses the skip).

## mrq/mrq_dmc_humanoid_walk_500k — DMC humanoid-walk @ 500K, 1/1 COMPLETED

- `humanoid_walk/` — same keep-list as above; `raw_frames` corrected to
  500,000 env transitions per [docs/envs/dmc.md](../docs/envs/dmc.md)
- Excluded as bulky: checkpoint weights, `train_log.jsonl`, TensorBoard
  curves, and the recorded rollout video (re-recordable from the source
  checkpoint in the research workspace via `record_episode.py`)

## mrq/mrq_car_racing_600k and mrq/mrq_car_racing_pilot_100k — CarRacing, COMPLETED

Summary artifacts of the pixel continuous-control campaign and its frozen
budget-reduced diagnostic. Full report:
[docs/reports/MRQ.md](../docs/reports/MRQ.md);
environment contract: [docs/envs/car_racing.md](../docs/envs/car_racing.md).

- `car_racing/` in each campaign — the standard MRQ keep-list (scores,
  configs, run state, sanitized provenance, final-checkpoint manifest).
  The source runs already recorded the correct ×2 raw-frame budget
  (600K steps = 1.2M frames; 100K = 200K); the derivation script asserts
  this instead of rewriting, so the only published-vs-source edits are the
  `run_root` relativization (these runs' `pip_freeze` was already free of
  local editable installs).
- Excluded as bulky: checkpoint weights, `train_log.jsonl`, TensorBoard
  curves.
- Protocol note: the 600K headline (audit 886.7 ± 24.4) and the 100K pilot
  (902.1 ± 27.2) are separate campaigns — never merged into one number.

Protocol notes: PPO_SB3 primary scores are final-checkpoint audits
(30 episodes, seeds 20000–20029, deterministic, unclipped raw return);
PPO_PyTorch uses the same seed range with a 2000-step per-episode cap;
EADream scores are 100-episode deterministic formal evaluations plus the same
30-seed audit. See [docs/reports/ppo-experiments.md](../docs/reports/ppo-experiments.md)
and [docs/reports/EADream.md](../docs/reports/EADream.md).
