# PPO_PyTorch — Upstream CartPole-semantics PPO on Atari-10

A reference PPO backend based on
[nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
(MIT; upstream attribution in `LICENSE` and `THIRD_PARTY.md`), deliberately
keeping the upstream **CartPole discrete defaults** on Atari pixels:

```text
max_ep_len=2000, update_timestep=1600, K_epochs=40, eps_clip=0.2,
gamma=0.99, lr_actor=3e-4, lr_critic=1e-3, entropy_coef=0.01, value_coef=0.5
```

The only Atari-specific adaptations are a Nature-CNN encoder and the
Gymnasium/ALE pixel environment adapter (same contract as PPO_SB3: noop 30,
frame skip 4, grayscale 84×84, stack 4, sticky off, sign-clipped training
rewards / raw evaluation). Training remains **single-environment, full-batch,
Monte-Carlo PPO** — no GAE, no minibatches, no gradient clipping, separate
actor/critic CNN encoders. This backend exists as a controlled contrast to
PPO_SB3's Atari-tuned configuration; see the
[experiment report](../../docs/reports/ppo-experiments.md).

Status: the completed campaign is `ppo_pytorch_atari2_10m_seed0_v2`
(alien, frostbite; results in [results/ppo_pytorch](../../results/ppo_pytorch/)).
The full 10-game campaign config is ready but not yet executed.

## Layout

- `PPO.py` — upstream algorithm (ActorCritic + PPO update), minimally adapted
- `experiment.py` — validate / run / campaign entry point
- `config.py` — YAML loading with upstream-default contract enforcement
- `atari_env.py`, `evaluation.py` — ALE adapter, raw-return evaluation
- `configs/atari10.yaml` — the 10-game manifest (Atari-16 minus the six
  hard-exploration games)
- `configs/atari10_campaign.yaml`, `configs/atari2_campaign.yaml` — campaign
  definitions (10-game planned; 2-game completed)
- `configs/games/<slug>.yaml` — per-game configs

## Usage

From the repository root. Full walkthrough with timings, incomplete-run rejection, and
expected outcomes: **[RUNBOOK.md](RUNBOOK.md)**.

```bash
# Validate a campaign config
python algorithms/ppo_pytorch/experiment.py validate \
  --config algorithms/ppo_pytorch/configs/atari10_campaign.yaml

# Smoke-test one game
python algorithms/ppo_pytorch/experiment.py run \
  --config algorithms/ppo_pytorch/configs/games/alien.yaml --smoke

# Run the full sequential campaign
python algorithms/ppo_pytorch/experiment.py campaign \
  --config algorithms/ppo_pytorch/configs/atari10_campaign.yaml
```

Each run writes `train_log.jsonl` (both clipped and raw episode returns),
TensorBoard scalars (`train/loss`, `time/fps`, `rollout/ep_rew`,
`rollout/ep_raw_rew`, `rollout/ep_len`), and `checkpoints/{last,final}.pth`
under `algorithms/ppo_pytorch/runs/` (git-ignored).

## Evaluation protocol

Final checkpoint, 30 episodes, seeds 20000–20029 (matching the PPO_SB3 audit
seed range), deterministic argmax, unclipped raw return. The per-episode step
cap is `max_ep_len=2000` — for the completed games this never truncated
(measured episode lengths 412–1061), but it must be stated in any
cross-backend comparison.

`config.py` enforces the upstream defaults: any config deviating from the
CartPole hyperparameters above fails validation, so the backend stays a
faithful reference implementation.
