"""MRQAgent (clean-room): select_action / observe / train_step / checkpoints.

Semantics (paper + reference behavior; see PROTOCOL.md §4/§7):

Cadence — per env step exactly 1 RL update; every ``target_update_freq``
train steps: hard-copy all three target networks, refresh ``reward_scale``
from the buffer (old value moves to ``target_reward_scale``), then run
``target_update_freq`` encoder-only updates. Net effect: 1 RL + amortized
1 encoder update per env step.

Encoder loss — horizon unroll with cumulative termination masking
(``prev_not_done``); dyn MSE + two-hot reward CE + done MSE, weights
(1, 0.1, 0.1). Done loss only if the buffer has seen any terminal
transition (``env_terminates``).

RL update — n-step target with twin-Q min and target-policy smoothing
noise (discrete-halved); smooth-L1 value loss with grad clip; policy loss
= -Q + pre-activation penalty; LAP priority = max |Q - target| clamped to
min priority, raised to alpha (no IS correction).

Resume policy (PROTOCOL.md decision 2): durable checkpoints are
model_only — exact resume is REJECTED (environment state is not
serialized).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from buffer import MRQReplayBuffer
from networks import StateEncoder, PolicyNet, TwinQCritic, TwoHotReward, shift_augment


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor,
                      discount: float):
    """Sum r_t * (gamma * not_done)^k over the horizon; returns (ms, scale)."""
    ms_reward = torch.zeros(reward.shape[0], 1, device=reward.device)
    scale = torch.ones(reward.shape[0], 1, device=reward.device)
    for i in range(reward.shape[1]):
        ms_reward = ms_reward + scale * reward[:, i]
        scale = scale * discount * not_done[:, i]
    return ms_reward, scale


def realign(x: torch.Tensor, discrete: bool) -> torch.Tensor:
    if discrete:
        return F.one_hot(x.argmax(1), x.shape[1]).float()
    return x.clamp(-1, 1)


class MRQAgent:
    def __init__(self, obs_shape, action_dim, discrete, device, *,
                 history=4, algorithm: dict):
        self.hp = dict(algorithm)
        self.device = device
        self.discrete = bool(discrete)
        self.action_dim = action_dim
        self.pixel_obs = len(obs_shape) == 3

        # Discrete actions live in [0,1] vs continuous [-1,1]: halve noises.
        self.exploration_noise = float(self.hp["exploration_noise" if not discrete
                                                else "exploration_noise_discrete"])
        if discrete:
            self.target_policy_noise = float(self.hp["target_policy_noise_discrete"])
            self.noise_clip = float(self.hp["target_policy_noise_clip_discrete"])
        else:
            self.target_policy_noise = float(self.hp["target_policy_noise"])
            self.noise_clip = float(self.hp["target_policy_noise_clip"])

        horizon = max(self.hp["enc_horizon"], self.hp["q_horizon"])
        self.replay_buffer = MRQReplayBuffer(
            obs_shape, action_dim, discrete, device,
            history=history, horizon=horizon,
            max_size=self.hp["buffer_size"], batch_size=self.hp["batch_size"],
            prioritized=True, initial_priority=self.hp["min_priority"],
        )
        self.state_shape = self.replay_buffer.state_shape

        obs_channels = obs_shape[0] * history
        self.encoder = StateEncoder(
            obs_channels, action_dim, self.pixel_obs,
            num_bins=self.hp["twohot_num_bins"], zs_dim=self.hp["zs_dim"],
            za_dim=self.hp["za_dim"], zsa_dim=self.hp["zsa_dim"],
            hdim=self.hp["hdim"],
            image_size=(obs_shape[1] if self.pixel_obs else 84),
        ).to(device)
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.hp["encoder_lr"],
            weight_decay=self.hp["weight_decay"])
        self.encoder_target = copy.deepcopy(self.encoder)

        self.policy = PolicyNet(
            action_dim, discrete, gumbel_tau=self.hp["gumbel_tau"],
            zs_dim=self.hp["zs_dim"], hdim=self.hp["hdim"],
        ).to(device)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.hp["policy_lr"],
            weight_decay=self.hp["weight_decay"])
        self.policy_target = copy.deepcopy(self.policy)

        self.value = TwinQCritic(
            zsa_dim=self.hp["zsa_dim"], hdim=self.hp["hdim"]).to(device)
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=self.hp["value_lr"],
            weight_decay=self.hp["weight_decay"])
        self.value_target = copy.deepcopy(self.value)

        self.two_hot = TwoHotReward(
            lower=self.hp["twohot_lower"], upper=self.hp["twohot_upper"],
            num_bins=self.hp["twohot_num_bins"], device=device,
        )

        self.reward_scale = 1.0
        self.target_reward_scale = 0.0
        self.training_steps = 0

    # ------------------------------------------------------------- acting

    def select_action(self, state, evaluate: bool = False):
        if (self.replay_buffer.size
                <= self.hp["buffer_size_before_training"] and not evaluate):
            return None  # caller samples uniformly from the action space
        with torch.no_grad():
            state_t = torch.as_tensor(
                np.asarray(state), dtype=torch.float32, device=self.device
            ).reshape(-1, *self.state_shape)
            zs = self.encoder.zs(state_t)
            if evaluate:
                # PROTOCOL.md decision 1: deterministic argmax/tanh evaluation
                action = self.policy.act(zs, deterministic=True)
            else:
                action = self.policy.act(zs)
                action = action + torch.randn_like(action) * self.exploration_noise
            if self.discrete:
                return int(action.argmax(dim=-1).item())
            return action.clamp(-1, 1).cpu().numpy().flatten()

    def observe(self, state, action, next_state, reward, terminated, truncated):
        self.replay_buffer.add(state, action, next_state, reward,
                               terminated, truncated)

    # ------------------------------------------------------------ training

    def train_step(self) -> dict:
        if self.replay_buffer.size <= self.hp["buffer_size_before_training"]:
            return {}
        horizon = max(self.hp["enc_horizon"], self.hp["q_horizon"])
        if not self.replay_buffer.can_sample(horizon):
            return {}

        self.training_steps += 1
        logs: dict = {}

        if (self.training_steps - 1) % self.hp["target_update_freq"] == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()
            for _ in range(self.hp["encoder_burst_steps"]):
                enc_logs = self._train_encoder()
            # only the last burst tensor is kept (amortized loss of the burst)
            logs.update({f"encoder/{k}": v for k, v in enc_logs.items()})

        rl_logs = self._train_rl()
        logs.update({f"rl/{k}": v for k, v in rl_logs.items()})
        # lazily convert tensors to floats exactly once per step for logging
        return {k: (float(v.item()) if torch.is_tensor(v) else v)
                for k, v in logs.items()}

    def _maybe_augment(self, state, next_state):
        if not self.pixel_obs:
            return state, next_state
        # Group both tensors, shift once, split back (per-image shift).
        shape = state.shape  # (B, [H,] C, h, w)
        if len(shape) == 4:
            both = torch.cat([state, next_state], dim=0)
            both = shift_augment(both, pad=self.hp["shift_augment_pad"])
            return both[: state.shape[0]], both[state.shape[0]:]
        # horizon variant: (B, H, C, h, w)
        b, h = shape[0], shape[1]
        flat_s = state.reshape(-1, *shape[2:])
        flat_ns = next_state.reshape(-1, *shape[2:])
        both = torch.cat([flat_s, flat_ns], dim=0)
        both = shift_augment(both, pad=self.hp["shift_augment_pad"])
        s = both[: flat_s.shape[0]].reshape(shape)
        ns = both[flat_s.shape[0]:].reshape(shape)
        return s, ns

    def _train_encoder(self) -> dict:
        state, action, next_state, reward, not_done = self.replay_buffer.sample(
            self.hp["enc_horizon"], include_intermediate=True)
        state, next_state = self._maybe_augment(state, next_state)

        with torch.no_grad():
            tgt_zs = self.encoder_target.zs(
                next_state.reshape(-1, *self.state_shape)
            ).reshape(state.shape[0], -1, self.hp["zs_dim"])

        pred_zs = self.encoder.zs(state[:, 0])
        prev_not_done = torch.ones_like(reward[:, 0])
        loss = torch.zeros((), device=self.device)
        for i in range(self.hp["enc_horizon"]):
            pred_d, pred_zs, pred_r = self.encoder.model_all(pred_zs, action[:, i])
            mask = prev_not_done
            dyn_loss = (F.mse_loss(pred_zs, tgt_zs[:, i], reduction="none")
                        * mask).mean()
            reward_loss = (self.two_hot.cross_entropy_loss(pred_r, reward[:, i])
                           * mask).mean()
            done_loss = torch.zeros((), device=self.device)
            if self.replay_buffer.env_terminates:
                tgt_done = 1.0 - not_done[:, i].reshape(-1, 1)
                done_loss = (F.mse_loss(pred_d, tgt_done, reduction="none")
                             * mask).mean()
            loss = (loss
                    + self.hp["encoder_loss_weight_dyn"] * dyn_loss
                    + self.hp["encoder_loss_weight_reward"] * reward_loss
                    + self.hp["encoder_loss_weight_done"] * done_loss)
            prev_not_done = not_done[:, i].reshape(-1, 1) * prev_not_done

        self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_optimizer.step()
        # keep the scalar on-device; only sync if the caller wants logs
        return {"loss_tensor": loss.detach()}

    def _train_rl(self) -> dict:
        state, action, next_state, reward, not_done = self.replay_buffer.sample(
            self.hp["q_horizon"], include_intermediate=False)
        state, next_state = self._maybe_augment(state, next_state)
        reward, term_discount = multi_step_reward(
            reward, not_done, self.hp["discount"])

        with torch.no_grad():
            next_zs = self.encoder_target.zs(next_state)
            noise = (torch.randn_like(action) * self.target_policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = realign(
                self.policy_target.act(next_zs) + noise, self.discrete)
            next_zsa = self.encoder_target(next_zs, next_action)
            q_target = self.value_target(next_zsa).min(1, keepdim=True).values
            q_target = (reward + term_discount * q_target
                        * self.target_reward_scale) / self.reward_scale

            zs = self.encoder.zs(state)
            zsa = self.encoder(zs, action)

        q = self.value(zsa)
        value_loss = F.smooth_l1_loss(q, q_target.expand(-1, 2))
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value.parameters(), self.hp["value_grad_clip"])
        self.value_optimizer.step()

        policy_action, pre_activ = self.policy(zs)
        q_policy = self.value(self.encoder(zs, policy_action))
        policy_loss = (-q_policy.mean()
                       + self.hp["pre_activation_penalty"] * pre_activ.pow(2).mean())
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        priority = (q - q_target.expand(-1, 2)).abs().max(1).values
        priority = priority.clamp(min=self.hp["min_priority"]).pow(
            self.hp["priority_alpha"])
        self.replay_buffer.update_priority(priority)

        # on-device scalars; synced lazily by the logging path only
        return {"value_loss_tensor": value_loss.detach(),
                "policy_loss_tensor": policy_loss.detach(),
                "q_mean_tensor": q.detach().mean(),
                "q_target_mean_tensor": q_target.detach().mean()}

    # ---------------------------------------------------------- checkpoints

    _STATE_KEYS = (
        "encoder", "encoder_target", "encoder_optimizer",
        "policy", "policy_target", "policy_optimizer",
        "value", "value_target", "value_optimizer",
    )

    def save_checkpoint(self, folder: str, durable: bool = False):
        import os
        os.makedirs(folder, exist_ok=True)
        for key in self._STATE_KEYS:
            torch.save(getattr(self, key).state_dict(), f"{folder}/{key}.pt")
        np.save(f"{folder}/agent_var.npy", {
            "reward_scale": self.reward_scale,
            "target_reward_scale": self.target_reward_scale,
            "training_steps": self.training_steps,
        })
        if durable:
            import random
            np.save(f"{folder}/rng_states.npy", {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            })
            self.replay_buffer.save(folder)

    def load_checkpoint(self, folder: str, durable: bool = False,
                        resume: bool = False):
        if resume:
            raise NotImplementedError(
                "model_only checkpoint cannot resume: ALE emulator state, "
                "current observation, and environment RNG are not serialized "
                "(PROTOCOL.md decision 2: resume policy is reject).")
        for key in self._STATE_KEYS:
            getattr(self, key).load_state_dict(
                torch.load(f"{folder}/{key}.pt", weights_only=True,
                           map_location=self.device))
        var = np.load(f"{folder}/agent_var.npy", allow_pickle=True).item()
        self.reward_scale = float(var["reward_scale"])
        self.target_reward_scale = float(var["target_reward_scale"])
        self.training_steps = int(var["training_steps"])
        if durable:
            self.replay_buffer.load(folder)
