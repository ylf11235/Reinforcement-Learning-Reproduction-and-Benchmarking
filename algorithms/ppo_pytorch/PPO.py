"""The upstream PPO-PyTorch discrete algorithm adapted for Atari pixels.

The update intentionally keeps the upstream CartPole behavior: Monte-Carlo
returns, normalized returns, value-based advantages, one full-batch update for
K epochs, and a categorical policy. Only the MLP encoder is replaced with a
small Nature-style convolutional encoder so Atari observations are usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class RolloutBuffer:
    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    logprobs: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    state_values: list[torch.Tensor] = field(default_factory=list)
    is_terminals: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class NatureEncoder(nn.Module):
    def __init__(self, state_shape: tuple[int, int, int]) -> None:
        super().__init__()
        channels, height, width = state_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            feature_shape = self.features(torch.zeros(1, channels, height, width)).shape
        self.flatten = nn.Flatten()
        self.projection = nn.Sequential(
            nn.Linear(int(np.prod(feature_shape[1:])), 512),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.flatten(self.features(observations)))


class ActorCritic(nn.Module):
    """Categorical actor and scalar critic with separate CNN encoders."""

    def __init__(self, state_shape: tuple[int, int, int], action_dim: int) -> None:
        super().__init__()
        if len(state_shape) != 3:
            raise ValueError(f"Atari state_shape must be (C,H,W), got {state_shape}")
        self.actor_encoder = NatureEncoder(tuple(state_shape))
        self.critic_encoder = NatureEncoder(tuple(state_shape))
        self.actor = nn.Linear(512, int(action_dim))
        self.critic = nn.Linear(512, 1)

    @staticmethod
    def _prepare(observations: torch.Tensor) -> torch.Tensor:
        observations = torch.as_tensor(observations)
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)
        if observations.ndim != 4:
            raise ValueError(f"expected (N,C,H,W) observations, got {tuple(observations.shape)}")
        return observations.float() / 255.0

    def evaluate_observation(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        observations = self._prepare(observations)
        return self.actor(self.actor_encoder(observations)), self.critic(self.critic_encoder(observations))

    def act(
        self, observation: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.evaluate_observation(observation)
        distribution = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        return int(action[0].item()), distribution.log_prob(action)[0].detach(), value[0].detach()

    def evaluate(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.evaluate_observation(observations)
        distribution = Categorical(logits=logits)
        actions = actions.long().reshape(-1)
        return distribution.log_prob(actions), values.reshape(-1), distribution.entropy()


class PPO:
    """Minimal discrete PPO matching the upstream CartPole defaults."""

    def __init__(
        self,
        state_shape: tuple[int, int, int],
        action_dim: int,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        K_epochs: int = 40,
        eps_clip: float = 0.2,
        *,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.eps_clip = float(eps_clip)
        self.K_epochs = int(K_epochs)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(tuple(state_shape), int(action_dim)).to(self.device)
        self.policy_old = ActorCritic(tuple(state_shape), int(action_dim)).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor_encoder.parameters(), "lr": lr_actor},
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic_encoder.parameters(), "lr": lr_critic},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.mse_loss = nn.MSELoss()

    def select_action(self, state: np.ndarray | torch.Tensor) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(state, device=self.device)
            action, logprob, state_value = self.policy_old.act(tensor)
        self.buffer.states.append(tensor.detach().cpu())
        self.buffer.actions.append(torch.tensor(action, dtype=torch.long))
        self.buffer.logprobs.append(logprob.cpu())
        self.buffer.state_values.append(state_value.cpu())
        return action

    def predict(self, state: np.ndarray | torch.Tensor, *, deterministic: bool = True) -> int:
        with torch.no_grad():
            tensor = torch.as_tensor(state, device=self.device)
            return self.policy_old.act(tensor, deterministic=deterministic)[0]

    def update(self) -> dict[str, float]:
        if not self.buffer.rewards:
            return {"loss": 0.0, "return_mean": 0.0}

        returns: list[float] = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = float(reward) + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        rewards = torch.tensor(returns, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)
        old_states = torch.stack(self.buffer.states).to(self.device)
        old_actions = torch.stack(self.buffer.actions).to(self.device)
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device)
        old_values = torch.stack(self.buffer.state_values).reshape(-1).to(self.device)
        advantages = rewards.detach() - old_values.detach()

        last_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.K_epochs):
            logprobs, state_values, entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surrogate = ratios * advantages
            loss = (
                -torch.min(surrogate, clipped)
                + self.value_coef * self.mse_loss(state_values, rewards)
                - self.entropy_coef * entropy
            )
            self.optimizer.zero_grad(set_to_none=True)
            last_loss = loss.mean()
            last_loss.backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        result = {"loss": float(last_loss.detach().cpu()), "return_mean": float(rewards.mean().cpu())}
        self.buffer.clear()
        return result

    def save(self, checkpoint_path: str | Path) -> None:
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.policy_old.load_state_dict(state)
        self.policy.load_state_dict(state)
