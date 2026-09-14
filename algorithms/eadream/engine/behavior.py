"""Imagination-based actor and value learning for native EADream."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eadream.networks.heads import ActorHead, ScalarHead
from eadream.networks.rssm import (
    RSSMState,
    detach_state,
    flatten_batch_time,
    stack_states,
)

from .math import RewardEMA, lambda_return
from .optimization import OptimizerBundle, make_adamw


@dataclass(frozen=True)
class ImaginedTrajectory:
    features: Tensor
    states: RSSMState
    actions: Tensor
    start_features: Tensor


@dataclass(frozen=True)
class BehaviorLoss:
    actor_loss: Tensor
    value_loss: Tensor
    trajectory: ImaginedTrajectory
    returns: Tensor
    weights: Tensor
    metrics: Mapping[str, float]


def _distribution_value(
    distribution: Any,
    name: str,
    features: Tensor,
) -> Tensor:
    value = getattr(distribution, name)
    result = value() if callable(value) else value
    if not isinstance(result, Tensor):
        raise TypeError(f"distribution {name} must be a tensor")
    expected_shape = features.shape[:-1]
    if result.shape == expected_shape:
        return result
    if result.shape == (*expected_shape, 1):
        return result.squeeze(-1)
    raise ValueError(
        f"distribution {name} value shape {tuple(result.shape)} must match "
        f"imagined feature prefix {tuple(expected_shape)} with at most one "
        "scalar event dimension"
    )


def _tensor_stats(tensor: Tensor, prefix: str) -> dict[str, float]:
    value = tensor.detach().float()
    quantiles = torch.quantile(
        value,
        torch.tensor((0.05, 0.95), device=value.device, dtype=value.dtype),
    )
    return {
        f"{prefix}_mean": float(value.mean().cpu()),
        f"{prefix}_std": float(value.std().cpu()),
        f"{prefix}_min": float(value.min().cpu()),
        f"{prefix}_max": float(value.max().cpu()),
        f"{prefix}_005": float(quantiles[0].cpu()),
        f"{prefix}_095": float(quantiles[1].cpu()),
    }


class ImagBehavior(nn.Module):
    """Source-compatible Atari REINFORCE learner over latent trajectories."""

    def __init__(self, config: Mapping[str, Any], world_model: nn.Module) -> None:
        super().__init__()
        self.config = config
        # The agent owns the world model separately. Registering it here would
        # duplicate checkpoint state and weaken actor/value optimizer ownership.
        object.__setattr__(self, "_world_model", world_model)
        model = config["model"]
        training = config["training"]
        provenance = config.get("provenance", {})

        if str(model["actor_distribution"]) != "onehot":
            raise ValueError("EADream Atari behavior requires a one-hot actor")
        if str(training["imag_gradient"]) != "reinforce":
            raise ValueError("EADream Atari behavior requires REINFORCE imagination")
        if int(training["slow_value_update"]) != 1:
            raise ValueError("EADream Atari behavior updates the slow value every step")

        feature_dim = int(model["dyn_stoch"]) * int(model["dyn_discrete"]) + int(
            model["dyn_deter"]
        )
        head_options = {
            "units": int(model["units"]),
            "layers": int(model["head_layers"]),
        }
        self.actor = ActorHead(
            feature_dim,
            int(config["action_dim"]),
            unimix=float(model["unimix_ratio"]),
            outscale=float(model["actor_outscale"]),
            **head_options,
        )
        self.value = ScalarHead(
            feature_dim,
            distribution=str(model["scalar_distribution"]),
            bins=int(model["scalar_buckets"]),
            low=float(model["scalar_low"]),
            high=float(model["scalar_high"]),
            outscale=float(model["value_outscale"]),
            **head_options,
        )
        self.slow_value = copy.deepcopy(self.value)
        self.slow_value.requires_grad_(False)

        device = getattr(world_model, "device", None)
        if device is None:
            try:
                device = next(world_model.parameters()).device
            except StopIteration:
                device = torch.device(str(config.get("runtime", {}).get("device", "cpu")))
        self.to(torch.device(device))

        declared_epsilon = provenance.get("declared_actor_value_eps", 1e-5)
        declared_weight_decay = provenance.get("declared_extra_weight_decay", 0.0)
        self.actor_optimizer: OptimizerBundle = make_adamw(
            self.actor.parameters(),
            learning_rate=float(training["actor_lr"]),
            clip_norm=float(training["actor_grad_clip"]),
            declared_epsilon=float(declared_epsilon),
            declared_weight_decay=float(declared_weight_decay),
        )
        self.value_optimizer: OptimizerBundle = make_adamw(
            self.value.parameters(),
            learning_rate=float(training["value_lr"]),
            clip_norm=float(training["value_grad_clip"]),
            declared_epsilon=float(declared_epsilon),
            declared_weight_decay=float(declared_weight_decay),
        )
        self.reward_ema = RewardEMA(
            device=torch.device(device), alpha=float(training["reward_ema_alpha"])
        )
        self.horizon = int(training["imag_horizon"])
        self.discount = float(training["discount"])
        self.lambda_ = float(training["lambda"])
        self.entropy_scale = float(training["actor_entropy"])
        self.slow_fraction = float(training["slow_value_fraction"])
        self.update_count = 0

    @property
    def world_model(self) -> nn.Module:
        return self._world_model

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def imagine(
        self,
        start: RSSMState,
        horizon: int,
        deterministic: bool,
    ) -> ImaginedTrajectory:
        if horizon < 2:
            raise ValueError("imagination horizon must be at least two")
        state = detach_state(flatten_batch_time(start))
        features: list[Tensor] = []
        states: list[RSSMState] = []
        actions: list[Tensor] = []

        # Atari uses pure score-function gradients. Sampling still matches the
        # source, but no rollout graph is retained through actor or dynamics.
        with torch.no_grad():
            for _ in range(horizon):
                feature = self.world_model.rssm.get_feat(state)
                distribution = self.actor(feature.detach())
                action = (
                    distribution.mode()
                    if deterministic
                    else distribution.sample()
                )
                features.append(feature)
                states.append(state)
                actions.append(action)
                state = self.world_model.rssm.img_step(
                    state, action, sample=not deterministic
                )

        stacked_features = torch.stack(features)
        return ImaginedTrajectory(
            features=stacked_features,
            states=stack_states(states),
            actions=torch.stack(actions),
            start_features=stacked_features[0].detach(),
        )

    def loss(self, start: RSSMState) -> BehaviorLoss:
        trajectory = self.imagine(start, self.horizon, deterministic=False)
        features = trajectory.features.detach()

        with torch.no_grad():
            reward = _distribution_value(
                self.world_model.predict_reward(features), "mode", features
            )
            continuation_features = self.world_model.rssm.get_feat(trajectory.states)
            continuation = _distribution_value(
                self.world_model.predict_continuation(continuation_features),
                "mean",
                continuation_features,
            )
            discounts = self.discount * continuation
            target_values = self.value(features).mode()
            returns = lambda_return(
                reward[1:],
                target_values[:-1],
                discounts[1:],
                bootstrap=target_values[-1],
                lambda_=self.lambda_,
            )
            weights = torch.cumprod(
                torch.cat((torch.ones_like(discounts[:1]), discounts[:-1]), dim=0),
                dim=0,
            ).detach()
            offset, scale = self.reward_ema.update(returns.detach())
            normalized_target = (returns - offset) / scale

        policy = self.actor(features)
        scored_actions = F.one_hot(
            trajectory.actions.argmax(dim=-1), trajectory.actions.shape[-1]
        ).to(trajectory.actions.dtype)
        log_probability = policy.log_prob(scored_actions)[:-1]
        entropy = policy.entropy()
        with torch.no_grad():
            baseline = self.value(features[:-1]).mode()
            advantage = returns - baseline
        actor_terms = -weights[:-1] * log_probability * advantage.detach()
        entropy_terms = -self.entropy_scale * entropy[:-1]
        actor_loss = (actor_terms + entropy_terms).mean()

        value_distribution = self.value(features[:-1])
        with torch.no_grad():
            slow_target = self.slow_value(features[:-1]).mode()
        value_nll = -value_distribution.log_prob(
            returns.detach()
        ) - value_distribution.log_prob(slow_target.detach())
        value_loss = (weights[:-1] * value_nll).mean()

        with torch.no_grad():
            state_entropy = self.world_model.rssm.get_dist(
                trajectory.states
            ).entropy()
        metrics: dict[str, float] = {}
        metrics.update(_tensor_stats(returns, "return"))
        metrics.update(_tensor_stats(baseline, "value"))
        metrics.update(_tensor_stats(normalized_target, "normalized_target"))
        metrics.update(_tensor_stats(reward, "imag_reward"))
        metrics.update(_tensor_stats(continuation, "continuation"))
        metrics.update(_tensor_stats(weights, "weight"))
        metrics.update(_tensor_stats(trajectory.actions.argmax(dim=-1).float(), "imag_action"))
        metrics.update(
            {
                "actor_entropy": float(entropy.detach().mean().cpu()),
                "actor_entropy_term": float(entropy_terms.detach().mean().cpu()),
                "state_entropy": float(state_entropy.detach().mean().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "reward_ema_005": float(self.reward_ema.values[0].detach().cpu()),
                "reward_ema_095": float(self.reward_ema.values[1].detach().cpu()),
                "reward_ema_offset": float(offset.detach().cpu()),
                "reward_ema_scale": float(scale.detach().cpu()),
            }
        )
        return BehaviorLoss(
            actor_loss=actor_loss,
            value_loss=value_loss,
            trajectory=trajectory,
            returns=returns,
            weights=weights,
            metrics=metrics,
        )

    @torch.no_grad()
    def update_slow_target(self) -> None:
        for source, target in zip(
            self.value.parameters(), self.slow_value.parameters(), strict=True
        ):
            target.mul_(1.0 - self.slow_fraction).add_(
                source, alpha=self.slow_fraction
            )

    @staticmethod
    def _check_finite(metrics: Mapping[str, float], update_count: int) -> None:
        for name, value in metrics.items():
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite behavior metric {name}={value!r} at update "
                    f"{update_count}"
                )

    def update(self, start: RSSMState) -> Mapping[str, float]:
        self.update_slow_target()
        output = self.loss(start)
        self._check_finite(output.metrics, self.update_count)
        actor_gradient_norm = self.actor_optimizer.step(
            output.actor_loss, self.actor_optimizer.parameters
        )
        value_gradient_norm = self.value_optimizer.step(
            output.value_loss, self.value_optimizer.parameters
        )
        self.update_count += 1
        metrics = dict(output.metrics)
        metrics.update(
            {
                "actor_grad_norm": actor_gradient_norm,
                "value_grad_norm": value_gradient_norm,
                "behavior_update_count": float(self.update_count),
            }
        )
        self._check_finite(metrics, self.update_count)
        return metrics


__all__ = ["BehaviorLoss", "ImagBehavior", "ImaginedTrajectory"]
