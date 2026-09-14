"""Discrete recurrent state-space model used by native EADream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from eadream.engine.distributions import OneHotDist

from .initialization import source_uniform_init, source_weight_init


@dataclass(frozen=True)
class RSSMState:
    logit: Tensor
    stoch: Tensor
    deter: Tensor


def stack_states(states: Sequence[RSSMState], dim: int = 0) -> RSSMState:
    if not states:
        raise ValueError("cannot stack an empty state sequence")
    return RSSMState(
        logit=torch.stack([state.logit for state in states], dim=dim),
        stoch=torch.stack([state.stoch for state in states], dim=dim),
        deter=torch.stack([state.deter for state in states], dim=dim),
    )


def detach_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        logit=state.logit.detach(),
        stoch=state.stoch.detach(),
        deter=state.deter.detach(),
    )


def flatten_batch_time(state: RSSMState) -> RSSMState:
    def flatten(tensor: Tensor) -> Tensor:
        if tensor.ndim < 2:
            raise ValueError("state tensors need batch and time dimensions")
        return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])

    return RSSMState(
        logit=flatten(state.logit),
        stoch=flatten(state.stoch),
        deter=flatten(state.deter),
    )


class _IndependentOneHot:
    """OneHotDist with the stochastic-variable axis treated as an event."""

    def __init__(self, logits: Tensor, unimix: float) -> None:
        self.base_dist = OneHotDist(logits=logits, unimix_ratio=unimix)
        self.logits = self.base_dist.logits
        self.probs = self.base_dist.probs

    def sample(self) -> Tensor:
        return self.base_dist.sample()

    def mode(self) -> Tensor:
        return self.base_dist.mode()

    def log_prob(self, value: Tensor) -> Tensor:
        return self.base_dist.log_prob(value).sum(dim=-1)

    def entropy(self) -> Tensor:
        return self.base_dist.entropy().sum(dim=-1)


class NormalizedGRUCell(nn.Module):
    """The source single-projection, LayerNorm GRU cell."""

    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.linear = nn.Linear(input_dim + state_dim, 3 * state_dim, bias=False)
        self.norm = nn.LayerNorm(3 * state_dim, eps=1e-3)
        self.apply(source_weight_init)

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        reset, candidate, update = self.norm(
            self.linear(torch.cat((inputs, state), dim=-1))
        ).chunk(3, dim=-1)
        candidate = torch.tanh(torch.sigmoid(reset) * candidate)
        update = torch.sigmoid(update - 1.0)
        return update * candidate + (1.0 - update) * state


class RSSM(nn.Module):
    """Source-compatible categorical RSSM for the formal Atari profile."""

    def __init__(
        self,
        stoch: int = 64,
        classes: int = 32,
        deter: int = 512,
        hidden: int = 512,
        action_dim: int | None = None,
        embed_dim: int = 4096,
        unimix: float = 0.01,
        rec_depth: int = 1,
    ) -> None:
        super().__init__()
        dimensions = (stoch, classes, deter, hidden, embed_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("RSSM dimensions must be positive")
        if action_dim is None or action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if rec_depth <= 0:
            raise ValueError("rec_depth must be positive")
        if not 0.0 <= unimix <= 1.0:
            raise ValueError("unimix must be in [0, 1]")
        self.stoch = int(stoch)
        self.classes = int(classes)
        self.deter = int(deter)
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.embed_dim = int(embed_dim)
        self.unimix = float(unimix)
        self.rec_depth = int(rec_depth)

        self.input_projection = nn.Sequential(
            nn.Linear(self.stoch * self.classes + self.action_dim, self.hidden, bias=False),
            nn.LayerNorm(self.hidden, eps=1e-3),
            nn.SiLU(),
        )
        self.input_projection.apply(source_weight_init)
        self.gru = NormalizedGRUCell(self.hidden, self.deter)
        self.prior_projection = nn.Sequential(
            nn.Linear(self.deter, self.hidden, bias=False),
            nn.LayerNorm(self.hidden, eps=1e-3),
            nn.SiLU(),
        )
        self.prior_projection.apply(source_weight_init)
        self.posterior_projection = nn.Sequential(
            nn.Linear(self.deter + self.embed_dim, self.hidden, bias=False),
            nn.LayerNorm(self.hidden, eps=1e-3),
            nn.SiLU(),
        )
        self.posterior_projection.apply(source_weight_init)
        self.prior_statistics = nn.Linear(self.hidden, self.stoch * self.classes)
        self.prior_statistics.apply(source_uniform_init(1.0))
        self.posterior_statistics = nn.Linear(self.hidden, self.stoch * self.classes)
        self.posterior_statistics.apply(source_uniform_init(1.0))
        self.initial_deter = nn.Parameter(torch.zeros(1, self.deter))

    def _dist(self, logit: Tensor) -> _IndependentOneHot:
        return _IndependentOneHot(logit, self.unimix)

    def get_dist(self, state: RSSMState) -> _IndependentOneHot:
        return self._dist(state.logit)

    def _prior_hidden(self, deter: Tensor) -> Tensor:
        return self.prior_projection(deter)

    def _prior_stats(self, hidden: Tensor) -> Tensor:
        logits = self.prior_statistics(hidden)
        return logits.reshape(*logits.shape[:-1], self.stoch, self.classes)

    def _posterior_hidden(self, deter: Tensor, embed: Tensor) -> Tensor:
        return self.posterior_projection(torch.cat((deter, embed), dim=-1))

    def _posterior_stats(self, hidden: Tensor) -> Tensor:
        logits = self.posterior_statistics(hidden)
        return logits.reshape(*logits.shape[:-1], self.stoch, self.classes)

    def initial(self, batch_size: int, *, device: torch.device) -> RSSMState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        deter = torch.tanh(self.initial_deter).to(device=device).expand(batch_size, -1)
        initial_logit = torch.zeros(
            batch_size,
            self.stoch,
            self.classes,
            device=device,
            dtype=deter.dtype,
        )
        derived_logit = self._prior_stats(self._prior_hidden(deter))
        stoch = self._dist(derived_logit).mode()
        return RSSMState(logit=initial_logit, stoch=stoch, deter=deter)

    def get_feat(self, state: RSSMState) -> Tensor:
        flattened = state.stoch.reshape(*state.stoch.shape[:-2], self.stoch * self.classes)
        return torch.cat((flattened, state.deter), dim=-1)

    def img_step(
        self,
        previous: RSSMState,
        previous_action: Tensor,
        sample: bool = True,
    ) -> RSSMState:
        flattened = previous.stoch.reshape(
            *previous.stoch.shape[:-2], self.stoch * self.classes
        )
        hidden = self.input_projection(torch.cat((flattened, previous_action), dim=-1))
        recurrent_input = hidden
        for _ in range(self.rec_depth):
            deter = self.gru(recurrent_input, previous.deter)
            recurrent_input = deter
        logits = self._prior_stats(self._prior_hidden(deter))
        distribution = self._dist(logits)
        stochastic = distribution.sample() if sample else distribution.mode()
        return RSSMState(logit=logits, stoch=stochastic, deter=deter)

    @staticmethod
    def _select_rows(current: Tensor, initial: Tensor, mask: Tensor) -> Tensor:
        shaped_mask = mask.reshape(mask.shape + (1,) * (current.ndim - mask.ndim))
        return torch.where(shaped_mask, initial, current)

    def obs_step(
        self,
        previous: RSSMState | None,
        previous_action: Tensor | None,
        embed: Tensor,
        is_first: Tensor,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        batch_size = embed.shape[0]
        mask = is_first.to(device=embed.device, dtype=torch.bool).reshape(batch_size)
        if previous is None:
            previous = self.initial(batch_size, device=embed.device)
            action = torch.zeros(
                batch_size,
                self.action_dim,
                device=embed.device,
                dtype=embed.dtype,
            )
        else:
            action = (
                torch.zeros(
                    batch_size,
                    self.action_dim,
                    device=embed.device,
                    dtype=embed.dtype,
                )
                if previous_action is None
                else previous_action
            )
            if torch.any(mask):
                fresh = self.initial(batch_size, device=embed.device)
                previous = RSSMState(
                    logit=self._select_rows(previous.logit, fresh.logit, mask),
                    stoch=self._select_rows(previous.stoch, fresh.stoch, mask),
                    deter=self._select_rows(previous.deter, fresh.deter, mask),
                )
                action = torch.where(mask[:, None], torch.zeros_like(action), action)

        # Deliberately omit sample=sample: source obs_step always samples its prior.
        prior = self.img_step(previous, action)
        logits = self._posterior_stats(self._posterior_hidden(prior.deter, embed))
        distribution = self._dist(logits)
        stochastic = distribution.sample() if sample else distribution.mode()
        post = RSSMState(logit=logits, stoch=stochastic, deter=prior.deter)
        return post, prior

    def observe(
        self,
        embed: Tensor,
        action: Tensor,
        is_first: Tensor,
        state: RSSMState | None = None,
        sample: bool = True,
    ) -> tuple[RSSMState, RSSMState]:
        if embed.ndim != 3 or action.ndim != 3 or is_first.ndim != 2:
            raise ValueError("observe expects [batch, time, ...] tensors")
        if embed.shape[:2] != action.shape[:2] or embed.shape[:2] != is_first.shape:
            raise ValueError("observe batch/time dimensions differ")
        posts: list[RSSMState] = []
        priors: list[RSSMState] = []
        current = state
        for index in range(embed.shape[1]):
            post, prior = self.obs_step(
                current,
                action[:, index],
                embed[:, index],
                is_first[:, index],
                sample=sample,
            )
            posts.append(post)
            priors.append(prior)
            current = post
        return stack_states(posts, dim=1), stack_states(priors, dim=1)

    def imagine(
        self, action: Tensor, state: RSSMState, sample: bool = True
    ) -> RSSMState:
        priors: list[RSSMState] = []
        current = state
        for index in range(action.shape[1]):
            current = self.img_step(current, action[:, index], sample=sample)
            priors.append(current)
        return stack_states(priors, dim=1)

    imagine_with_action = imagine

    def _categorical_kl(self, post: Tensor, prior: Tensor) -> Tensor:
        post_dist = OneHotDist(logits=post, unimix_ratio=self.unimix)
        prior_dist = OneHotDist(logits=prior, unimix_ratio=self.unimix)
        return torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=-1)

    def kl_loss(
        self,
        post: RSSMState,
        prior: RSSMState,
        free: float = 1.0,
        dyn_scale: float = 0.5,
        rep_scale: float = 0.1,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        rep_value = self._categorical_kl(post.logit, prior.logit.detach())
        dyn_value = self._categorical_kl(post.logit.detach(), prior.logit)
        value = rep_value
        rep_loss = torch.clamp(rep_value, min=float(free))
        dyn_loss = torch.clamp(dyn_value, min=float(free))
        loss = float(dyn_scale) * dyn_loss + float(rep_scale) * rep_loss
        return loss, value, dyn_loss, rep_loss


__all__ = [
    "NormalizedGRUCell",
    "RSSM",
    "RSSMState",
    "detach_state",
    "flatten_batch_time",
    "stack_states",
]
