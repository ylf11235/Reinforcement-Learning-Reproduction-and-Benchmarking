"""Local EADream learner and recurrent inference state."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from eadream.interfaces import RecurrentStatePayload, validate_recurrent_payload
from eadream.networks.rssm import RSSMState
from eadream.observation import validate_observation

from .behavior import ImagBehavior
from .world_model import WorldModel


_OBSERVATION_CONTRACT_HASH = hashlib.sha256(
    b"image:uint8[64,64,3];event:uint8[64,64];is_first:bool;is_terminal:bool"
).hexdigest()


@dataclass
class AgentState:
    """Process-local RSSM state retained between policy calls."""

    latent: RSSMState
    previous_action: Tensor

    def to_payload(self) -> RecurrentStatePayload:
        def array(tensor: Tensor) -> np.ndarray:
            return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)

        return RecurrentStatePayload(
            logit=array(self.latent.logit),
            stoch=array(self.latent.stoch),
            deter=array(self.latent.deter),
            previous_action=array(self.previous_action),
        )

    @classmethod
    def from_payload(
        cls, payload: RecurrentStatePayload, *, device: torch.device
    ) -> "AgentState":
        validate_recurrent_payload(payload)
        state = RSSMState(
            logit=torch.from_numpy(payload.logit).to(device=device),
            stoch=torch.from_numpy(payload.stoch).to(device=device),
            deter=torch.from_numpy(payload.deter).to(device=device),
        )
        previous_action = torch.from_numpy(payload.previous_action).to(device=device)
        return cls(latent=state, previous_action=previous_action)


def _as_device(value: str | torch.device | None, config: Mapping[str, Any]) -> torch.device:
    candidate = value if value is not None else config.get("runtime", {}).get("device", "cpu")
    device = torch.device(str(candidate))
    if device.type == "cuda" and not torch.cuda.is_available():
        if bool(config.get("runtime", {}).get("require_cuda", False)):
            raise RuntimeError("EADream requires CUDA but no CUDA device is available")
        return torch.device("cpu")
    return device


class EADreamAgent(nn.Module):
    """World model, behavior learner, and direct local policy path.

    Construction deliberately happens on CPU.  The complete module is moved to
    the selected device only after all source initializers and optimizers exist.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        action_dim: int | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = copy.deepcopy(dict(config))
        configured_action_dim = self.config.get("action_dim") if action_dim is None else action_dim
        if type(configured_action_dim) is not int or configured_action_dim <= 0:
            raise ValueError("EADreamAgent requires a positive action_dim")
        self.action_dim = configured_action_dim
        self.config["action_dim"] = configured_action_dim
        self.device_target = _as_device(device, self.config)

        # Runtime device is overridden only while constructing.  This preserves
        # source CPU RNG/initializer order before the final module migration.
        cpu_config = copy.deepcopy(self.config)
        cpu_config.setdefault("runtime", {})["device"] = "cpu"
        cpu_config["runtime"]["require_cuda"] = False
        self.world_model = WorldModel(cpu_config, action_dim=configured_action_dim)
        self.behavior = ImagBehavior(cpu_config, self.world_model)
        self.to(torch.device("cpu"))
        if self.device_target.type != "cpu":
            self.to(self.device_target)
        self.policy_version = 0
        configured_meanings = self.config.get("action_meanings")
        self.action_meanings = (
            tuple(configured_meanings)
            if configured_meanings is not None
            else tuple(range(self.action_dim))
        )

    @property
    def device(self) -> torch.device:
        return self.world_model.device

    @staticmethod
    def _episode_start_tensor(value: object, *, device: torch.device) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(device=device, dtype=torch.bool)
        else:
            tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        if tensor.ndim != 1 or tensor.shape[0] != 1:
            raise ValueError("phase-one predict supports batch size one")
        return tensor

    @torch.no_grad()
    def policy(
        self,
        observation: Mapping[str, np.ndarray | Tensor | bool],
        state: AgentState | None,
        *,
        episode_start: bool | np.ndarray | Tensor,
        deterministic: bool,
    ) -> tuple[np.ndarray, AgentState]:
        observation = validate_observation(observation, copy=False)
        data = self.world_model.preprocess_observation(observation)
        embed = self.world_model.encoder(data["image"])
        start = self._episode_start_tensor(episode_start, device=self.device)
        latent = None if state is None else state.latent
        previous = None if state is None else state.previous_action
        if state is not None:
            for tensor in (latent.logit, latent.stoch, latent.deter, previous):
                if tensor.device != self.device:
                    raise ValueError("AgentState device does not match the local agent")
            if previous.shape != (1, self.action_dim):
                raise ValueError("AgentState previous_action has the wrong shape")
        posterior, _ = self.world_model.rssm.obs_step(
            latent,
            previous,
            embed,
            is_first=start,
            sample=True,
        )
        feature = self.world_model.rssm.get_feat(posterior)
        action_dist = self.behavior.actor(feature)
        action = action_dist.mode() if bool(deterministic) else action_dist.sample()
        action_tensor = action[0].to(dtype=torch.float32)
        result = np.ascontiguousarray(action_tensor.cpu().numpy(), dtype=np.float32)
        return result, AgentState(latent=posterior, previous_action=action)

    def train_batch(self, batch: Mapping[str, np.ndarray | Tensor]) -> Mapping[str, float]:
        model_output = self.world_model.update(batch)
        behavior_metrics = self.behavior.update(model_output.posterior)
        self.policy_version += 1
        metrics = dict(model_output.metrics)
        metrics.update(behavior_metrics)
        return metrics

    def inference_state_dict(self) -> dict[str, Any]:
        """Return the phase-one policy payload; checkpoint persistence is Task 10."""
        return {
            "schema_version": 1,
            "encoder": copy.deepcopy(self.world_model.encoder.state_dict()),
            "rssm": copy.deepcopy(self.world_model.rssm.state_dict()),
            "actor": copy.deepcopy(self.behavior.actor.state_dict()),
            "action_dim": self.action_dim,
            "policy_version": self.policy_version,
            "action_meanings": self.action_meanings,
            "observation_contract_hash": _OBSERVATION_CONTRACT_HASH,
        }


__all__ = ["AgentState", "EADreamAgent"]
