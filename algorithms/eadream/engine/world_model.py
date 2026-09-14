"""EADream world-model objective, optimizer update, and open-loop diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from eadream.engine.distributions import EventAwareImageLoss, FocalBinaryDist
from eadream.engine.optimization import OptimizerBundle, make_adamw
from eadream.networks.decoder import EventDecoder, ImageDecoder
from eadream.networks.encoder import ImageEncoder
from eadream.networks.harmony import Harmonizer
from eadream.networks.heads import ScalarHead
from eadream.networks.rssm import RSSM, RSSMState, detach_state
from eadream.observation import validate_observation


@dataclass(frozen=True)
class WorldModelOutput:
    loss: Tensor
    posterior: RSSMState
    prior: RSSMState
    posterior_features: Tensor
    prior_features: Tensor
    per_timestep: Mapping[str, Tensor]
    debug_shapes: Mapping[str, object]
    metrics: Mapping[str, float]


def _state_at(state: RSSMState, index: int) -> RSSMState:
    return RSSMState(
        logit=state.logit[:, index],
        stoch=state.stoch[:, index],
        deter=state.deter[:, index],
    )


def _detach_mapping(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach() for name, value in values.items()}


class WorldModel(nn.Module):
    """Native EADream RSSM-OP world model for normalized Atari replay."""

    _BATCH_FIELDS = (
        "image",
        "event",
        "action",
        "reward",
        "discount",
        "is_first",
        "is_terminal",
    )

    def __init__(
        self,
        config: Mapping[str, Any],
        action_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model = config["model"]
        training = config["training"]
        events = config["events"]
        runtime = config.get("runtime", {})
        provenance = config.get("provenance", {})
        configured_action_dim = config.get("action_dim")
        action_dim = configured_action_dim if action_dim is None else action_dim
        if type(action_dim) is not int or action_dim <= 0:
            raise ValueError(
                "WorldModel requires a positive action_dim from the environment"
            )
        self.action_dim = action_dim

        self.encoder = ImageEncoder(
            depth=int(model["cnn_depth"]),
            attention=bool(model["encoder_attention"]),
            attention_kernel=int(model["encoder_attention_kernel"]),
            attention_ratio=int(model["encoder_attention_ratio"]),
            kernel_size=int(model["cnn_kernel"]),
        )
        self.rssm = RSSM(
            stoch=int(model["dyn_stoch"]),
            classes=int(model["dyn_discrete"]),
            deter=int(model["dyn_deter"]),
            hidden=int(model["dyn_hidden"]),
            action_dim=action_dim,
            embed_dim=self.encoder.output_dim,
            unimix=float(model["unimix_ratio"]),
            rec_depth=int(model["dyn_rec_depth"]),
        )
        feature_dim = (
            int(model["dyn_stoch"]) * int(model["dyn_discrete"])
            + int(model["dyn_deter"])
        )
        event_feature_dim = (
            2 * int(model["dyn_stoch"]) * int(model["dyn_discrete"])
            + int(model["dyn_deter"])
        )
        decoder_options = {
            "depth": int(model["cnn_depth"]),
            "kernel_size": int(model["cnn_kernel"]),
            "outscale": 1.0,
        }
        self.event_decoder = EventDecoder(
            event_feature_dim,
            **decoder_options,
        )
        self.image_decoder = ImageDecoder(
            feature_dim,
            output_offset=float(model["decoder_output_offset"]),
            **decoder_options,
        )
        head_options = {
            "units": int(model["units"]),
            "layers": int(model["head_layers"]),
        }
        self.reward_head = ScalarHead(
            feature_dim,
            distribution=str(model["scalar_distribution"]),
            bins=int(model["scalar_buckets"]),
            low=float(model["scalar_low"]),
            high=float(model["scalar_high"]),
            outscale=float(model["reward_outscale"]),
            **head_options,
        )
        self.continuation_head = ScalarHead(
            feature_dim,
            distribution="binary",
            outscale=float(model["continuation_outscale"]),
            **head_options,
        )
        self.harmonizers = nn.ModuleDict(
            {
                "rep": Harmonizer(),
                "image": Harmonizer(),
                "reward": Harmonizer(),
            }
        )
        self.image_objective = EventAwareImageLoss(
            attention_weight=float(model["image_attention_weight"]),
            event_pred_ratio=float(events["event_pred_ratio"]),
        )
        self.event_pred_ratio = float(events["event_pred_ratio"])
        self.event_focal_alpha = float(model["event_focal_alpha"])
        self.event_focal_gamma = float(model["event_focal_gamma"])
        self.kl_free = float(model["kl_free"])
        self.dyn_scale = float(model["dyn_scale"])
        self.rep_scale = float(model["rep_scale"])
        self.event_loss_scale = float(model["event_loss_scale"])
        self.discount = float(training["discount"])
        self.detect_anomaly = bool(training["detect_anomaly_world_model"])
        self.update_count = 0

        initial_device = torch.device(str(runtime.get("device", "cpu")))
        self.to(initial_device)
        self.optimizer: OptimizerBundle = make_adamw(
            self.parameters(),
            learning_rate=float(training["model_lr"]),
            clip_norm=float(training["model_grad_clip"]),
            declared_epsilon=float(training["optimizer_effective_eps"]),
            declared_weight_decay=float(
                provenance.get("declared_extra_weight_decay", 0.0)
            ),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _as_tensor(value: Tensor | np.ndarray, device: torch.device) -> Tensor:
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, device=device)

    def preprocess(
        self, batch: Mapping[str, Tensor | np.ndarray]
    ) -> dict[str, Tensor]:
        missing = set(self._BATCH_FIELDS) - set(batch)
        if missing:
            raise KeyError(f"world-model batch is missing {sorted(missing)}")
        data = {
            key: self._as_tensor(batch[key], self.device)
            for key in self._BATCH_FIELDS
        }
        data["image"] = data["image"].float().div(255.0)
        data["event"] = data["event"].float().div(255.0).unsqueeze(-1)
        data["action"] = data["action"].float()
        data["reward"] = data["reward"].float()
        data["discount"] = (
            data["discount"].float().unsqueeze(-1) * self.discount
        )
        data["is_first"] = data["is_first"].bool()
        data["is_terminal"] = data["is_terminal"].bool()
        data["cont"] = (1.0 - data["is_terminal"].float()).unsqueeze(-1)
        self._validate_preprocessed_shapes(data)
        metric_names = {
            "image": "image_loss_raw",
            "event": "event_loss_raw",
            "action": "kl_loss_raw",
            "reward": "reward_loss_raw",
            "discount": "cont_loss_raw",
        }
        for name, metric_name in metric_names.items():
            if not bool(torch.isfinite(data[name]).all()):
                self.optimizer.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite world-model metric {metric_name}=input:{name} "
                    f"at update {self.update_count}"
                )
        return data

    def preprocess_observation(
        self, observation: Mapping[str, Tensor | np.ndarray | bool]
    ) -> dict[str, Tensor]:
        observation = validate_observation(observation, copy=False)
        image = self._as_tensor(observation["image"], self.device).float().div(255.0)
        if image.shape == (64, 64, 3):
            image = image.unsqueeze(0)
        if image.ndim != 4 or tuple(image.shape[-3:]) != (64, 64, 3):
            raise ValueError("observation image must have shape [batch, 64, 64, 3]")
        result = {"image": image}
        event = self._as_tensor(observation["event"], self.device).float().div(255.0)
        if event.shape == (64, 64):
            event = event.unsqueeze(0)
        result["event"] = event.unsqueeze(-1)
        return result

    def _validate_preprocessed_shapes(self, data: Mapping[str, Tensor]) -> None:
        if data["image"].ndim != 5 or tuple(data["image"].shape[-3:]) != (
            64,
            64,
            3,
        ):
            raise ValueError("image must have shape [batch, time, 64, 64, 3]")
        batch_time = data["image"].shape[:2]
        expected = {
            "event": (*batch_time, 64, 64, 1),
            "action": (*batch_time, self.action_dim),
            "reward": batch_time,
            "discount": (*batch_time, 1),
            "is_first": batch_time,
            "is_terminal": batch_time,
            "cont": (*batch_time, 1),
        }
        for name, shape in expected.items():
            if tuple(data[name].shape) != tuple(shape):
                raise ValueError(f"{name} must have shape {tuple(shape)}")

    def predict_image(self, features: Tensor) -> Tensor:
        return self.image_decoder(features)

    def predict_event(
        self, prior: RSSMState, posterior: RSSMState
    ) -> tuple[Tensor, Tensor]:
        event_feature = torch.cat(
            (
                prior.stoch.flatten(-2),
                posterior.stoch.flatten(-2),
                posterior.deter,
            ),
            dim=-1,
        )
        return self.event_decoder(event_feature), event_feature

    def predict_reward(self, features: Tensor):
        return self.reward_head(features)

    def predict_continuation(self, features: Tensor):
        return self.continuation_head(features)

    def imagine_step(
        self, state: RSSMState, action: Tensor, sample: bool = True
    ) -> RSSMState:
        return self.rssm.img_step(state, action, sample=sample)

    def loss(
        self,
        batch: Mapping[str, Tensor | np.ndarray],
        *,
        sample: bool = True,
    ) -> WorldModelOutput:
        data = self.preprocess(batch)
        embedding = self.encoder(data["image"])
        posterior, prior = self.rssm.observe(
            embedding,
            data["action"],
            data["is_first"],
            sample=sample,
        )
        posterior_features = self.rssm.get_feat(posterior)
        prior_features = self.rssm.get_feat(prior)

        image_feature = torch.cat(
            (posterior_features[:, :1], prior_features[:, 1:]), dim=1
        )
        image_prediction = self.predict_image(image_feature)
        image_loss = self.image_objective(
            image_prediction, data["image"], data["event"]
        )

        event_prediction, event_feature = self.predict_event(prior, posterior)
        focal_loss = FocalBinaryDist(
            event_prediction,
            alpha=self.event_focal_alpha,
            gamma=self.event_focal_gamma,
        ).loss(data["event"])
        event_count = data["event"].sum(dim=(2, 3, 4))
        pixel_count = math.prod(data["event"].shape[2:])
        sparse_event = event_count < self.event_pred_ratio * pixel_count
        event_loss = torch.where(sparse_event, focal_loss, torch.zeros_like(focal_loss))

        reward_loss = -self.predict_reward(posterior_features).log_prob(data["reward"])
        continuation_loss = -self.predict_continuation(posterior_features).log_prob(
            data["cont"]
        )
        _, kl_value, dyn_loss, rep_loss = self.rssm.kl_loss(
            posterior,
            prior,
            free=self.kl_free,
            dyn_scale=self.dyn_scale,
            rep_scale=self.rep_scale,
        )
        kl_base = self.dyn_scale * dyn_loss + self.rep_scale * rep_loss

        image_loss = image_loss.clone()
        event_loss = event_loss.clone()
        image_loss[:, 0] = 0.0
        event_loss[:, 0] = 0.0

        kl_term = self.harmonizers["rep"](kl_base).mean()
        image_scaled = self.harmonizers["image"](image_loss.mean())
        reward_scaled = self.harmonizers["reward"](reward_loss.mean())
        event_scaled = self.event_loss_scale * event_loss.mean()
        continuation_scaled = continuation_loss.mean()
        total = (
            kl_term
            + image_scaled
            + reward_scaled
            + event_scaled
            + continuation_scaled
        )

        rep_coefficient = self.harmonizers["rep"].coefficient()
        metrics = {
            "image_loss_raw": float(image_loss.mean().detach().cpu()),
            "reward_loss_raw": float(reward_loss.mean().detach().cpu()),
            "event_loss_raw": float(event_loss.mean().detach().cpu()),
            "cont_loss_raw": float(continuation_loss.mean().detach().cpu()),
            "kl_loss_raw": float(kl_base.mean().detach().cpu()),
            "image_loss_scaled": float(image_scaled.detach().cpu()),
            "reward_loss_scaled": float(reward_scaled.detach().cpu()),
            "event_loss_scaled": float(event_scaled.detach().cpu()),
            "cont_loss_scaled": float(continuation_scaled.detach().cpu()),
            "kl_loss_scaled": float(kl_term.detach().cpu()),
            "model_loss": float(total.detach().cpu()),
            "kl_value": float(kl_value.mean().detach().cpu()),
            "dyn_loss_raw": float(dyn_loss.mean().detach().cpu()),
            "rep_loss_raw": float(rep_loss.mean().detach().cpu()),
            "dyn_loss_scaled": float(
                (self.dyn_scale * dyn_loss.mean()).detach().cpu()
            ),
            "rep_loss_scaled": float(
                (self.rep_scale * rep_loss.mean()).detach().cpu()
            ),
            "dyn_loss_adjusted": float(
                (dyn_loss.mean() * rep_coefficient).detach().cpu()
            ),
            "rep_loss_adjusted": float(
                (rep_loss.mean() * rep_coefficient).detach().cpu()
            ),
            "posterior_entropy": float(
                self.rssm.get_dist(posterior).entropy().mean().detach().cpu()
            ),
            "prior_entropy": float(
                self.rssm.get_dist(prior).entropy().mean().detach().cpu()
            ),
            "event_pixel_ratio": float(
                (event_count / pixel_count).mean().detach().cpu()
            ),
            "sparse_event_fraction": float(sparse_event.float().mean().detach().cpu()),
            "harmony_rep": float(rep_coefficient.detach().cpu()),
            "harmony_image": float(
                self.harmonizers["image"].coefficient().detach().cpu()
            ),
            "harmony_reward": float(
                self.harmonizers["reward"].coefficient().detach().cpu()
            ),
            "model_grad_norm": 0.0,
            "update_duration_seconds": 0.0,
            "cuda_peak_allocated_bytes": 0.0,
            "update_count": float(self.update_count),
        }
        per_timestep = {
            "image": image_loss,
            "reward": reward_loss,
            "event": event_loss,
            "cont": continuation_loss,
            "kl": kl_base,
            "kl_value": kl_value,
            "dyn": dyn_loss,
            "rep": rep_loss,
        }
        return WorldModelOutput(
            loss=total,
            posterior=posterior,
            prior=prior,
            posterior_features=posterior_features,
            prior_features=prior_features,
            per_timestep=per_timestep,
            debug_shapes={
                "embedding": embedding.shape,
                "image_feature": image_feature.shape,
                "event_feature": event_feature.shape,
                "event_feature_last": event_feature.shape[-1],
                "image_prediction": image_prediction.shape,
                "event_prediction": event_prediction.shape,
            },
            metrics=metrics,
        )

    def _check_finite_metrics(self, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            if not math.isfinite(value):
                self.optimizer.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite world-model metric {name}={value!r} at update "
                    f"{self.update_count}"
                )

    def update(
        self, batch: Mapping[str, Tensor | np.ndarray]
    ) -> WorldModelOutput:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.autograd.set_detect_anomaly(self.detect_anomaly):
            output = self.loss(batch, sample=True)
            self._check_finite_metrics(output.metrics)
            try:
                gradient_norm = self.optimizer.step(
                    output.loss, self.optimizer.parameters
                )
            except FloatingPointError as error:
                raise FloatingPointError(
                    "non-finite world-model metric model_grad_norm="
                    f"{error} at update {self.update_count}"
                ) from error
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocation = float(torch.cuda.max_memory_allocated(self.device))
        else:
            peak_allocation = 0.0
        duration = time.perf_counter() - started
        self.update_count += 1
        metrics = dict(output.metrics)
        metrics.update(
            {
                "model_grad_norm": float(gradient_norm),
                "update_duration_seconds": float(duration),
                "cuda_peak_allocated_bytes": peak_allocation,
                "update_count": float(self.update_count),
            }
        )
        self._check_finite_metrics(metrics)
        return WorldModelOutput(
            loss=output.loss.detach(),
            posterior=detach_state(output.posterior),
            prior=detach_state(output.prior),
            posterior_features=output.posterior_features.detach(),
            prior_features=output.prior_features.detach(),
            per_timestep=_detach_mapping(output.per_timestep),
            debug_shapes=dict(output.debug_shapes),
            metrics=metrics,
        )

    @staticmethod
    def _copy_batch(
        batch: Mapping[str, Tensor | np.ndarray]
    ) -> dict[str, Tensor | np.ndarray]:
        return {
            name: value.clone() if isinstance(value, Tensor) else value.copy()
            for name, value in batch.items()
        }

    @torch.inference_mode()
    def video_pred(self, batch: Mapping[str, Tensor | np.ndarray]) -> Tensor:
        copied = self._copy_batch(batch)
        data = self.preprocess(copied)
        if data["image"].shape[1] < 5:
            raise ValueError("open-loop prediction requires at least five timesteps")
        batch_size = min(6, data["image"].shape[0])
        image = data["image"][:batch_size]
        action = data["action"][:batch_size]
        is_first = data["is_first"][:batch_size]
        embedding = self.encoder(image)
        posterior, prior = self.rssm.observe(
            embedding, action, is_first, sample=True
        )
        posterior_features = self.rssm.get_feat(posterior)
        prior_features = self.rssm.get_feat(prior)
        reconstruction_features = torch.cat(
            (posterior_features[:, :1], prior_features[:, 1:5]), dim=1
        )
        reconstruction = self.predict_image(reconstruction_features)

        if image.shape[1] > 5:
            imagined = self.rssm.imagine(
                action[:, 5:], _state_at(posterior, 4), sample=True
            )
            open_loop = self.predict_image(self.rssm.get_feat(imagined))
            model = torch.cat((reconstruction, open_loop), dim=1)
        else:
            model = reconstruction

        event_prediction, _ = self.predict_event(prior, posterior)
        predicted_event = (event_prediction[:, 1:] > 0.5).to(image.dtype)
        initial_event = torch.zeros_like(predicted_event[:, :1])
        event_panel = torch.cat((initial_event, predicted_event), dim=1).repeat(
            1, 1, 1, 1, 3
        )
        error = (model - image + 1.0) / 2.0
        return torch.cat((image, model, error, event_panel), dim=2)

    video_prediction = video_pred


__all__ = ["WorldModel", "WorldModelOutput"]
