"""Networks for MRQ (clean-room): encoder, policy, twin Q, two-hot, shift aug.

Semantics follow the paper + reference behavior:
- All MLPs: 3 linear layers; LayerNorm-before-activation on hidden layers,
  linear output (no LN); Xavier-uniform init (relu gain), zero bias.
- Pixel path: 4 convs 32x3x3 strides (2,2,2,1) -> flatten 32*s*s
  (1568 at 84x84, 2592 at 96x96) -> linear -> LN+elu; input frames
  normalized as ``x/255 - 0.5``.
- ``za = elu(Linear(action))``; ``zsa = MLP(cat[zs, za])``;
  ``model head = Linear(zsa -> 1 + zs_dim + num_bins)`` split as
  ``[done, zs_pred, reward_logits]``.
- Policy activation: Gumbel-Softmax(tau) for discrete (stochastic training
  path), tanh for continuous. ``act(deterministic=True)`` uses logits argmax
  (discrete) or tanh (continuous).
- Two-hot bins: symexp-spaced ``sign(x)*(exp(|x|)-1)`` over [lower, upper];
  transform scatters mass onto the two adjacent bins by linear interpolation.
- shift_augment: random-translate via padded grid sample (replicate pad),
  per-image shift in [-pad, pad] pixels.
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight.data, nn.init.calculate_gain("relu"))
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def _ln_activ(x: torch.Tensor, activ) -> torch.Tensor:
    return activ(F.layer_norm(x, (x.shape[-1],)))


class MLP3(nn.Module):
    """3-layer MLP, LN+activation on the two hidden layers, linear output."""

    def __init__(self, input_dim: int, output_dim: int, hdim: int, activ: str = "elu"):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, output_dim)
        self.activ = getattr(F, activ)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ln_activ(self.l1(x), self.activ)
        x = _ln_activ(self.l2(x), self.activ)
        return self.l3(x)


class StateEncoder(nn.Module):
    def __init__(self, obs_channels: int, action_dim: int, pixel_obs: bool,
                 num_bins: int = 65, zs_dim: int = 512, za_dim: int = 256,
                 zsa_dim: int = 512, hdim: int = 512, activ: str = "elu",
                 image_size: int = 84):
        super().__init__()
        self.pixel_obs = pixel_obs
        self.zs_dim = zs_dim
        if pixel_obs:
            # strides (2,2,2,1) with 3x3 kernels: 84 -> 7 (1568 flat),
            # 96 -> 9 (2592 flat)
            s = image_size
            for stride in (2, 2, 2, 1):
                s = (s - 3) // stride + 1
            self.conv1 = nn.Conv2d(obs_channels, 32, 3, stride=2)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=1)
            self.conv_lin = nn.Linear(32 * s * s, zs_dim)
        else:
            self.state_mlp = MLP3(obs_channels, zs_dim, hdim, activ)
        self.za = nn.Linear(action_dim, za_dim)
        self.zsa = MLP3(zs_dim + za_dim, zsa_dim, hdim, activ)
        self.model = nn.Linear(zsa_dim, num_bins + zs_dim + 1)
        self.activ = getattr(F, activ)
        self.apply(_init_weights)

    # ---------------------------------------------------------------- zs
    def zs(self, state: torch.Tensor) -> torch.Tensor:
        if not self.pixel_obs:
            return _ln_activ(self.state_mlp(state), self.activ)
        x = state.float()
        if x.max() > 1.5:  # uint8-range input
            x = x / 255.0
        x = x - 0.5
        x = self.activ(self.conv1(x))
        x = self.activ(self.conv2(x))
        x = self.activ(self.conv3(x))
        x = self.activ(self.conv4(x)).reshape(x.shape[0], -1)
        return _ln_activ(self.conv_lin(x), self.activ)

    # -------------------------------------------------------------- zsa
    def forward(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        za = self.activ(self.za(action))
        return self.zsa(torch.cat([zs, za], dim=1))

    def model_all(self, zs: torch.Tensor, action: torch.Tensor):
        zsa = self.forward(zs, action)
        out = self.model(zsa)
        done = out[:, 0:1]
        zs_pred = out[:, 1:self.zs_dim + 1]
        reward_logits = out[:, self.zs_dim + 1:]
        return done, zs_pred, reward_logits


class PolicyNet(nn.Module):
    def __init__(self, action_dim: int, discrete: bool, gumbel_tau: float = 10,
                 zs_dim: int = 512, hdim: int = 512, activ: str = "relu"):
        super().__init__()
        self.discrete = discrete
        self.net = MLP3(zs_dim, action_dim, hdim, activ)
        if discrete:
            self.sample_activ = partial(F.gumbel_softmax, tau=gumbel_tau)
        self.apply(_init_weights)

    def forward(self, zs: torch.Tensor):
        pre_activ = self.net(zs)
        if self.discrete:
            action = self.sample_activ(pre_activ)
        else:
            action = torch.tanh(pre_activ)
        return action, pre_activ

    def act(self, zs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if not deterministic:
            action, _ = self.forward(zs)
            return action
        pre_activ = self.net(zs)
        if self.discrete:
            return F.one_hot(
                pre_activ.argmax(dim=-1), pre_activ.shape[-1]
            ).float()
        return torch.tanh(pre_activ)


class _QHead(nn.Module):
    def __init__(self, input_dim: int, hdim: int, activ: str):
        super().__init__()
        self.trunk = MLP3(input_dim, hdim, hdim, activ)
        self.out = nn.Linear(hdim, 1)
        self.activ = getattr(F, activ)
        self.apply(_init_weights)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        return self.out(_ln_activ(self.trunk(zsa), self.activ))


class TwinQCritic(nn.Module):
    def __init__(self, zsa_dim: int = 512, hdim: int = 512, activ: str = "elu"):
        super().__init__()
        self.q1 = _QHead(zsa_dim, hdim, activ)
        self.q2 = _QHead(zsa_dim, hdim, activ)

    def forward(self, zsa: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.q1(zsa), self.q2(zsa)], dim=1)


class TwoHotReward:
    """Two-hot encoding over symexp-spaced bins (clean-room)."""

    def __init__(self, lower: float = -10, upper: float = 10, num_bins: int = 65,
                 device: torch.device | None = None):
        device = device or torch.device("cpu")
        linear = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = linear.sign() * (linear.abs().exp() - 1)  # symexp spacing
        self.num_bins = num_bins

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        # nearest lower bin: the argmin over (x - bins) restricted to bins <= x
        diff = x - self.bins.reshape(1, -1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)  # +1e8 penalty where bins > x
        ind = torch.argmin(diff, dim=1, keepdim=True)
        lower = self.bins[ind]
        upper = self.bins[(ind + 1).clamp(0, self.num_bins - 1)]
        weight = (x - lower) / (upper - lower)
        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind + 1).clamp(0, self.num_bins), weight)
        return two_hot

    def inverse(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_pred = F.log_softmax(pred, dim=-1)
        target_two_hot = self.transform(target)
        return -(target_two_hot * log_pred).sum(-1, keepdim=True)


_SHIFT_GRID_CACHE: dict = {}


def shift_augment(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random translate each image by up to ``pad`` pixels (both axes).

    Uses replicate padding then grid sampling with a per-image offset —
    equivalent behavior to the standard DrQ-style random-shift augmentation.
    The base grid is cached per (height, width, pad, device).
    """
    batch, channels, height, width = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    key = (height, width, pad, images.device)
    base_grid = _SHIFT_GRID_CACHE.get(key)
    if base_grid is None or base_grid.device != images.device:
        eps = 1.0 / (height + 2 * pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, height + 2 * pad,
            device=images.device, dtype=torch.float,
        )[:height]
        arange = arange.unsqueeze(0).repeat(height, 1).unsqueeze(2)
        grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = grid.unsqueeze(0)
        _SHIFT_GRID_CACHE[key] = base_grid
    base_grid = base_grid.expand(batch, -1, -1, -1)
    shift = torch.randint(
        0, 2 * pad + 1, size=(batch, 1, 1, 2),
        device=images.device, dtype=torch.float,
    )
    shift = shift * 2.0 / (height + 2 * pad)
    return F.grid_sample(
        padded.float(), base_grid + shift,
        padding_mode="zeros", align_corners=False,
    ).to(images.dtype)
