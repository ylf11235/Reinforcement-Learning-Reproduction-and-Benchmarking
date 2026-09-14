"""RSSM-OP image and event decoders for EADream."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .initialization import source_uniform_init, source_weight_init


class _ImageChannelLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _transpose_same_padding(kernel_size: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    value = dilation * (kernel_size - 1) - stride + 1
    padding = math.ceil(value / 2)
    output_padding = padding * 2 - value
    return padding, output_padding


class _SpatialDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        output_channels: int,
        *,
        depth: int,
        kernel_size: int,
        outscale: float,
        sigmoid: bool,
        output_offset: float,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or depth <= 0 or output_channels <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.depth = int(depth)
        self.output_channels = int(output_channels)
        self.sigmoid_output = bool(sigmoid)
        self.output_offset = float(output_offset)
        self.expansion_channels = 8 * depth
        self.expansion = nn.Linear(feature_dim, 4 * 4 * self.expansion_channels)
        self.expansion.apply(source_uniform_init(outscale))

        padding, output_padding = _transpose_same_padding(kernel_size, 2)
        channels = (4 * depth, 2 * depth, depth, output_channels)
        stages: list[nn.Module] = []
        initialized_prefix = 0
        in_channels = self.expansion_channels
        for index, out_channels in enumerate(channels):
            final = index == len(channels) - 1
            convolution = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=final,
            )
            stages.append(convolution)
            if not final:
                normalization = _ImageChannelLayerNorm(out_channels)
                stages.extend((normalization, nn.SiLU()))
            if index == 1:
                for module in stages[:-1]:
                    module.apply(source_weight_init)
                initialized_prefix = len(stages)
            in_channels = out_channels
        for module in stages[initialized_prefix:-1]:
            module.apply(source_weight_init)
        stages[-1].apply(source_uniform_init(outscale))
        self.layers = nn.Sequential(*stages)

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"decoder expected {self.feature_dim} features, got {features.shape[-1]}"
            )
        leading_shape = features.shape[:-1]
        x = self.expansion(features)
        x = x.reshape(-1, 4, 4, self.expansion_channels).permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape(*leading_shape, self.output_channels, 64, 64)
        mean = x.movedim(-3, -1)
        if self.sigmoid_output:
            return torch.sigmoid(mean)
        return mean + self.output_offset


class ImageDecoder(_SpatialDecoder):
    """Decode 2560-element RSSM-OP features to HWC RGB means."""

    def __init__(
        self,
        feature_dim: int = 2560,
        *,
        depth: int = 32,
        kernel_size: int = 4,
        outscale: float = 1.0,
        output_offset: float = 0.5,
    ) -> None:
        super().__init__(
            feature_dim,
            3,
            depth=depth,
            kernel_size=kernel_size,
            outscale=outscale,
            sigmoid=False,
            output_offset=output_offset,
        )


class EventDecoder(_SpatialDecoder):
    """Decode prior/posterior/deterministic features to sigmoid event masks."""

    def __init__(
        self,
        feature_dim: int = 4608,
        *,
        depth: int = 32,
        kernel_size: int = 4,
        outscale: float = 1.0,
    ) -> None:
        super().__init__(
            feature_dim,
            1,
            depth=depth,
            kernel_size=kernel_size,
            outscale=outscale,
            sigmoid=True,
            output_offset=0.0,
        )


__all__ = ["EventDecoder", "ImageDecoder"]
