"""Attentive 64x64 RGB encoder for the native EADream world model."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import ChannelAttention, SpatialAttention
from .initialization import source_weight_init


def _same_padding(size: int, kernel: int, stride: int, dilation: int = 1) -> int:
    return max(
        (math.ceil(size / stride) - 1) * stride
        + (kernel - 1) * dilation
        + 1
        - size,
        0,
    )


class _SamePadConv2d(nn.Conv2d):
    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        pad_h = _same_padding(height, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = _same_padding(width, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h or pad_w:
            x = F.pad(
                x,
                (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ),
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _ImageChannelLayerNorm(nn.Module):
    """Apply LayerNorm to channels after an NCHW-to-NHWC permutation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ImageEncoder(nn.Module):
    """Encode normalized HWC RGB images into flat EADream embeddings."""

    def __init__(
        self,
        depth: int = 32,
        *,
        attention: bool = True,
        attention_kernel: int = 3,
        attention_ratio: int = 2,
        kernel_size: int = 4,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.depth = int(depth)
        self.attention = bool(attention)

        layers: list[nn.Module] = []
        if self.attention:
            layers.extend(
                (ChannelAttention(3, ratio=1), SpatialAttention(attention_kernel))
            )

        in_channels = 3
        for out_channels in (depth, 2 * depth, 4 * depth, 8 * depth):
            layers.extend(
                (
                    _SamePadConv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        bias=False,
                    ),
                    _ImageChannelLayerNorm(out_channels),
                    nn.SiLU(),
                )
            )
            in_channels = out_channels

        if self.attention:
            layers.extend(
                (
                    ChannelAttention(8 * depth, ratio=attention_ratio),
                    SpatialAttention(attention_kernel),
                )
            )

        self.layers = nn.ModuleList(layers)
        self.outdim = 4 * 4 * 8 * depth
        self.output_dim = self.outdim
        self.hw = 4 * 4
        self.ch = 8 * depth
        self.apply(source_weight_init)

    @property
    def input_channel_attention(self) -> ChannelAttention | None:
        return cast(ChannelAttention, self.layers[0]) if self.attention else None

    @property
    def input_spatial_attention(self) -> SpatialAttention | None:
        return cast(SpatialAttention, self.layers[1]) if self.attention else None

    @property
    def final_channel_attention(self) -> ChannelAttention | None:
        return cast(ChannelAttention, self.layers[-2]) if self.attention else None

    @property
    def final_spatial_attention(self) -> SpatialAttention | None:
        return cast(SpatialAttention, self.layers[-1]) if self.attention else None

    @staticmethod
    def _center_input(image: Tensor) -> Tensor:
        """Make the private encoder copy that receives the source -0.5 offset."""
        return image.clone().sub(0.5)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim < 3 or tuple(image.shape[-3:]) != (64, 64, 3):
            raise ValueError("ImageEncoder expects normalized [..., 64, 64, 3] RGB")
        leading_shape = image.shape[:-3]
        x = self._center_input(image).reshape(-1, 64, 64, 3).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(*leading_shape, self.outdim)


__all__ = ["ImageEncoder"]
