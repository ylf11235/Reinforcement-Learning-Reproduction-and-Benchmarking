"""Channel and spatial attention used by the EADream image encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """CBAM channel attention with shared average/max-pool projections."""

    def __init__(self, channels: int, ratio: int = 2) -> None:
        super().__init__()
        if channels <= 0 or ratio <= 0 or channels // ratio <= 0:
            raise ValueError("channels and ratio must define a positive hidden width")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        average = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        maximum = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(average + maximum)


class SpatialAttention(nn.Module):
    """CBAM spatial attention over concatenated channel mean and maximum."""

    def __init__(
        self, kernel_size: int = 3, *, kernel: int | None = None
    ) -> None:
        super().__init__()
        if kernel is not None:
            if kernel_size != 3 and kernel_size != kernel:
                raise ValueError("kernel and kernel_size disagree")
            kernel_size = kernel
        if kernel_size not in (3, 7):
            raise ValueError("kernel size must be 3 or 7")
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    @property
    def conv1(self) -> nn.Conv2d:
        """Source-name compatibility without registering the module twice."""
        return self.conv

    def forward(self, x: Tensor) -> Tensor:
        average = torch.mean(x, dim=1, keepdim=True)
        maximum = torch.amax(x, dim=1, keepdim=True)
        attention = self.conv(torch.cat((average, maximum), dim=1))
        return x * self.sigmoid(attention)


__all__ = ["ChannelAttention", "SpatialAttention"]
