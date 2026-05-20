from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class RingMaskedConv2d(nn.Conv2d):
    """环形盲点卷积。

    这个卷积层只允许使用距离卷积中心大于 blind_radius 的像素。
    用它做 Q 分支的第一层，可以保证中心像素及其相关邻域不会直接进入预测。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        blind_radius: float,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        if kernel_size % 2 == 0:
            raise ValueError("RingMaskedConv2d 的 kernel_size 必须是奇数。")
        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.blind_radius = float(blind_radius)

        radius = kernel_size // 2
        yy, xx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        dist = torch.sqrt(xx.square() + yy.square())

        # dist <= blind_radius 的权重全部置零，形成严格的中心盲区。
        mask_2d = (dist > self.blind_radius).to(torch.float32)
        self.register_buffer("weight_mask", mask_2d.view(1, 1, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.weight * self.weight_mask
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(
            x,
            masked_weight,
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, blind_radius={self.blind_radius:g}"
