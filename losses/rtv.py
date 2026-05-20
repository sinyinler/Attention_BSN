from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_kernel_2d(ksize: int, sigma: float, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """生成二维高斯核。"""

    radius = (ksize - 1) / 2.0
    axis = torch.arange(ksize, device=device, dtype=dtype) - radius
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1.0e-12)


class RTVRegularizer(nn.Module):
    """Relative Total Variation 正则。

    RTV 倾向于压制小尺度纹理/噪声，同时相对保留较强结构边缘。
    在这里它只作为预测结果的弱正则项，不参与 BSN 盲点 mask。
    """

    def __init__(self, radius: int = 2, sigma: float = 2.0, eps: float = 1.0e-3, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction 必须是 mean/sum/none 之一。")
        self.radius = int(radius)
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.reduction = reduction

        ksize = 2 * self.radius + 1
        kernel = _gaussian_kernel_2d(ksize, self.sigma, device="cpu", dtype=torch.float32)
        self.register_buffer("_kernel_cpu", kernel)

    @staticmethod
    def _dx(x: torch.Tensor) -> torch.Tensor:
        diff = x[..., :, 1:] - x[..., :, :-1]
        return F.pad(diff, (0, 1, 0, 0), mode="replicate")

    @staticmethod
    def _dy(x: torch.Tensor) -> torch.Tensor:
        diff = x[..., 1:, :] - x[..., :-1, :]
        return F.pad(diff, (0, 0, 0, 1), mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(f"RTVRegularizer 期望输入 [B,C,H,W]，但得到 shape={tuple(x.shape)}")

        _, channels, _, _ = x.shape
        dx = self._dx(x)
        dy = self._dy(x)

        kernel = self._kernel_cpu.to(device=x.device, dtype=x.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        weight = kernel.expand(channels, 1, kernel.shape[-2], kernel.shape[-1]).contiguous()
        pad = self.radius

        # WTV 是窗口内梯度幅值加权和，WIV 是带符号梯度加权和再取绝对值。
        wtv_x = F.conv2d(dx.abs(), weight, padding=pad, groups=channels)
        wtv_y = F.conv2d(dy.abs(), weight, padding=pad, groups=channels)
        wiv_x = F.conv2d(dx, weight, padding=pad, groups=channels).abs()
        wiv_y = F.conv2d(dy, weight, padding=pad, groups=channels).abs()

        rtv_map = wtv_x / (wiv_x + self.eps) + wtv_y / (wiv_y + self.eps)
        if self.reduction == "mean":
            return rtv_map.mean()
        if self.reduction == "sum":
            return rtv_map.sum()
        return rtv_map
