from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch import nn

from .charbonnier import CharbonnierLoss


class BlindSpotReconstructionLoss(nn.Module):
    """BSN 自监督重建损失。

    模型结构已经保证预测位置不看中心邻域，因此训练时只需要在一部分像素上
    对预测值和观测 BFI 值做重建损失。grid 模式会随机平移一个周期采样 pattern，
    比每次纯随机采样更稳定。
    """

    def __init__(
        self,
        mask_mode: Literal["grid", "random"] = "grid",
        grid_period: int = 5,
        random_ratio: float = 0.03,
        loss_type: Literal["mse", "l1", "charbonnier"] = "charbonnier",
        charbonnier_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.mask_mode = mask_mode
        self.grid_period = int(grid_period)
        self.random_ratio = float(random_ratio)
        self.loss_type = loss_type
        self.charbonnier = CharbonnierLoss(eps=charbonnier_eps, reduction="none")

    def make_mask(self, target: torch.Tensor) -> torch.Tensor:
        _, _, h, w = target.shape
        device = target.device

        if self.mask_mode == "grid":
            oy = torch.randint(0, self.grid_period, (1,), device=device)
            ox = torch.randint(0, self.grid_period, (1,), device=device)
            yy = torch.arange(h, device=device)[:, None]
            xx = torch.arange(w, device=device)[None, :]
            mask = ((yy - oy) % self.grid_period == 0) & ((xx - ox) % self.grid_period == 0)
        elif self.mask_mode == "random":
            mask = torch.rand((h, w), device=device) < self.random_ratio
        else:
            raise ValueError(f"未知 mask_mode: {self.mask_mode}")

        return mask[None, None, :, :].to(target.dtype)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = self.make_mask(target)

        if self.loss_type == "mse":
            diff = (prediction - target).square()
        elif self.loss_type == "l1":
            diff = (prediction - target).abs()
        elif self.loss_type == "charbonnier":
            diff = self.charbonnier(prediction, target)
        else:
            raise ValueError(f"未知 loss_type: {self.loss_type}")

        mask = mask.to(dtype=diff.dtype, device=diff.device)
        denom = mask.expand_as(diff).sum().clamp_min(1.0)
        loss = (diff * mask).sum() / denom
        return loss, mask


def attention_entropy_regularizer(entropy: torch.Tensor) -> torch.Tensor:
    """熵正则项。

    训练时最小化 -entropy，相当于鼓励 attention 分布更分散，
    降低单个噪声相似 patch 被过度信任的风险。
    """

    return -entropy
