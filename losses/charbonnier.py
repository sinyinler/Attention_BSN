from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """Charbonnier 损失。

    它可以看作平滑版 L1，比 MSE 更不容易被 BFI 中的局部异常噪声带偏。
    reduction="none" 时返回逐像素 loss，方便配合 BSN 盲点 mask。
    """

    def __init__(self, eps: float = 1.0e-3, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction 必须是 mean/sum/none 之一。")
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
