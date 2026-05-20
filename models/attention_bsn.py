from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .masked_layers import RingMaskedConv2d
from .nonlocal_attention import SparseNonLocalAttention


class PointwiseBlock(nn.Module):
    """只做 1x1 混合的 MLP block。

    在融合阶段不用 3x3 卷积，是为了避免把相邻位置的信息重新混进中心预测里。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBSN(nn.Module):
    """单张 BFI 图的 attention-BSN 去噪网络。

    结构分成三部分：
    1. Q/local 分支：环形盲点卷积，只看中心半径外的信息；
    2. K/V 分支：普通局部特征，但 attention 会排除中心附近候选；
    3. 1x1 融合解码：输出每个像素的自监督预测。
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        blind_radius: int = 8,
        annulus_width: int = 6,
        kv_depth: int = 2,
        attention_topk: int = 32,
        candidate_stride: int = 4,
        attention_chunk_size: int = 4096,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.blind_radius = int(blind_radius)
        self.annulus_width = int(annulus_width)
        self.kv_depth = int(kv_depth)

        ring_radius = self.blind_radius + self.annulus_width
        ring_kernel = 2 * ring_radius + 1

        self.local_query = nn.Sequential(
            RingMaskedConv2d(
                in_channels=self.in_channels,
                out_channels=self.base_channels,
                kernel_size=ring_kernel,
                blind_radius=self.blind_radius,
            ),
            nn.GELU(),
            PointwiseBlock(self.base_channels, self.base_channels),
            PointwiseBlock(self.base_channels, self.base_channels),
        )

        kv_layers = []
        current_channels = self.in_channels
        for _ in range(self.kv_depth):
            kv_layers.extend(
                [
                    # 使用零填充可以避免边界 reflect 把中心盲区信息折回候选特征。
                    nn.Conv2d(current_channels, self.base_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                ]
            )
            current_channels = self.base_channels
        self.kv_encoder = nn.Sequential(*kv_layers)

        # K/V 分支有 kv_depth 层 3x3 卷积，感受野半径约为 kv_depth。
        # 因此 attention 的排除半径需要比 blind_radius 再多留出这部分余量。
        self.attention_exclude_radius = float(self.blind_radius + self.kv_depth)
        self.nonlocal_attention = SparseNonLocalAttention(
            dim=self.base_channels,
            value_dim=self.base_channels,
            topk=attention_topk,
            candidate_stride=candidate_stride,
            exclude_radius=self.attention_exclude_radius,
            chunk_size=attention_chunk_size,
        )

        decoder_hidden = max(1, self.base_channels // 2)
        self.decoder = nn.Sequential(
            PointwiseBlock(self.base_channels * 2, self.base_channels),
            PointwiseBlock(self.base_channels, decoder_hidden),
            nn.Conv2d(decoder_hidden, self.in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        local_feat = self.local_query(x)
        kv_feat = self.kv_encoder(x)
        nonlocal_feat, aux = self.nonlocal_attention(local_feat, kv_feat, kv_feat)

        pred = self.decoder(torch.cat([local_feat, nonlocal_feat], dim=1))
        aux["attention_exclude_radius"] = torch.as_tensor(
            self.attention_exclude_radius,
            dtype=x.dtype,
            device=x.device,
        )
        return pred, aux
