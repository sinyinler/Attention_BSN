from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SparseNonLocalAttention(nn.Module):
    """面向单图 BSN 的稀疏非局部 attention。

    朴素全局 self-attention 是 O((HW)^2)，单张 BFI 图稍大就会爆显存。
    这里采用“下采样候选池 + top-k + 分块 query”的方式，保留非局部搜索能力，
    同时把显存压力控制在可用范围内。
    """

    def __init__(
        self,
        dim: int,
        value_dim: int | None = None,
        topk: int = 32,
        candidate_stride: int = 4,
        exclude_radius: float = 10.0,
        chunk_size: int = 4096,
        normalize_qk: bool = True,
    ) -> None:
        super().__init__()
        if candidate_stride <= 0:
            raise ValueError("candidate_stride 必须大于 0。")
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0。")
        value_dim = dim if value_dim is None else value_dim
        self.topk = int(topk)
        self.candidate_stride = int(candidate_stride)
        self.exclude_radius = float(exclude_radius)
        self.chunk_size = int(chunk_size)
        self.normalize_qk = bool(normalize_qk)

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(value_dim, dim, kernel_size=1)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = 1.0 / math.sqrt(dim)

    @staticmethod
    def _grid_coords(h: int, w: int, device: torch.device, stride: int = 1) -> torch.Tensor:
        ys = torch.arange(0, h, stride, device=device)
        xs = torch.arange(0, w, stride, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).to(torch.float32)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对已经含有 -inf 的分数做稳定 softmax。

        返回 attention 权重和每个 query 的熵。没有任何合法候选时，权重全为 0。
        """

        valid = torch.isfinite(scores)
        # AMP/float16 下 -1e9 可能溢出，使用当前 dtype 可表示的最小值更稳。
        safe_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        max_scores = safe_scores.max(dim=-1, keepdim=True).values
        weights = torch.exp(safe_scores - max_scores) * valid.to(scores.dtype)
        denom = weights.sum(dim=-1, keepdim=True)
        weights = weights / denom.clamp_min(1.0e-12)

        log_weights = torch.log(weights.clamp_min(1.0e-12))
        entropy = -(weights * log_weights).sum(dim=-1)
        entropy = entropy.masked_fill(denom.squeeze(-1) <= 0, 0.0)
        return weights, entropy

    def forward(
        self,
        query_feat: torch.Tensor,
        key_feat: torch.Tensor,
        value_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = query_feat.shape
        device = query_feat.device

        q = self.q_proj(query_feat)
        k = self.k_proj(key_feat)
        v = self.v_proj(value_feat)

        k_pool = k[:, :, :: self.candidate_stride, :: self.candidate_stride]
        v_pool = v[:, :, :: self.candidate_stride, :: self.candidate_stride]

        q_flat = q.flatten(2).transpose(1, 2)  # [B, HW, C]
        k_flat = k_pool.flatten(2).transpose(1, 2)  # [B, M, C]
        v_flat = v_pool.flatten(2).transpose(1, 2)  # [B, M, C]

        if self.normalize_qk:
            q_flat = F.normalize(q_flat, dim=-1)
            k_flat = F.normalize(k_flat, dim=-1)

        query_coords = self._grid_coords(h, w, device, stride=1)
        candidate_coords = self._grid_coords(h, w, device, stride=self.candidate_stride)
        num_query = q_flat.shape[1]
        num_candidate = k_flat.shape[1]
        topk = min(self.topk, num_candidate) if self.topk > 0 else num_candidate

        outputs = []
        entropy_sum = q_flat.new_tensor(0.0)
        valid_count = q_flat.new_tensor(0.0)

        for start in range(0, num_query, self.chunk_size):
            end = min(start + self.chunk_size, num_query)
            q_chunk = q_flat[:, start:end, :]  # [B, L, C]
            coords = query_coords[start:end]

            scores = torch.bmm(q_chunk, k_flat.transpose(1, 2)) * self.scale

            # J-invariance 关键：中心附近的候选点永远不能被 attention 选到。
            dist2 = (
                (coords[:, None, 0] - candidate_coords[None, :, 0]).square()
                + (coords[:, None, 1] - candidate_coords[None, :, 1]).square()
            )
            invalid = dist2 <= (self.exclude_radius * self.exclude_radius)
            scores = scores.masked_fill(invalid.unsqueeze(0), -torch.inf)

            if topk < num_candidate:
                top_scores, top_idx = torch.topk(scores, k=topk, dim=-1)
                weights, entropy = self._masked_softmax(top_scores)

                batch_idx = torch.arange(b, device=device)[:, None, None]
                top_values = v_flat[batch_idx, top_idx]
                out_chunk = (weights.unsqueeze(-1) * top_values).sum(dim=2)
            else:
                weights, entropy = self._masked_softmax(scores)
                out_chunk = torch.bmm(weights, v_flat)

            has_valid = weights.sum(dim=-1) > 0
            entropy_sum = entropy_sum + entropy[has_valid].sum()
            valid_count = valid_count + has_valid.to(q_flat.dtype).sum()
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=1).transpose(1, 2).reshape(b, c, h, w)
        out = self.out_proj(out)

        aux = {
            "attention_entropy": entropy_sum / valid_count.clamp_min(1.0),
            "valid_query_ratio": valid_count / float(b * num_query),
        }
        return out, aux
