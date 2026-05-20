from __future__ import annotations

from typing import Callable, List, Tuple

import torch


def random_crop_tensor(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    """从单张图中随机裁一个训练 patch。

    attention-BSN 的显存主要随像素数增长。单图自监督训练时用随机 patch
    可以覆盖整张图的不同区域，同时避免全图 attention 反传导致 OOM。
    """

    if patch_size <= 0:
        return image
    _, _, h, w = image.shape
    if h <= patch_size and w <= patch_size:
        return image

    crop_h = min(patch_size, h)
    crop_w = min(patch_size, w)
    y0 = torch.randint(0, h - crop_h + 1, (1,), device=image.device).item()
    x0 = torch.randint(0, w - crop_w + 1, (1,), device=image.device).item()
    return image[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w]


def _tile_slices(length: int, tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    if length <= tile_size:
        return [(0, length)]

    step = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + tile_size) for start in starts]


def _cosine_ramp(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    t = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(torch.pi * t)


def _blend_weight_1d(
    length: int,
    overlap: int,
    has_left_neighbor: bool,
    has_right_neighbor: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """生成一维平滑融合权重。

    只在内部 tile 边界做渐入/渐出；图像真实边界不降权，避免边界变暗。
    """

    weight = torch.ones(length, device=device, dtype=dtype)
    ramp = min(max(overlap, 0), max(length // 2, 1))

    if has_left_neighbor and ramp > 1:
        weight[:ramp] = _cosine_ramp(ramp, device, dtype).clamp_min(1.0e-3)
    if has_right_neighbor and ramp > 1:
        weight[-ramp:] = torch.flip(_cosine_ramp(ramp, device, dtype), dims=(0,)).clamp_min(1.0e-3)
    return weight


def tiled_predict(
    image: torch.Tensor,
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = 256,
    overlap: int = 32,
    context: int | None = None,
) -> torch.Tensor:
    """带 halo 上下文的滑窗推理。

    每个 tile 前向时会额外读取一圈 context，但只把中心区域写回输出。
    这样 tile 边缘缺少上下文导致的不可靠预测不会直接形成拼接线。
    """

    if tile_size <= 0:
        return predict_fn(image)

    b, c, h, w = image.shape
    if b != 1:
        raise ValueError("tiled_predict 当前只支持 batch=1 的单图推理。")

    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    context = overlap if context is None else int(context)
    context = max(0, context)
    y_slices = _tile_slices(h, tile_size, overlap)
    x_slices = _tile_slices(w, tile_size, overlap)

    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)

    for y0, y1 in y_slices:
        for x0, x1 in x_slices:
            # 输入区域比写回区域更大，给 attention 和卷积提供 tile 外上下文。
            in_y0 = max(0, y0 - context)
            in_y1 = min(h, y1 + context)
            in_x0 = max(0, x0 - context)
            in_x1 = min(w, x1 + context)

            tile = image[:, :, in_y0:in_y1, in_x0:in_x1]
            pred = predict_fn(tile)

            crop_y0 = y0 - in_y0
            crop_y1 = crop_y0 + (y1 - y0)
            crop_x0 = x0 - in_x0
            crop_x1 = crop_x0 + (x1 - x0)
            pred_core = pred[:, :, crop_y0:crop_y1, crop_x0:crop_x1]

            wy = _blend_weight_1d(
                y1 - y0,
                overlap,
                has_left_neighbor=y0 > 0,
                has_right_neighbor=y1 < h,
                device=image.device,
                dtype=image.dtype,
            )
            wx = _blend_weight_1d(
                x1 - x0,
                overlap,
                has_left_neighbor=x0 > 0,
                has_right_neighbor=x1 < w,
                device=image.device,
                dtype=image.dtype,
            )
            tile_weight = wy[None, None, :, None] * wx[None, None, None, :]

            output[:, :, y0:y1, x0:x1] += pred_core * tile_weight
            weight[:, :, y0:y1, x0:x1] += tile_weight

    return output / weight.clamp_min(1.0)
