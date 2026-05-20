from __future__ import annotations

import math

import numpy as np

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - 没装 skimage 时只影响 SSIM
    structural_similarity = None


def compute_psnr(pred: np.ndarray, target: np.ndarray, data_range: float | None = None) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1.0e-20:
        return float("inf")
    if data_range is None:
        data_range = float(np.nanmax(target) - np.nanmin(target))
    data_range = max(float(data_range), 1.0e-12)
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float | None = None) -> float:
    if structural_similarity is None:
        raise RuntimeError("计算 SSIM 需要安装 scikit-image。")
    if data_range is None:
        data_range = float(np.nanmax(target) - np.nanmin(target))
    data_range = max(float(data_range), 1.0e-12)
    return float(structural_similarity(target, pred, data_range=data_range))
