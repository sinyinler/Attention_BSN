from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image


def load_image_array(path: str | Path) -> np.ndarray:
    """读取单张 BFI 图，支持 npy/npz 和常见图像格式。"""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        first_key = list(data.keys())[0]
        arr = data[first_key]
    else:
        arr = np.asarray(Image.open(path))

    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 3:
        # RGB/RGBA 图像转灰度；BFI 本身通常是单通道，这里只是增强兼容性。
        arr = arr[..., :3].astype(np.float32)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    if arr.ndim != 2:
        raise ValueError(f"期望读取到二维单通道图像，但得到 shape={arr.shape}")
    return arr.astype(np.float32)


def normalize_image(
    arr: np.ndarray,
    mode: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """把图像归一化到适合训练的数值范围。"""

    arr = arr.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("输入图像没有有效数值。")

    if mode == "none":
        offset = 0.0
        scale = 1.0
    elif mode == "minmax":
        offset = float(np.nanmin(arr))
        high = float(np.nanmax(arr))
        scale = high - offset
    elif mode == "percentile":
        offset = float(np.nanpercentile(arr, percentile_low))
        high = float(np.nanpercentile(arr, percentile_high))
        scale = high - offset
    else:
        raise ValueError(f"未知归一化方式: {mode}")

    if abs(scale) < 1.0e-12:
        scale = 1.0
    norm = (arr - offset) / scale
    if mode != "none":
        norm = np.clip(norm, 0.0, 1.0)

    meta = {
        "mode": mode,
        "offset": float(offset),
        "scale": float(scale),
        "percentile_low": float(percentile_low),
        "percentile_high": float(percentile_high),
        "shape": list(arr.shape),
    }
    return norm.astype(np.float32), meta


def load_normalized_image(
    path: str | Path,
    mode: str = "percentile",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr = load_image_array(path)
    return normalize_image(arr, mode, percentile_low, percentile_high)


def denormalize_image(norm: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """把网络输出还原到原始 BFI 数值尺度。"""

    return norm.astype(np.float32) * float(meta.get("scale", 1.0)) + float(meta.get("offset", 0.0))


def save_array(path: str | Path, arr: np.ndarray) -> None:
    """保存浮点结果；npy 会保留真实数值，其它格式用于快速查看。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, arr.astype(np.float32))
    else:
        save_preview(path, arr)


def save_preview(path: str | Path, arr: np.ndarray) -> None:
    """保存预览图。

    PNG/JPG/BMP 默认保存为 8-bit，兼容普通图片查看器；
    TIF/TIFF 保存为 16-bit，适合需要更细灰度层级的场景。
    若输入不在 [0,1]，会按当前图像 min-max 拉伸。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("无法保存预览图：数组没有有效数值。")

    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if lo < 0.0 or hi > 1.0:
        scale = max(hi - lo, 1.0e-12)
        arr = (arr - lo) / scale
    arr = np.clip(arr, 0.0, 1.0)

    if path.suffix.lower() in {".tif", ".tiff"}:
        out = (arr * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (arr * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(path)
