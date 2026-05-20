from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from datasets import load_single_image_data
from models import AttentionBSN
from utils.image_io import denormalize_image, load_image_array, save_array, save_preview
from utils.metrics import compute_psnr, compute_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用训练好的 attention-BSN checkpoint 做单图降噪")
    parser.add_argument("--image", required=True, help="待降噪 BFI 图路径")
    parser.add_argument("--checkpoint", required=True, help="train.py 保存的 checkpoint")
    parser.add_argument("--output", default="runs/attention_bsn/test_denoised.npy", help="输出 npy 路径")
    parser.add_argument("--preview", default=None, help="可选预览图路径，例如 denoised_preview.tif")
    parser.add_argument("--gt", default=None, help="可选 clean/long-window BFI 参考图，用于计算指标")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]

    data = load_single_image_data(args.image, config["data"])
    image = torch.from_numpy(data.image)[None, None].to(device=device, dtype=torch.float32)

    model = AttentionBSN(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        pred_norm, _ = model(image)
    pred_norm_np = pred_norm.squeeze().detach().cpu().numpy().astype(np.float32)
    pred_raw = denormalize_image(pred_norm_np, data.norm_meta)

    output = Path(args.output)
    save_array(output, pred_raw)
    preview = Path(args.preview) if args.preview is not None else output.with_name(output.stem + "_preview.tif")
    save_preview(preview, pred_norm_np)

    if args.gt is not None:
        gt = load_image_array(args.gt)
        if gt.shape != pred_raw.shape:
            raise ValueError(f"GT shape {gt.shape} 与预测 shape {pred_raw.shape} 不一致。")
        data_range = float(np.nanmax(gt) - np.nanmin(gt))
        print(f"PSNR: {compute_psnr(pred_raw, gt, data_range):.4f} dB")
        print(f"SSIM: {compute_ssim(pred_raw, gt, data_range):.4f}")

    print(f"降噪结果: {output}")
    print(f"预览图: {preview}")


if __name__ == "__main__":
    main()
