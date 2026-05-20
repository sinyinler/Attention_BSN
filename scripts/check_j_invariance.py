from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from datasets import load_single_image_data
from models import AttentionBSN
from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 attention-BSN 对中心盲区扰动是否 J-invariant")
    parser.add_argument("--image", default=None, help="可选输入图；不提供时使用随机图")
    parser.add_argument("--checkpoint", default=None, help="可选 checkpoint；不提供时使用随机初始化模型")
    parser.add_argument("--config", default="configs/attention_bsn_default.json", help="没有 checkpoint 时使用的配置")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--num-points", type=int, default=8, help="随机检查的像素数量")
    parser.add_argument("--height", type=int, default=96, help="随机图高度")
    parser.add_argument("--width", type=int, default=96, help="随机图宽度")
    parser.add_argument("--delta", type=float, default=10.0, help="盲区内扰动幅度")
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


def make_disk_mask(h: int, w: int, y: int, x: int, radius: int, device: torch.device) -> torch.Tensor:
    yy = torch.arange(h, device=device)[:, None]
    xx = torch.arange(w, device=device)[None, :]
    return (yy - y).square() + (xx - x).square() <= radius * radius


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    if args.checkpoint is not None:
        checkpoint = load_checkpoint(args.checkpoint, device)
        config = checkpoint["config"]
    else:
        checkpoint = None
        config = load_config(args.config)

    if args.image is not None:
        data = load_single_image_data(args.image, config["data"])
        image = torch.from_numpy(data.image)[None, None].to(device=device, dtype=torch.float32)
    else:
        image = torch.rand(1, 1, args.height, args.width, device=device)

    model = AttentionBSN(**config["model"]).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()

    blind_radius = int(config["model"].get("blind_radius", 8))
    _, _, h, w = image.shape
    margin = max(blind_radius + int(config["model"].get("kv_depth", 2)) + 2, 4)
    if h <= 2 * margin or w <= 2 * margin:
        raise ValueError("图像太小，无法在边界外稳定检查 J-invariance。")

    generator = torch.Generator(device=device)
    generator.manual_seed(1234)
    ys = torch.randint(margin, h - margin, (args.num_points,), generator=generator, device=device)
    xs = torch.randint(margin, w - margin, (args.num_points,), generator=generator, device=device)

    diffs = []
    with torch.no_grad():
        base_pred, _ = model(image)
        for y_t, x_t in zip(ys, xs):
            y = int(y_t.item())
            x = int(x_t.item())
            perturbed = image.clone()
            disk = make_disk_mask(h, w, y, x, blind_radius, device)[None, None]
            noise = args.delta * torch.randn_like(perturbed)
            perturbed = torch.where(disk, perturbed + noise, perturbed)
            new_pred, _ = model(perturbed)
            diff = (new_pred[0, 0, y, x] - base_pred[0, 0, y, x]).abs().item()
            diffs.append(diff)

    mean_diff = sum(diffs) / max(len(diffs), 1)
    max_diff = max(diffs) if diffs else 0.0
    print(f"检查点数: {len(diffs)}")
    print(f"盲区半径: {blind_radius}")
    print(f"平均输出变化: {mean_diff:.8e}")
    print(f"最大输出变化: {max_diff:.8e}")
    if max_diff < 1.0e-5:
        print("结果: 通过，中心盲区扰动没有明显影响被检查像素的预测。")
    else:
        print("结果: 需要排查，预测可能仍依赖中心盲区。")


if __name__ == "__main__":
    main()
