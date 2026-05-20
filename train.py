from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from datasets import load_single_image_data
from losses import BlindSpotReconstructionLoss, attention_entropy_regularizer
from models import AttentionBSN
from utils.config import deep_update, load_config, save_config
from utils.image_io import denormalize_image, save_array, save_preview
from utils.seed import set_seed
from utils.tiling import random_crop_tensor, tiled_predict

try:
    from tqdm import trange
except Exception:  # pragma: no cover - 没装 tqdm 时退化为普通 range
    trange = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单张 BFI 图 attention-BSN 自监督降噪训练")
    parser.add_argument("--image", required=True, help="输入 BFI 图路径，支持 npy/npz/png/tif 等")
    parser.add_argument("--config", default="configs/attention_bsn_default.json", help="JSON 配置文件")
    parser.add_argument("--output-dir", default=None, help="实验输出目录")
    parser.add_argument("--steps", type=int, default=None, help="覆盖配置中的训练步数")
    parser.add_argument("--lr", type=float, default=None, help="覆盖配置中的学习率")
    parser.add_argument("--device", default=None, help="cuda / cpu / auto")
    parser.add_argument("--save-interval", type=int, default=None, help="覆盖 checkpoint 保存间隔")
    parser.add_argument("--patch-size", type=int, default=None, help="训练随机裁剪 patch 尺寸，0 表示全图")
    parser.add_argument("--tile-size", type=int, default=None, help="最终整图输出的滑窗 tile 尺寸，0 表示全图")
    parser.add_argument("--tile-context", type=int, default=None, help="滑窗推理额外上下文半径")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if args.device is not None:
        updates["device"] = args.device
    train_updates: Dict[str, Any] = {}
    if args.output_dir is not None:
        train_updates["output_dir"] = args.output_dir
    if args.steps is not None:
        train_updates["steps"] = args.steps
    if args.lr is not None:
        train_updates["lr"] = args.lr
    if args.save_interval is not None:
        train_updates["save_interval"] = args.save_interval
    if args.patch_size is not None:
        train_updates["patch_size"] = args.patch_size
    if train_updates:
        updates["train"] = train_updates
    infer_updates: Dict[str, Any] = {}
    if args.tile_size is not None:
        infer_updates["tile_size"] = args.tile_size
    if args.tile_context is not None:
        infer_updates["tile_context"] = args.tile_context
    if infer_updates:
        updates["infer"] = infer_updates
    return updates


def save_checkpoint(
    path: Path,
    model: AttentionBSN,
    config: Dict[str, Any],
    norm_meta: Dict[str, Any],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "normalization": norm_meta,
            "step": step,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = deep_update(load_config(args.config), build_overrides(args))
    set_seed(int(config.get("seed", 42)))

    device = select_device(config.get("device", "auto"))
    output_dir = Path(config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.json")

    data = load_single_image_data(args.image, config["data"])
    image = torch.from_numpy(data.image)[None, None].to(device=device, dtype=torch.float32)

    model = AttentionBSN(**config["model"]).to(device)
    criterion = BlindSpotReconstructionLoss(
        mask_mode=config["loss"].get("mask_mode", "grid"),
        grid_period=int(config["loss"].get("grid_period", 5)),
        random_ratio=float(config["loss"].get("random_ratio", 0.03)),
    )
    entropy_weight = float(config["loss"].get("entropy_weight", 0.0))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )

    steps = int(config["train"]["steps"])
    patch_size = int(config["train"].get("patch_size", 0))
    log_interval = int(config["train"].get("log_interval", 50))
    save_interval = int(config["train"].get("save_interval", 500))
    iterator = trange(1, steps + 1, desc="train") if trange is not None else range(1, steps + 1)

    history = []
    for step in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_image = random_crop_tensor(image, patch_size)
        pred, aux = model(train_image)
        recon_loss, _ = criterion(pred, train_image)
        entropy_loss = attention_entropy_regularizer(aux["attention_entropy"])
        loss = recon_loss + entropy_weight * entropy_loss

        loss.backward()
        optimizer.step()

        if step % log_interval == 0 or step == 1:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "recon_loss": float(recon_loss.detach().cpu()),
                "attention_entropy": float(aux["attention_entropy"].detach().cpu()),
                "valid_query_ratio": float(aux["valid_query_ratio"].detach().cpu()),
            }
            history.append(record)
            message = (
                f"step={step:05d} loss={record['loss']:.6f} "
                f"mse={record['recon_loss']:.6f} "
                f"entropy={record['attention_entropy']:.4f} "
                f"valid={record['valid_query_ratio']:.3f}"
            )
            if trange is not None and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(message)
            else:
                print(message)

        if save_interval > 0 and step % save_interval == 0:
            save_checkpoint(output_dir / f"checkpoint_{step:06d}.pt", model, config, data.norm_meta, step)

    save_checkpoint(output_dir / "checkpoint_final.pt", model, config, data.norm_meta, steps)

    model.eval()
    with torch.no_grad():
        tile_size = int(config.get("infer", {}).get("tile_size", 0))
        tile_overlap = int(config.get("infer", {}).get("tile_overlap", 32))
        tile_context = int(config.get("infer", {}).get("tile_context", tile_overlap))
        denoised_norm = tiled_predict(
            image,
            lambda tile: model(tile)[0],
            tile_size=tile_size,
            overlap=tile_overlap,
            context=tile_context,
        )
    denoised_norm_np = denoised_norm.squeeze().detach().cpu().numpy().astype(np.float32)
    denoised_raw = denormalize_image(denoised_norm_np, data.norm_meta)

    save_array(output_dir / "denoised.npy", denoised_raw)
    save_preview(output_dir / "denoised_preview.tif", denoised_norm_np)

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"训练完成。结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
