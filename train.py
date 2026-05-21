from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from datasets import load_single_image_data
from losses import BlindSpotReconstructionLoss, RTVRegularizer, attention_entropy_regularizer
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


def make_autocast(device: torch.device, enabled: bool):
    """创建 AMP autocast 上下文，只在 CUDA 上启用。"""

    enabled = bool(enabled) and device.type == "cuda"
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except AttributeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(enabled: bool):
    """兼容不同 PyTorch 版本的 GradScaler。"""

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def compute_learning_rate(
    step: int,
    total_steps: int,
    initial_lr: float,
    peak_lr: float,
    final_lr: float,
    warmup_ratio: float,
) -> float:
    """前 20% warmup 到 peak_lr，之后 cosine annealing 到 final_lr。"""

    total_steps = max(1, int(total_steps))
    step = min(max(1, int(step)), total_steps)
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    if warmup_steps > 1 and step <= warmup_steps:
        progress = (step - 1) / float(warmup_steps - 1)
        return initial_lr + (peak_lr - initial_lr) * progress

    if step <= warmup_steps:
        return peak_lr

    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / float(decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (peak_lr - final_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """把当前 step 的学习率写入 optimizer。"""

    for group in optimizer.param_groups:
        group["lr"] = lr


def make_fixed_mask(
    shape: torch.Size,
    device: torch.device,
    dtype: torch.dtype,
    mode: str,
    grid_period: int,
    random_ratio: float,
    index: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """为监测集生成固定盲点 mask。"""

    _, _, height, width = shape
    if mode == "grid":
        period = max(1, int(grid_period))
        oy = index % period
        ox = (index * 3) % period
        yy = torch.arange(height, device=device)[:, None]
        xx = torch.arange(width, device=device)[None, :]
        mask = ((yy - oy) % period == 0) & ((xx - ox) % period == 0)
        return mask[None, None].to(dtype=dtype)

    if mode == "random":
        return (torch.rand(shape, device=device, generator=generator) < float(random_ratio)).to(dtype=dtype)

    raise ValueError(f"未知 monitor mask_mode: {mode}")


def build_monitor_batches(
    image: torch.Tensor,
    monitor_config: Dict[str, Any],
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """构建固定验证 patch，用于观察单图自监督是否开始拟合噪声。"""

    num_patches = max(1, int(monitor_config.get("num_patches", 4)))
    patch_size = int(monitor_config.get("patch_size", 512))
    mask_mode = monitor_config.get("mask_mode", "grid")
    grid_period = int(monitor_config.get("grid_period", 5))
    random_ratio = float(monitor_config.get("random_ratio", 0.03))

    _, _, height, width = image.shape
    crop_h = height if patch_size <= 0 else min(patch_size, height)
    crop_w = width if patch_size <= 0 else min(patch_size, width)
    generator = torch.Generator(device=image.device)
    generator.manual_seed(int(seed) + 2027)

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(num_patches):
        if height == crop_h:
            y0 = 0
        else:
            y0 = int(torch.randint(0, height - crop_h + 1, (1,), device=image.device, generator=generator).item())
        if width == crop_w:
            x0 = 0
        else:
            x0 = int(torch.randint(0, width - crop_w + 1, (1,), device=image.device, generator=generator).item())

        patch = image[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w]
        mask = make_fixed_mask(
            patch.shape,
            image.device,
            image.dtype,
            mode=mask_mode,
            grid_period=grid_period,
            random_ratio=random_ratio,
            index=index,
            generator=generator,
        )
        batches.append((patch, mask))

    return batches


def evaluate_monitor(
    model: AttentionBSN,
    monitor_batches: list[tuple[torch.Tensor, torch.Tensor]],
    criterion: BlindSpotReconstructionLoss,
    rtv_regularizer: RTVRegularizer,
    recon_weight: float,
    rtv_weight: float,
    entropy_weight: float,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    """在固定验证 patch 上评估，用于选择中期最优 checkpoint。"""

    was_training = model.training
    model.eval()
    totals = {
        "val_loss": 0.0,
        "val_recon_loss": 0.0,
        "val_rtv_loss": 0.0,
        "val_attention_entropy": 0.0,
    }

    with torch.no_grad():
        for patch, mask in monitor_batches:
            with make_autocast(device, amp_enabled):
                pred, aux = model(patch)

            pred_for_loss = pred.float()
            target_for_loss = patch.float()
            recon_loss, _ = criterion(pred_for_loss, target_for_loss, mask.float())
            if rtv_weight != 0.0:
                rtv_loss = rtv_regularizer(pred_for_loss)
            else:
                rtv_loss = pred_for_loss.new_tensor(0.0)
            entropy_loss = attention_entropy_regularizer(aux["attention_entropy"].float())
            loss = recon_weight * recon_loss + rtv_weight * rtv_loss + entropy_weight * entropy_loss

            totals["val_loss"] += float(loss.detach().cpu())
            totals["val_recon_loss"] += float(recon_loss.detach().cpu())
            totals["val_rtv_loss"] += float(rtv_loss.detach().cpu())
            totals["val_attention_entropy"] += float(aux["attention_entropy"].detach().cpu())

    denom = float(max(len(monitor_batches), 1))
    stats = {key: value / denom for key, value in totals.items()}
    if was_training:
        model.train()
    return stats


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """读取 checkpoint，兼容不同 PyTorch 版本。"""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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
        loss_type=config["loss"].get("type", "mse"),
        charbonnier_eps=float(config["loss"].get("charbonnier_eps", 1.0e-3)),
    )
    rtv_regularizer = RTVRegularizer(
        radius=int(config["loss"].get("rtv_radius", 2)),
        sigma=float(config["loss"].get("rtv_sigma", 2.0)),
        eps=float(config["loss"].get("rtv_eps", 1.0e-3)),
        reduction="mean",
    ).to(device)
    recon_weight = float(config["loss"].get("reconstruction_weight", config["loss"].get("charbonnier_weight", 1.0)))
    rtv_weight = float(config["loss"].get("rtv_weight", 0.0))
    entropy_weight = float(config["loss"].get("entropy_weight", 0.0))

    initial_lr = float(config["train"].get("initial_lr", config["train"].get("lr", 5.0e-4)))
    peak_lr = float(config["train"].get("peak_lr", initial_lr))
    final_lr = float(config["train"].get("final_lr", 0.0))
    warmup_ratio = float(config["train"].get("warmup_ratio", 0.2))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    train_amp = bool(config["train"].get("amp", True)) and device.type == "cuda"
    infer_amp = bool(config.get("infer", {}).get("amp", train_amp)) and device.type == "cuda"
    scaler = make_grad_scaler(train_amp)

    steps = int(config["train"]["steps"])
    patch_size = int(config["train"].get("patch_size", 0))
    log_interval = int(config["train"].get("log_interval", 50))
    save_interval = int(config["train"].get("save_interval", 500))
    iterator = trange(1, steps + 1, desc="train") if trange is not None else range(1, steps + 1)

    monitor_config = config.get("monitor", {})
    monitor_enabled = bool(monitor_config.get("enabled", False))
    monitor_interval = max(1, int(monitor_config.get("interval", 100)))
    monitor_metric = str(monitor_config.get("metric", "val_recon_loss"))
    monitor_min_delta = float(monitor_config.get("min_delta", 0.0))
    monitor_save_best = bool(monitor_config.get("save_best", True))
    monitor_batches = build_monitor_batches(image, monitor_config, int(config.get("seed", 42))) if monitor_enabled else []
    best_metric = float("inf")
    best_step = 0
    best_checkpoint_path = output_dir / "checkpoint_best.pt"

    history = []
    for step in iterator:
        model.train()
        current_lr = compute_learning_rate(
            step=step,
            total_steps=steps,
            initial_lr=initial_lr,
            peak_lr=peak_lr,
            final_lr=final_lr,
            warmup_ratio=warmup_ratio,
        )
        set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)

        train_image = random_crop_tensor(image, patch_size)
        with make_autocast(device, train_amp):
            pred, aux = model(train_image)

        # loss 用 FP32 计算，避免 AMP 下小 eps 和正则项数值不稳。
        pred_for_loss = pred.float()
        target_for_loss = train_image.float()
        recon_loss, _ = criterion(pred_for_loss, target_for_loss)
        if rtv_weight != 0.0:
            rtv_loss = rtv_regularizer(pred_for_loss)
        else:
            rtv_loss = pred_for_loss.new_tensor(0.0)
        entropy_loss = attention_entropy_regularizer(aux["attention_entropy"].float())
        loss = recon_weight * recon_loss + rtv_weight * rtv_loss + entropy_weight * entropy_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        val_stats = None
        if monitor_enabled and (step == 1 or step % monitor_interval == 0):
            val_stats = evaluate_monitor(
                model=model,
                monitor_batches=monitor_batches,
                criterion=criterion,
                rtv_regularizer=rtv_regularizer,
                recon_weight=recon_weight,
                rtv_weight=rtv_weight,
                entropy_weight=entropy_weight,
                device=device,
                amp_enabled=infer_amp,
            )
            metric_value = float(val_stats.get(monitor_metric, val_stats["val_recon_loss"]))
            is_best = metric_value < best_metric - monitor_min_delta
            if is_best:
                best_metric = metric_value
                best_step = step
                if monitor_save_best:
                    save_checkpoint(best_checkpoint_path, model, config, data.norm_meta, step)
            val_stats["monitor_metric"] = metric_value
            val_stats["is_best"] = is_best
            val_stats["best_step"] = best_step
            val_stats["best_metric"] = best_metric

        if step % log_interval == 0 or step == 1 or val_stats is not None:
            record = {
                "step": step,
                "lr": current_lr,
                "loss": float(loss.detach().cpu()),
                "recon_loss": float(recon_loss.detach().cpu()),
                "rtv_loss": float(rtv_loss.detach().cpu()),
                "attention_entropy": float(aux["attention_entropy"].detach().cpu()),
                "valid_query_ratio": float(aux["valid_query_ratio"].detach().cpu()),
            }
            if val_stats is not None:
                record.update(val_stats)
            history.append(record)
            message = (
                f"step={step:05d} loss={record['loss']:.6f} "
                f"lr={record['lr']:.6g} "
                f"recon={record['recon_loss']:.6f} "
                f"rtv={record['rtv_loss']:.6f} "
                f"entropy={record['attention_entropy']:.4f} "
                f"valid={record['valid_query_ratio']:.3f}"
            )
            if val_stats is not None:
                message += (
                    f" val_recon={record['val_recon_loss']:.6f} "
                    f"best_step={record['best_step']}"
                )
                if record["is_best"]:
                    message += " best=*"
            if trange is not None and hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(message)
            else:
                print(message)

        if save_interval > 0 and step % save_interval == 0:
            save_checkpoint(output_dir / f"checkpoint_{step:06d}.pt", model, config, data.norm_meta, step)

    save_checkpoint(output_dir / "checkpoint_final.pt", model, config, data.norm_meta, steps)

    used_best_for_output = False
    if monitor_enabled and monitor_save_best and best_checkpoint_path.exists():
        best_checkpoint = load_checkpoint(best_checkpoint_path, device)
        model.load_state_dict(best_checkpoint["model_state"])
        used_best_for_output = True

    model.eval()
    with torch.no_grad():
        tile_size = int(config.get("infer", {}).get("tile_size", 0))
        tile_overlap = int(config.get("infer", {}).get("tile_overlap", 32))
        tile_context = int(config.get("infer", {}).get("tile_context", tile_overlap))
        with make_autocast(device, infer_amp):
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
    save_preview(output_dir / "denoised_preview.png", denoised_raw)

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    monitor_summary = {
        "enabled": monitor_enabled,
        "metric": monitor_metric,
        "best_step": best_step if best_step > 0 else None,
        "best_metric": best_metric if best_step > 0 else None,
        "checkpoint_best": str(best_checkpoint_path) if best_checkpoint_path.exists() else None,
        "used_best_for_output": used_best_for_output,
    }
    with (output_dir / "monitor_summary.json").open("w", encoding="utf-8") as f:
        json.dump(monitor_summary, f, ensure_ascii=False, indent=2)

    print(f"训练完成。结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
