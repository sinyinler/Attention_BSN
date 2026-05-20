# Attention-BSN for Single BFI Denoising

这是一个面向**单张 BFI 图自监督降噪**的 attention-BSN 工程模板。代码按工程结构拆分为 `models/`、`losses/`、`datasets/`、`utils/`、`train.py`、`test.py`，方便后续继续改模型和做消融实验。

## 方法概述

核心目标是让预测像素 `x0` 不依赖 `x0` 及其附近相关噪声邻域，同时用非局部相似 patch 补回大盲区损失的信息。

- `models/masked_layers.py`：环形盲点卷积，Q 分支只看中心半径外的外环信息。
- `models/nonlocal_attention.py`：下采样候选池 + top-k + 分块 query 的稀疏非局部 attention。
- `models/attention_bsn.py`：attention-BSN 主模型，融合 local blind feature 和 non-local feature。
- `losses/bsn_loss.py`：整图周期盲点采样的自监督 MSE，以及 attention 熵正则。
- `scripts/check_j_invariance.py`：扰动中心盲区，检查目标像素预测是否变化。

默认配置里 `blind_radius=8`，K/V 分支有两层 `3x3` 卷积，因此 attention 的实际候选排除半径是 `8 + 2 = 10` 像素，避免 K/V 的局部感受野把中心信息间接带回来。

## 安装依赖

```bash
pip install -r requirements.txt
```

如果你的 Python 版本太新导致 PyTorch 没有对应 wheel，建议用 Python 3.10 或 3.11 创建环境。

## 训练单张 BFI 图

```bash
python train.py \
  --image path/to/bfi.npy \
  --output-dir runs/bfi_attention_bsn \
  --steps 3000
```

支持输入格式：

- `.npy` / `.npz`
- `.png` / `.tif` / `.jpg` 等 Pillow 可读取的图像

训练输出：

- `denoised.npy`：还原到原始 BFI 数值尺度的降噪结果
- `denoised_preview.tif`：16-bit 预览图
- `checkpoint_final.pt`：最终模型
- `history.json`：训练过程中的 loss、attention entropy 等日志
- `resolved_config.json`：本次实际使用的配置

大图显存不足时，优先调这些参数：

- `train.patch_size`：训练随机裁剪尺寸，默认 `256`。OOM 时先改成 `192` 或 `128`。
- `infer.tile_size`：最终整图输出的滑窗尺寸，默认 `256`。推理 OOM 时同步调小。
- `infer.tile_context`：滑窗推理额外上下文半径，默认 `64`。它能明显减轻拼接伪影。
- `model.base_channels`：默认 `24`，越小越省显存。
- `model.attention_topk`：默认 `8`，越小越省显存。
- `model.candidate_stride`：默认 `8`，越大越省显存。

也可以直接用命令行覆盖：

```bash
python train.py \
  --image path/to/bfi.npy \
  --output-dir runs/bfi_attention_bsn \
  --steps 3000 \
  --patch-size 192 \
  --tile-size 192 \
  --tile-context 64
```

如果最终推理仍能放进显存，可以用 `--tile-size 0` 做整图前向，完全避免滑窗拼接；如果整图前向 OOM，再使用默认的 halo 滑窗。

## 推理 / 测试

```bash
python test.py \
  --image path/to/bfi.npy \
  --checkpoint runs/bfi_attention_bsn/checkpoint_final.pt \
  --output runs/bfi_attention_bsn/test_denoised.npy
```

如果有长时间窗口 BFI 或其它 clean reference，可以加 `--gt` 计算 PSNR/SSIM：

```bash
python test.py \
  --image path/to/noisy_bfi.npy \
  --checkpoint runs/bfi_attention_bsn/checkpoint_final.pt \
  --output runs/bfi_attention_bsn/test_denoised.npy \
  --gt path/to/reference_bfi.npy
```

## 检查 J-invariance

训练前后都建议跑一次：

```bash
python scripts/check_j_invariance.py \
  --checkpoint runs/bfi_attention_bsn/checkpoint_final.pt \
  --image path/to/bfi.npy
```

这个脚本会随机选若干像素，只扰动该像素 `blind_radius` 半径内的输入，再比较该像素输出是否改变。最大变化接近 `0` 才说明中心盲区约束基本成立。

## 关于多卡

这个项目默认是单图训练，batch size 等于 1。常规 `DataParallel/DDP` 是按 batch 维度切分，多卡不会把一张大图自动拆到两张显卡上，因此对当前 OOM 帮助很小。更推荐使用 `train.patch_size` 做随机 patch 训练，用 `infer.tile_size` 做滑窗整图推理。

## 常用配置

配置文件在 `configs/attention_bsn_default.json`。

几个最重要的参数：

- `model.blind_radius`：中心盲区半径。BFI 的相关核 FWHM 约 4 像素时，建议从 7-8 开始。
- `model.annulus_width`：Q 分支外环宽度。太小信息少，太大局部性弱。
- `model.candidate_stride`：K/V 候选池下采样步长。越小越准，越大越省显存。
- `model.attention_topk`：每个 query 使用的非局部候选数量。
- `loss.grid_period`：周期盲点采样间隔，默认 `5`，大约每步监督 4% 像素。
- `loss.entropy_weight`：attention 熵正则权重，默认很小，只用于抑制“噪声相似”的过尖锐 attention。

## 当前实现边界

这个版本优先保证结构清晰和 J-invariance 逻辑可检查。它没有加入血管 mask、预训练相似度 encoder 或 INR 坐标解码器；这些适合作为下一步消融扩展。单图 BSN 的合理目标是明显优于普通 BSN，并尽量接近 N2N baseline，而不是稳定超过 N2N。
