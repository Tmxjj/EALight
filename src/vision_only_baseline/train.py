"""Train the vision-only image/action baseline. See the module README for data and CLI examples."""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

# 添加项目根目录到 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.vision_only_baseline.model import build_model, MAX_PHASES
from src.vision_only_baseline.dataset import build_dataloader, DURATION_CANDIDATES


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> dict:
    """
    训练一个 epoch。
    
    Args:
        model: VisionOnlyTSC 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        device: 计算设备
        epoch: 当前 epoch 编号
        log_interval: 日志打印间隔（步数）
    Returns:
        包含 loss、accuracy 等指标的字典
    """
    model.train()
    total_loss = 0.0
    phase_correct = 0
    duration_correct = 0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    # 进度条（tqdm）：实时显示 batch 进度与累计指标
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (images, phase_labels, duration_labels) in enumerate(pbar):
        # images: [B, 4, C, H, W]
        # phase_labels: [B] (int)
        # duration_labels: [B] (int, duration index)
        images = images.to(device, non_blocking=True)
        phase_labels = phase_labels.to(device=device, dtype=torch.long, non_blocking=True)
        duration_labels = duration_labels.to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()

        # 前向传播
        phase_logits, duration_logits = model(images)

        # 计算损失：CE(phase) + CE(duration)
        loss_phase = ce_loss(phase_logits, phase_labels)
        loss_duration = ce_loss(duration_logits, duration_labels)
        loss = loss_phase + loss_duration

        # 反向传播
        loss.backward()
        # 梯度裁剪，防止训练不稳定
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计指标
        bs = images.size(0)
        total_loss += loss.item() * bs
        phase_pred = phase_logits.argmax(dim=-1)
        duration_pred = duration_logits.argmax(dim=-1)
        phase_correct += (phase_pred == phase_labels).sum().item()
        duration_correct += (duration_pred == duration_labels).sum().item()
        total_samples += bs

        # 实时刷新进度条右侧指标（去除周期性 logger 刷屏）
        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "ph_acc": f"{phase_correct / total_samples * 100:.1f}%",
            "dur_acc": f"{duration_correct / total_samples * 100:.1f}%",
        })

    if total_samples == 0:
        raise ValueError("Cannot compute metrics from an empty dataloader")
    metrics = {
        "loss": total_loss / total_samples,
        "phase_acc": phase_correct / total_samples * 100,
        "duration_acc": duration_correct / total_samples * 100,
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    验证集评估。
    
    Args:
        model: VisionOnlyTSC 模型
        dataloader: 验证数据加载器
        device: 计算设备
    Returns:
        包含 loss、accuracy 等指标的字典
    """
    model.eval()
    total_loss = 0.0
    phase_correct = 0
    duration_correct = 0
    total_samples = 0

    ce_loss = nn.CrossEntropyLoss()

    # 进度条（tqdm）
    pbar = tqdm(dataloader, desc="[Val]", leave=False)
    for images, phase_labels, duration_labels in pbar:
        images = images.to(device, non_blocking=True)
        phase_labels = phase_labels.to(device=device, dtype=torch.long, non_blocking=True)
        duration_labels = duration_labels.to(device=device, dtype=torch.long, non_blocking=True)

        phase_logits, duration_logits = model(images)

        loss_phase = ce_loss(phase_logits, phase_labels)
        loss_duration = ce_loss(duration_logits, duration_labels)
        loss = loss_phase + loss_duration

        bs = images.size(0)
        total_loss += loss.item() * bs
        phase_pred = phase_logits.argmax(dim=-1)
        duration_pred = duration_logits.argmax(dim=-1)
        phase_correct += (phase_pred == phase_labels).sum().item()
        duration_correct += (duration_pred == duration_labels).sum().item()
        total_samples += bs

        pbar.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "ph_acc": f"{phase_correct / total_samples * 100:.1f}%",
            "dur_acc": f"{duration_correct / total_samples * 100:.1f}%",
        })

    if total_samples == 0:
        raise ValueError("Cannot compute metrics from an empty dataloader")
    metrics = {
        "loss": total_loss / total_samples,
        "phase_acc": phase_correct / total_samples * 100,
        "duration_acc": duration_correct / total_samples * 100,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Only TSC Baseline 训练脚本"
    )
    # ── 数据参数 ──
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据文件路径，支持 JSON Array(.json) 或 JSON Lines(.jsonl，每行一个样本)")
    parser.add_argument("--val_path", type=str, default=None,
                        help="验证数据文件路径，支持 .json / .jsonl（可选）")
    parser.add_argument("--num_phases", type=int, default=4,
                        help="训练相位数固定为 4；2/3 相位掩码仅用于推理")

    # ── 训练参数 ──
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载线程数")
    parser.add_argument("--image_size", type=int, default=224,
                        help="输入图像尺寸")

    # ── 输出参数 ──
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--backbone_ckpt", type=str, default=None,
                        help="本地 EfficientFormerV2-S1 权重路径（.pth）。指定后离线加载，无需联网；"
                             "不指定则 timm 自动联网下载 ImageNet 预训练权重")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="模型保存间隔（epoch 数）")
    parser.add_argument("--resume", type=str, default=None,
                        help="断点续训：从指定 checkpoint 路径恢复模型与优化器状态（可选）")

    # ── 设备参数 ──
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="单卡时的 GPU 编号（与 --gpus 二选一，--gpus 优先）")
    parser.add_argument("--gpus", type=str, default=None,
                        help="多卡并行 GPU 列表，逗号分隔，如 '0,1'（双卡）。"
                             "指定后启用 DataParallel，覆盖 --gpu_id")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    parser.add_argument("--image_root", default=None,
                        help="Root for relative image paths; defaults to the JSON file directory")
    args = parser.parse_args()
    if args.num_phases != MAX_PHASES:
        parser.error(f"Training uses {MAX_PHASES} phases; scene-specific masking is inference-only")
    if args.image_size != 224:
        parser.error("EfficientFormerV2-S1 requires --image_size 224")
    if min(args.epochs, args.batch_size, args.save_interval, args.log_interval) < 1:
        parser.error("epochs, batch_size, save_interval and log_interval must be positive")
    if args.num_workers < 0:
        parser.error("num_workers must be nonnegative")
    if args.resume and not os.path.isfile(args.resume):
        parser.error(f"Resume checkpoint does not exist: {args.resume}")

    # ── 设置随机种子 ──
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 设置设备（支持单卡 / 多卡并行）──
    if torch.cuda.is_available():
        # 解析 GPU 列表：优先使用 --gpus，否则回退到 --gpu_id
        if args.gpus:
            gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        else:
            gpu_ids = [args.gpu_id]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(
            g < 0 or g >= torch.cuda.device_count() for g in gpu_ids
        ):
            parser.error("GPU IDs must be distinct and within the process-visible device range")
        device = torch.device(f"cuda:{gpu_ids[0]}")
        if len(gpu_ids) > 1:
            logger.info(
                f"[Train] 多卡并行 (DataParallel): GPU {gpu_ids} | "
                f"主卡: {torch.cuda.get_device_name(gpu_ids[0])}"
            )
        else:
            logger.info(f"[Train] 单卡训练: {torch.cuda.get_device_name(gpu_ids[0])}")
    else:
        gpu_ids = []
        device = torch.device("cpu")
        logger.warning("[Train] CUDA 不可用，使用 CPU 训练（速度较慢）")

    # ── 创建输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 构建模型 ──
    logger.info(f"[Train] 构建模型: EfficientFormerV2-S1 + AttentionPooling + DualHead")
    logger.info(
        f"[Train] 训练相位数固定为 MAX_PHASES={MAX_PHASES}（与 Syn-Train 对齐）\n"
        f"[Train] 推理时通过 Masked Inference 支持 2/3/4 相位场景 zero-shot 迁移\n"
        f"[Train] num_durations={len(DURATION_CANDIDATES)}, 候选集={DURATION_CANDIDATES}"
    )
    model = build_model(
        pretrained=not bool(args.resume),
        backbone_ckpt=None if args.resume else args.backbone_ckpt,  # None 则联网下载；指定本地 .pth 则离线加载
    )  # 固定 MAX_PHASES=4，无需传入 num_phases
    model = model.to(device)

    # ── 多卡并行封装（DataParallel）──
    # 仅当检测到 >1 张可用 GPU 时才启用，对单卡/CPU 透明
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        logger.info(f"[Train] 已启用 DataParallel，模型复制到 GPU {gpu_ids}")

    # DataParallel 包裹后 state_dict 会带 'module.' 前缀，统一剥离后再保存，
    # 保证 eval_agent.py 用普通模型 load_state_dict 可直接加载（避免前缀不匹配）
    def get_state_dict(m):
        return m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict()

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[Train] 模型参数: 总计 {total_params / 1e6:.2f}M | "
        f"可训练 {trainable_params / 1e6:.2f}M"
    )

    # ── 构建数据加载器 ──
    logger.info(f"[Train] 加载训练数据: {args.data_path}")
    train_loader = build_dataloader(
        data_path=args.data_path,
        num_phases=MAX_PHASES,  # 固定 4，与 Syn-Train 一致
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        shuffle=True,
        image_size=args.image_size,
        image_root=args.image_root,
    )

    val_loader = None
    if args.val_path:
        logger.info(f"[Train] 加载验证数据: {args.val_path}")
        val_loader = build_dataloader(
            data_path=args.val_path,
            num_phases=MAX_PHASES,  # 固定 4
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=False,
            shuffle=False,
            image_size=args.image_size,
            image_root=args.image_root,
        )

    # ── 优化器与调度器 ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── 断点续训（可选）──
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        if not os.path.exists(args.resume):
            logger.warning(f"[Train] 指定的 resume 文件不存在: {args.resume}，将从头训练")
        else:
            logger.info(f"[Train] 加载断点: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
            previous_config = ckpt.get("config", {})
            if previous_config.get("image_size", 224) != args.image_size:
                raise ValueError("Resume image_size differs from the checkpoint")
            # 兼容 DataParallel 保存的 'module.' 前缀
            state_dict = ckpt.get("model_state_dict", ckpt)
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
            (model.module if isinstance(model, torch.nn.DataParallel) else model).load_state_dict(state_dict)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                logger.warning("Legacy checkpoint has no scheduler state; LR schedule restarts")
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            start_epoch = ckpt.get("epoch", 0) + 1
            if start_epoch > args.epochs:
                raise ValueError("--epochs must exceed the resumed checkpoint epoch")
            logger.info(f"[Train] 已从 epoch {ckpt.get('epoch', '?')} 恢复，继续训练至 {args.epochs}")

    # ── 训练循环 ──
    logger.info(f"[Train] 开始训练: {args.epochs} epochs, batch_size={args.batch_size}")
    logger.info(f"[Train] 时长候选集: {DURATION_CANDIDATES}")
    logger.info(
        f"[Train] Zero-Shot 设计：训练仅用 Syn-Train(4相位)，"
        f"测试时 Masked Inference 自动适配 2/3/4 相位场景"
    )

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Training", unit="epoch"):
        epoch_start = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        # 学习率调度
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start
        logger.info(
            f"[Train] Epoch {epoch}/{args.epochs} 完成 | "
            f"耗时: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Phase Acc: {train_metrics['phase_acc']:.1f}% | "
            f"Duration Acc: {train_metrics['duration_acc']:.1f}%"
        )

        # 验证
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            logger.info(
                f"[Val] Epoch {epoch} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Phase Acc: {val_metrics['phase_acc']:.1f}% | "
                f"Duration Acc: {val_metrics['duration_acc']:.1f}%"
            )

            # 保存最优模型
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": get_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss": best_val_loss,
                    "val_phase_acc": val_metrics["phase_acc"],
                    "val_duration_acc": val_metrics["duration_acc"],
                    "num_phases": MAX_PHASES,  # 始终保存 MAX_PHASES=4
                    "config": vars(args),
                }, best_path)
                logger.info(f"[Train] ✓ 最优模型已保存: {best_path} (val_loss={best_val_loss:.4f})")

        # 定期保存 checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": get_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_metrics": train_metrics,
                "num_phases": args.num_phases,
                "config": vars(args),
            }, ckpt_path)
            logger.info(f"[Train] Checkpoint 已保存: {ckpt_path}")

    # ── 保存最终模型 ──
    final_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": get_state_dict(model),
        "num_phases": MAX_PHASES,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": vars(args),
    }, final_path)
    logger.info(f"[Train] 训练完成！最终模型: {final_path}")


if __name__ == "__main__":
    main()
