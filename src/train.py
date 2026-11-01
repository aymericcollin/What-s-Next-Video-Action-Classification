"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm
    python train.py experiment=timesformer_ssv2

Training supports automatic resume: if ``latest_checkpoint.pt`` exists in the
output directory, training continues from where it left off (safe to disconnect).

Pick an **experiment** under ``configs/experiment/``.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import hydra
import torch
import wandb
import torch.nn as nn
from torch.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.efficientnet_gru import EfficientNetGRU
from models.timesformer import TimeSformerSSV2
from models.videomae import VideoMAESSV2
from models.vjepa2 import VJEPA2SSV2
from models.vivit import ViViTSSV2
from utils import FocalLoss, build_transforms, set_seed, split_train_val


def build_model(cfg: DictConfig) -> nn.Module:
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=pretrained)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "timesformer":
        return TimeSformerSSV2(
            num_classes=num_classes,
            pretrained_model=cfg.model.get(
                "pretrained_model", "facebook/timesformer-base-finetuned-ssv2"
            ),
        )
    if name == "videomae":
        return VideoMAESSV2(
            num_classes=num_classes,
            pretrained_model=cfg.model.get(
                "pretrained_model", "MCG-NJU/videomae-base-finetuned-ssv2"
            ),
        )
    if name == "vjepa2":
        model = VJEPA2SSV2(
            num_classes=num_classes,
            pretrained_model=cfg.model.get(
                "pretrained_model", "facebook/vjepa2-vitl-fpc16-256-ssv2"
            ),
        )
        if cfg.model.get("gradient_checkpointing", False):
            model.model.gradient_checkpointing_enable()
        return model

    if name == "vivit":
        model = ViViTSSV2(
            num_classes=num_classes,
            pretrained_model=cfg.model.get(
                "pretrained_model", "google/vivit-b-16x2-kinetics400"
            ),
        )
        if cfg.model.get("gradient_checkpointing", False):
            model.model.gradient_checkpointing_enable()
        return model

    if name == "efficientnet_gru":
        return EfficientNetGRU(
            num_classes=num_classes,
            dropout=float(cfg.model.get("dropout", 0.3)),
        )

    raise ValueError(f"Unknown model.name: {name}")


def build_optimizer(
    model: nn.Module, cfg: DictConfig, head_only: bool = False
) -> torch.optim.Optimizer:
    """
    Pretrained transformers: either the head only (phase 1), or the full model with
    a lower backbone LR, decayed layer by layer except for TimeSformer (phase 2).
    Other models: a single parameter group with cfg.training.lr.
    """
    lr = float(cfg.training.lr)
    wd = float(cfg.training.get("weight_decay", 1e-4))
    lr_backbone = float(cfg.training.get("lr_backbone", lr * 0.1))
    lr_decay = float(cfg.training.get("lr_decay", 0.75))

    if hasattr(model, "classifier_parameters"):
        if head_only:
            return torch.optim.AdamW(model.classifier_parameters(), lr=lr, weight_decay=wd)
        if isinstance(model, TimeSformerSSV2):
            groups = model.backbone_parameter_groups(lr_head=lr, lr_backbone=lr_backbone)
        else:
            groups = model.layer_wise_lr_groups(lr_head=lr, lr_backbone=lr_backbone, lr_decay=lr_decay)
        return torch.optim.AdamW(groups, weight_decay=wd)

    if isinstance(model, EfficientNetGRU):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    return torch.optim.Adam(model.parameters(), lr=lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    if total_epochs <= 0:
        return None

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_mixup: bool = False,
    mixup_alpha: float = 0.8,
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        if use_mixup:
            video_batch, y_a, y_b, lam = mixup_batch(video_batch, labels, mixup_alpha, device)

        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(video_batch)
            if use_mixup:
                loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
            else:
                loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        # Accuracy tracked on original labels (before mix)
        ref_labels = y_a if use_mixup else labels
        correct += int((logits.argmax(dim=1) == ref_labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for video_batch, labels in data_loader:
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        with autocast("cuda"):
            logits = model(video_batch)
            loss = loss_fn(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def save_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    scaler: GradScaler,
    epoch: int,
    best_val_accuracy: float,
    cfg: DictConfig,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def save_best_checkpoint(
    path: Path,
    model: nn.Module,
    val_acc: float,
    cfg: DictConfig,
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_name": cfg.model.name,
        "num_classes": int(cfg.model.num_classes),
        "pretrained": bool(cfg.model.pretrained),
        "num_frames": int(cfg.dataset.num_frames),
        "val_accuracy": val_acc,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if cfg.model.name == "cnn_lstm":
        payload["lstm_hidden_size"] = int(cfg.model.get("lstm_hidden_size", 512))
    if cfg.model.name == "timesformer":
        payload["pretrained_model"] = cfg.model.get(
            "pretrained_model", "facebook/timesformer-base-finetuned-ssv2"
        )
    torch.save(payload, path)


def build_weighted_sampler(
    samples: List,
    hard_class_weights: Optional[Dict[int, float]] = None,
) -> WeightedRandomSampler:
    labels = [label for _, label in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    if hard_class_weights:
        weights = [w * hard_class_weights.get(label, 1.0) for w, label in zip(weights, labels)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def load_data(
    cfg: DictConfig,
    train_transform,
    eval_transform,
) -> Tuple[List, List]:
    train_dir = Path(cfg.dataset.train_dir).resolve()
    val_ratio = float(cfg.dataset.val_ratio)
    merge_val = bool(cfg.dataset.get("merge_val_to_train", False))

    if merge_val and cfg.dataset.get("val_dir"):
        val_dir = Path(cfg.dataset.val_dir).resolve()
        all_samples = collect_video_samples(train_dir) + collect_video_samples(val_dir)
        print(f"Merged train+val: {len(all_samples)} total samples")
        train_samples, val_samples = split_train_val(
            all_samples, val_ratio=val_ratio, seed=int(cfg.dataset.seed)
        )
    elif val_ratio <= 0.0 and cfg.dataset.get("val_dir"):
        val_dir = Path(cfg.dataset.val_dir).resolve()
        train_samples = collect_video_samples(train_dir)
        val_samples = collect_video_samples(val_dir)
        print(f"Using dedicated val dir: {val_dir} ({len(val_samples)} samples)")
    else:
        all_samples = collect_video_samples(train_dir)
        max_samples = cfg.dataset.get("max_samples")
        if max_samples is not None:
            all_samples = all_samples[: int(max_samples)]
        train_samples, val_samples = split_train_val(
            all_samples,
            val_ratio=val_ratio,
            seed=int(cfg.dataset.seed),
        )

    return train_samples, val_samples


def _init_wandb(cfg: DictConfig) -> bool:
    wcfg = cfg.get("wandb", {})
    if not bool(wcfg.get("enabled", False)):
        return False
    try:
        model_name = cfg.model.name
        run_name = wcfg.get("name") or f"{model_name}"
        wandb.init(
            project=str(wcfg.get("project", "csc43m04-challenge")),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        return True
    except Exception as e:
        print(f"[wandb] init failed ({e}), continuing without wandb.")
        return False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.dataset.seed))
    use_wandb = _init_wandb(cfg)

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    use_imagenet_norm = bool(cfg.model.pretrained)
    random_crop = bool(cfg.dataset.get("random_crop", False))
    color_jitter = bool(cfg.dataset.get("color_jitter", False))
    rand_augment = bool(cfg.dataset.get("rand_augment", False))
    random_erasing = bool(cfg.dataset.get("random_erasing", False))
    use_rrc = bool(cfg.dataset.get("use_rrc", False))
    image_size = int(cfg.dataset.get("image_size", 224))
    train_transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=use_imagenet_norm,
        random_crop=random_crop,
        color_jitter=color_jitter,
        rand_augment=rand_augment,
        random_erasing=random_erasing,
        use_rrc=use_rrc,
    )
    eval_transform = build_transforms(
        image_size=image_size,
        is_training=False,
        use_imagenet_norm=use_imagenet_norm,
        random_crop=random_crop,
    )

    train_dir = Path(cfg.dataset.train_dir).resolve()
    train_samples, val_samples = load_data(cfg, train_transform, eval_transform)
    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")

    num_frames = int(cfg.dataset.num_frames)
    hflip_prob = float(cfg.dataset.get("hflip_prob", 0.0))
    hflip_map_cfg = cfg.dataset.get("hflip_label_map", None)
    hflip_label_map = {int(k): int(v) for k, v in hflip_map_cfg.items()} if hflip_map_cfg else {}
    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=num_frames,
        transform=train_transform,
        sample_list=train_samples,
        hflip_prob=hflip_prob,
        hflip_label_map=hflip_label_map,
    )
    val_root = (
        Path(cfg.dataset.val_dir).resolve()
        if float(cfg.dataset.val_ratio) <= 0.0 and cfg.dataset.get("val_dir")
        else train_dir
    )
    val_dataset = VideoFrameDataset(
        root_dir=val_root,
        num_frames=num_frames,
        transform=eval_transform,
        sample_list=val_samples,
    )

    batch_size = int(cfg.training.batch_size)
    num_workers = int(cfg.training.num_workers)
    use_weighted = bool(cfg.training.get("weighted_sampler", False))
    hard_class_weights_cfg = cfg.dataset.get("hard_class_weights", None)
    hard_class_weights = {int(k): float(v) for k, v in hard_class_weights_cfg.items()} if hard_class_weights_cfg else None
    if use_weighted:
        sampler = build_weighted_sampler(train_samples, hard_class_weights=hard_class_weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    label_smoothing = float(cfg.training.get("label_smoothing", 0.0))
    if cfg.training.get("focal_loss", False):
        gamma = float(cfg.training.get("focal_gamma", 2.0))
        loss_fn = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
        print(f"Using FocalLoss(gamma={gamma}, label_smoothing={label_smoothing})")
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler("cuda")
    use_mixup = bool(cfg.training.get("use_mixup", False))
    mixup_alpha = float(cfg.training.get("mixup_alpha", 0.8))
    grad_clip = float(cfg.training.get("grad_clip", 0.0))

    head_only_epochs = int(cfg.training.get("head_only_epochs", 0))
    total_epochs = int(cfg.training.epochs)
    warmup_epochs = int(cfg.training.get("warmup_epochs", 0))
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    resume_path = Path(cfg.training.get("resume_path", str(checkpoint_path.parent / "latest_checkpoint.pt"))).resolve()

    start_epoch = 0
    best_val_accuracy = 0.0
    resumed_in_phase2 = False

    if resume_path.exists():
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_accuracy = ckpt.get("best_val_accuracy", 0.0)
        resumed_in_phase2 = start_epoch >= head_only_epochs
        print(f"  Resuming at epoch {start_epoch + 1}/{total_epochs} | best val acc so far: {best_val_accuracy:.4f}")

    if head_only_epochs > 0 and start_epoch < head_only_epochs:
        print(f"\n=== Phase 1: head-only training (epochs 1–{head_only_epochs}) ===")
        optimizer = build_optimizer(model, cfg, head_only=True)
        scheduler = build_scheduler(optimizer, head_only_epochs, warmup_epochs=0)

        if resume_path.exists() and not resumed_in_phase2:
            ckpt = torch.load(resume_path, map_location=device)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        for epoch in range(start_epoch, head_only_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, scaler,
                use_mixup=use_mixup, mixup_alpha=mixup_alpha, grad_clip=grad_clip,
            )
            val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
            if scheduler:
                scheduler.step()

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[P1] Epoch {epoch + 1}/{head_only_epochs} | "
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr_now:.2e}"
            )
            if use_wandb:
                wandb.log({"phase": 1, "epoch": epoch + 1, "train/loss": train_loss,
                           "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc,
                           "lr": lr_now})

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                save_best_checkpoint(checkpoint_path, model, val_acc, cfg)
                print(f"  → New best model saved (val acc={val_acc:.4f})")
                if use_wandb:
                    wandb.run.summary["best_val_acc"] = val_acc

            save_resume_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_accuracy, cfg
            )

        start_epoch = head_only_epochs

    print(f"\n=== Phase 2: full fine-tuning (epochs {head_only_epochs + 1}–{total_epochs}) ===")
    optimizer = build_optimizer(model, cfg, head_only=False)
    scheduler = build_scheduler(
        optimizer, total_epochs - head_only_epochs, warmup_epochs=warmup_epochs
    )

    if resume_path.exists() and resumed_in_phase2:
        ckpt = torch.load(resume_path, map_location=device)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            print("  (optimizer state incompatible with phase 2 — starting fresh optimizer)")

    for epoch in range(start_epoch, total_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, loss_fn, device)
        if scheduler:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[P2] Epoch {epoch + 1}/{total_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr_now:.2e}"
        )
        if use_wandb:
            wandb.log({"phase": 2, "epoch": epoch + 1, "train/loss": train_loss,
                       "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc,
                       "lr": lr_now})

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_best_checkpoint(checkpoint_path, model, val_acc, cfg)
            print(f"  → New best model saved (val acc={val_acc:.4f})")
            if use_wandb:
                wandb.run.summary["best_val_acc"] = val_acc

        save_resume_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_accuracy, cfg
        )

    print(f"\nDone. Best validation accuracy: {best_val_accuracy:.4f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
