"""
VideoMAE masked autoencoder pretraining from scratch (closed world track).

Run from the ``src/`` directory::

    python pretrain.py
    python pretrain.py pretraining.epochs=300

Saves the full VideoMAEForPreTraining model every ``save_every`` epochs and at the end.
The saved checkpoint can be used for downstream fine-tuning via ``train.py experiment=videomae_closed``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
from transformers import VideoMAEConfig, VideoMAEForPreTraining

from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from utils import build_transforms, set_seed


def build_bool_masked_pos(
    batch_size: int, num_patches: int, mask_ratio: float, device: torch.device
) -> torch.BoolTensor:
    """Random tube masking: each batch item gets an independent random mask."""
    num_masked = int(num_patches * mask_ratio)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for i in range(batch_size):
        perm = torch.randperm(num_patches, device=device)[:num_masked]
        mask[i, perm] = True
    return mask


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain_one_epoch(
    model: VideoMAEForPreTraining,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    num_patches: int,
    mask_ratio: float,
) -> float:
    model.train()
    total_loss = 0.0

    for frames, _ in loader:
        frames = frames.to(device)
        bool_masked_pos = build_bool_masked_pos(
            frames.shape[0], num_patches, mask_ratio, device
        )

        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(pixel_values=frames, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@hydra.main(config_path="configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.dataset.get("seed", 42))

    wcfg = cfg.get("wandb", {})
    use_wandb = bool(wcfg.get("enabled", False))
    if use_wandb:
        try:
            wandb.init(
                project=str(wcfg.get("project", "csc43m04-challenge")),
                name=str(wcfg.get("name", "videomae-pretrain")),
                config=OmegaConf.to_container(cfg, resolve=True),
                resume="allow",
            )
        except Exception as e:
            print(f"[wandb] init failed ({e}), continuing without wandb.")
            use_wandb = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = int(cfg.dataset.image_size)
    num_frames = int(cfg.dataset.num_frames)
    transform = build_transforms(
        image_size=image_size,
        is_training=True,
        use_imagenet_norm=True,
        random_crop=True,
        color_jitter=True,
    )

    train_samples = collect_video_samples(Path(cfg.dataset.train_dir))
    datasets = [VideoFrameDataset(cfg.dataset.train_dir, num_frames, transform, train_samples)]

    val_dir = cfg.dataset.get("val_dir", None)
    if val_dir:
        val_samples = collect_video_samples(Path(val_dir))
        datasets.append(VideoFrameDataset(val_dir, num_frames, transform, val_samples))

    dataset = ConcatDataset(datasets)
    print(f"Pretraining on {len(dataset)} clips (train + val combined)")

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.pretraining.batch_size),
        shuffle=True,
        num_workers=int(cfg.pretraining.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    config = VideoMAEConfig(
        image_size=image_size,
        patch_size=16,
        num_channels=3,
        num_frames=num_frames,
        tubelet_size=2,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        decoder_num_attention_heads=16,
        decoder_hidden_size=384,
        decoder_num_hidden_layers=4,
        decoder_intermediate_size=1536,
        norm_pix_loss=True,
    )
    model = VideoMAEForPreTraining(config).to(device)

    num_patches = (num_frames // config.tubelet_size) * (image_size // config.patch_size) ** 2
    print(f"num_patches={num_patches}, mask_ratio={cfg.pretraining.mask_ratio}")
    print(f"Encoder params: {sum(p.numel() for p in model.videomae.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.pretraining.lr),
        weight_decay=float(cfg.pretraining.weight_decay),
    )
    scheduler = build_scheduler(
        optimizer,
        total_epochs=int(cfg.pretraining.epochs),
        warmup_epochs=int(cfg.pretraining.warmup_epochs),
    )
    scaler = GradScaler()

    checkpoint_dir = Path(cfg.pretraining.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / "resume_state.pt"
    start_epoch = 0

    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        print(f"Resumed from epoch {start_epoch}")

    epochs = int(cfg.pretraining.epochs)
    save_every = int(cfg.pretraining.save_every)
    mask_ratio = float(cfg.pretraining.mask_ratio)

    for epoch in range(start_epoch, epochs):
        avg_loss = pretrain_one_epoch(
            model, loader, optimizer, scaler, device, num_patches, mask_ratio
        )
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Pretrain] Epoch {epoch + 1}/{epochs} | loss {avg_loss:.4f} | lr {lr_now:.2e}")
        if use_wandb:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, "lr": lr_now})

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            resume_path,
        )

        if (epoch + 1) % save_every == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:04d}"
            model.save_pretrained(ckpt_path)
            print(f"  → Saved checkpoint to {ckpt_path}")

    model.save_pretrained(checkpoint_dir / "final")
    print(f"Pretraining done. Final checkpoint at {checkpoint_dir / 'final'}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
