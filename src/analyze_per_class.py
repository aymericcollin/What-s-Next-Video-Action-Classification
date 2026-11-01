#!/usr/bin/env python3
"""Per-class accuracy analysis on the validation set."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms

CHECKPOINT = "checkpoints/vjepa2/best_model.pt"
VAL_DIR    = "processed_data/val"
BATCH_SIZE = 16
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def get_class_names(val_root: Path) -> dict[int, str]:
    names = {}
    for d in sorted(val_root.iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            idx = int(parts[0])
            names[idx] = parts[1].replace("_", " ")
    return names


def main():
    device = torch.device(DEVICE)
    val_root = Path(VAL_DIR).resolve()

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    cfg  = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    image_size = int(cfg.dataset.get("image_size", 224))
    num_frames  = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    transform   = build_transforms(image_size=image_size, is_training=False, use_imagenet_norm=True)

    sample_list = collect_video_samples(val_root)

    dataset = VideoFrameDataset(
        root_dir=val_root, num_frames=num_frames,
        transform=transform, sample_list=sample_list,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=(device.type == "cuda"))

    print(f"Running inference on {len(dataset)} val clips...")
    all_preds, all_labels = [], []
    # track top-5 confusions per class: class -> {predicted_class: count}
    confusions: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for i, (videos, labels) in enumerate(loader, 1):
            logits = model(videos.to(device))
            probs  = F.softmax(logits.float(), dim=1).cpu()
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            for pred, label in zip(preds.tolist(), labels.tolist()):
                if pred != label:
                    confusions[label][pred] += 1
            if i % (len(loader) // 10 or 1) == 0:
                print(f"  {i}/{len(loader)}", flush=True)

    # Per-class accuracy
    class_correct: dict[int, int] = defaultdict(int)
    class_total:   dict[int, int] = defaultdict(int)
    for pred, label in zip(all_preds, all_labels):
        class_total[label]   += 1
        class_correct[label] += int(pred == label)

    class_names = get_class_names(val_root)
    overall = sum(class_correct.values()) / sum(class_total.values())
    print(f"\nOverall val accuracy: {overall:.4f} ({sum(class_correct.values())}/{sum(class_total.values())})\n")

    # Sort by accuracy ascending (worst first)
    rows = []
    for cls in sorted(class_total.keys()):
        acc = class_correct[cls] / class_total[cls]
        rows.append((acc, cls, class_correct[cls], class_total[cls]))
    rows.sort()

    print(f"{'Cls':>3}  {'Acc':>6}  {'Correct':>7}  {'Total':>5}  {'Class name':<45}  Top confusions")
    print("-" * 110)
    for acc, cls, correct, total in rows:
        name = class_names.get(cls, f"class_{cls}")[:44]
        top_conf = sorted(confusions[cls].items(), key=lambda x: -x[1])[:3]
        conf_str = "  ".join(f"{class_names.get(c,'?')[:20]}({n})" for c, n in top_conf)
        marker = " <<" if acc < 0.50 else (" <" if acc < 0.65 else "")
        print(f"{cls:>3}  {acc:>6.1%}  {correct:>7}  {total:>5}  {name:<45}  {conf_str}{marker}")

    print("\n--- Summary ---")
    below_50 = [(acc, cls) for acc, cls, *_ in rows if acc < 0.50]
    print(f"Classes below 50%: {len(below_50)}")
    for acc, cls in below_50:
        print(f"  {cls:>3} {class_names.get(cls,'?')} — {acc:.1%}")


if __name__ == "__main__":
    main()
