#!/usr/bin/env python3
"""
Ensemble several checkpoints and write a submission CSV.

Each checkpoint is given as ``path[:weight]`` (default weight 1.0). Two ways to
combine the per-model class probabilities p_i:

* ``mean``     weighted arithmetic mean, sum_i w_i * p_i
* ``logspace`` weighted geometric mean, sum_i w_i * log p_i. With ``--fit-weights``
               the weights are fitted on the validation set (Nelder-Mead on the
               ensemble cross-entropy), which helps when models are calibrated
               differently.

Per-model log-probabilities are cached in ``--cache-dir`` so that changing the
weights does not require a new GPU pass.

Examples (from ``src/``)::

    # Final open-world submission: 3 V-JEPA 2 ViT-L runs
    python ensemble.py \\
        --ckpt checkpoints/vjepa2_balanced/best_model.pt:1.0 \\
        --ckpt checkpoints/vjepa2/best_model.pt:0.9 \\
        --ckpt checkpoints/vjepa2_run2/best_model.pt:0.7

    # Fit the weights on the validation set
    python ensemble.py --method logspace --fit-weights \\
        --ckpt checkpoints/vjepa2/best_model.pt \\
        --ckpt checkpoints/vjepa2_balanced/best_model.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.optimize import minimize
from torch.utils.data import DataLoader

from create_submission import discover_all_test_videos, run_inference_probs
from dataset.video_dataset import VideoFrameDataset, collect_video_samples
from train import build_model
from utils import build_transforms, build_tta_transforms, set_seed


def parse_ckpt(spec: str) -> Tuple[Path, float]:
    path, sep, weight = spec.rpartition(":")
    if sep and weight.replace(".", "", 1).isdigit():
        return Path(path), float(weight)
    return Path(spec), 1.0


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    val_acc = float(ckpt.get("val_accuracy", ckpt.get("best_val_accuracy", 0.0)))
    print(f"Loaded {path} ({cfg.model.name}, val acc {val_acc:.4f})", flush=True)
    return model, cfg


def predict_log_probs(
    model: torch.nn.Module,
    cfg,
    samples: List[Tuple[Path, int]],
    root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Log of the softmax probabilities, averaged over TTA crops if enabled."""
    image_size = int(cfg.dataset.get("image_size", 224))
    num_frames = int(cfg.dataset.get("num_frames", 8))
    if args.tta:
        transforms = build_tta_transforms(image_size=image_size, use_imagenet_norm=True)
    else:
        transforms = [build_transforms(image_size=image_size, is_training=False, use_imagenet_norm=True)]

    probs = None
    for transform in transforms:
        dataset = VideoFrameDataset(
            root_dir=root,
            num_frames=num_frames,
            transform=transform,
            sample_list=samples,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        p = run_inference_probs(model, loader, device, len(dataset))
        probs = p if probs is None else probs + p
    return torch.log(probs / len(transforms) + 1e-12)


def cached_log_probs(
    ckpt_path: Path,
    split: str,
    samples: List[Tuple[Path, int]],
    root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    tag = "tta" if args.tta else "single"
    cache_file = Path(args.cache_dir) / f"{ckpt_path.parent.name}_{ckpt_path.stem}_{split}_{tag}.pt"
    if cache_file.exists():
        return torch.load(cache_file)

    model, cfg = load_model(ckpt_path, device)
    log_probs = predict_log_probs(model, cfg, samples, root, device, args)
    del model
    torch.cuda.empty_cache()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(log_probs, cache_file)
    return log_probs


def combine(log_probs: List[torch.Tensor], weights: np.ndarray, method: str) -> torch.Tensor:
    weights = weights / weights.sum()
    if method == "mean":
        return sum(float(w) * lp.exp() for w, lp in zip(weights, log_probs))
    return sum(float(w) * lp for w, lp in zip(weights, log_probs))


def fit_weights(val_log_probs: List[torch.Tensor], val_labels: torch.Tensor) -> np.ndarray:
    """Minimise the val cross-entropy of the log-space ensemble.

    Weights are parameterised as softmax(alpha) so they stay on the simplex
    and Nelder-Mead can run unconstrained.
    """

    def to_weights(alpha: np.ndarray) -> np.ndarray:
        e = np.exp(alpha - alpha.max())
        return e / e.sum()

    def loss(alpha: np.ndarray) -> float:
        ens = combine(val_log_probs, to_weights(alpha), "logspace")
        return F.cross_entropy(ens, val_labels).item()

    alpha0 = np.zeros(len(val_log_probs))
    result = minimize(loss, alpha0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6})
    print(f"Val cross-entropy: uniform {loss(alpha0):.4f} -> fitted {result.fun:.4f}")
    return to_weights(result.x)


def accuracy(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return (scores.argmax(dim=1) == labels).float().mean().item()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", action="append", required=True, help="checkpoint as path[:weight], repeatable")
    p.add_argument("--method", choices=["mean", "logspace"], default="mean")
    p.add_argument("--fit-weights", action="store_true", help="fit weights on the val set (implies --method logspace)")
    p.add_argument("--tta", action="store_true", help="average over 3 spatial crops")
    p.add_argument("--val-dir", default="processed_data/val")
    p.add_argument("--test-dir", default="processed_data/test")
    p.add_argument("--output", default="submission_ensemble.csv")
    p.add_argument("--cache-dir", default="checkpoints/ensemble_cache")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fit_weights:
        args.method = "logspace"
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [parse_ckpt(spec) for spec in args.ckpt]
    paths = [path for path, _ in ckpts]
    weights = np.array([w for _, w in ckpts], dtype=np.float64)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing checkpoint(s): " + ", ".join(missing))

    if args.fit_weights:
        val_root = Path(args.val_dir).resolve()
        val_samples = collect_video_samples(val_root)
        val_labels = torch.tensor([label for _, label in val_samples], dtype=torch.long)
        val_log_probs = [cached_log_probs(p, "val", val_samples, val_root, device, args) for p in paths]
        for path, lp in zip(paths, val_log_probs):
            print(f"  {path}: val acc {accuracy(lp, val_labels):.4f}")
        weights = fit_weights(val_log_probs, val_labels)
        print(f"Ensemble val acc: {accuracy(combine(val_log_probs, weights, 'logspace'), val_labels):.4f}")

    print("Weights: " + ", ".join(f"{p} = {w / weights.sum():.3f}" for p, w in zip(paths, weights)))

    test_root = Path(args.test_dir).resolve()
    video_names, video_dirs = discover_all_test_videos(test_root)
    test_samples = [(d, 0) for d in video_dirs]
    print(f"Found {len(video_names)} test clips.")
    test_log_probs = [cached_log_probs(p, "test", test_samples, test_root, device, args) for p in paths]
    predictions = combine(test_log_probs, weights, args.method).argmax(dim=1).tolist()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "predicted_class"])
        writer.writerows(zip(video_names, predictions))
    print(f"Wrote {len(predictions)} rows to {output}")


if __name__ == "__main__":
    main()
