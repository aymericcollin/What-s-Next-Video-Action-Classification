#!/usr/bin/env python3
"""
Run a trained checkpoint on the test split and write a submission CSV::

    video_name,predicted_class

Uses the same Hydra layout as ``train.py`` / ``evaluate.py``. Paths and checkpoint
come from the composed config (see ``configs/data/default.yaml`` and
``configs/train/default.yaml``).

Example (from ``src/``)::

    python create_submission.py
    python create_submission.py training.checkpoint_path=/path/to/best_model.pt
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset.video_dataset import VideoFrameDataset
from train import build_model
from utils import build_transforms, build_tta_transforms, set_seed


def load_manifest_video_names(manifest_path: Path) -> List[str]:
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "video_name" not in reader.fieldnames:
            raise ValueError(f"{manifest_path} must contain a 'video_name' column.")
        return [row["video_name"].strip() for row in reader]


def _index_video_folders(test_root: Path) -> Dict[str, Path]:
    test_root = test_root.resolve()
    index: Dict[str, Path] = {}
    for dirpath, dirs, _files in os.walk(test_root, topdown=True):
        base = Path(dirpath)
        for name in list(dirs):
            if not name.startswith("video_"):
                continue
            p = (base / name).resolve()
            if name in index:
                raise FileNotFoundError(
                    f"Duplicate video folder name {name!r}: {index[name]} and {p}"
                )
            index[name] = p
            dirs.remove(name)
    return index


def resolve_video_dirs(test_root: Path, video_names: List[str]) -> List[Path]:
    """Map each ``video_<id>`` folder name to a path using a pre-built index."""
    index = _index_video_folders(test_root)
    out: List[Path] = []
    missing: List[str] = []
    for name in video_names:
        p = index.get(name)
        if p is None:
            missing.append(name)
        else:
            out.append(p)
    if missing:
        sample = ", ".join(repr(m) for m in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise FileNotFoundError(
            f"{len(missing)} manifest video(s) not found under {test_root}: {sample}{extra}"
        )
    return out


def discover_all_test_videos(test_root: Path) -> Tuple[List[str], List[Path]]:
    index = _index_video_folders(test_root)
    video_names = sorted(index.keys())
    video_dirs = [index[name] for name in video_names]
    return video_names, video_dirs


def build_model_from_checkpoint(ckpt: Dict[str, Any]) -> torch.nn.Module:
    """Rebuild the model using the saved Hydra config when available."""
    if "config" in ckpt and ckpt["config"] is not None:
        cfg = OmegaConf.create(ckpt["config"])
        return build_model(cfg)

    cfg = OmegaConf.create(
        {
            "model": {
                "name": ckpt.get("model_name", "cnn_baseline"),
                "num_classes": int(ckpt["num_classes"]),
                "pretrained": bool(ckpt.get("pretrained", True)),
                "lstm_hidden_size": int(ckpt.get("lstm_hidden_size", 512)),
            }
        }
    )
    return build_model(cfg)


@torch.no_grad()
def run_inference_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
) -> torch.Tensor:
    """Run the model on the loader; return (N, num_classes) softmax probabilities."""
    model.eval()
    all_probs: List[torch.Tensor] = []
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)
    processed = 0
    for batch_idx, (video_batch, _labels) in enumerate(loader, start=1):
        video_batch = video_batch.to(device)
        logits = model(video_batch)
        all_probs.append(F.softmax(logits.float(), dim=1).cpu())
        processed += video_batch.size(0)
        if batch_idx % log_interval == 0 or batch_idx == n_batches:
            print(
                f"  Inference batch {batch_idx}/{n_batches} "
                f"({processed}/{total_videos} clips)",
                flush=True,
            )
    return torch.cat(all_probs, dim=0)  # (N, C)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    total_videos: int,
) -> List[int]:
    """Single-crop inference; returns predicted class indices."""
    return run_inference_probs(model, loader, device, total_videos).argmax(dim=1).tolist()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    ckpt: Dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
    model = build_model_from_checkpoint(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Model on device: {device}", flush=True)

    num_frames = int(ckpt.get("num_frames", cfg.dataset.num_frames))
    pretrained = bool(ckpt.get("pretrained", cfg.model.pretrained))
    image_size = int(cfg.dataset.get("image_size", 224))
    eval_transform = build_transforms(is_training=False, use_imagenet_norm=pretrained, image_size=image_size)

    test_root = Path(cfg.dataset.test_dir).resolve()
    output_path = Path(cfg.dataset.submission_output).resolve()
    manifest_cfg = cfg.dataset.get("test_manifest")

    print(f"Indexing video folders under: {test_root}", flush=True)
    if manifest_cfg:
        manifest_path = Path(str(manifest_cfg)).resolve()
        print(f"Reading manifest: {manifest_path}", flush=True)
        video_names = load_manifest_video_names(manifest_path)
        video_dirs = resolve_video_dirs(test_root, video_names)
        print(
            f"Resolved {len(video_dirs)} video folders from manifest for inference.",
            flush=True,
        )
    else:
        print(
            "No dataset.test_manifest provided; using all video_* folders found in test_dir.",
            flush=True,
        )
        video_names, video_dirs = discover_all_test_videos(test_root)
        print(
            f"Discovered {len(video_dirs)} video folders (sorted by video name).",
            flush=True,
        )
    sample_list: List[Tuple[Path, int]] = [(p, 0) for p in video_dirs]
    batch_size = int(cfg.training.batch_size)
    tta_crops = int(OmegaConf.select(cfg, "tta_crops", default=1))

    def make_loader(transform: Any) -> DataLoader:
        ds = VideoFrameDataset(
            root_dir=test_root,
            num_frames=num_frames,
            transform=transform,
            sample_list=sample_list,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(cfg.training.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    n_clips = len(sample_list)
    if tta_crops > 1:
        tta_tfms = build_tta_transforms(image_size, pretrained)[:tta_crops]
        print(f"TTA enabled: {len(tta_tfms)} crops, {n_clips} clips each", flush=True)
        ensemble_probs: torch.Tensor | None = None
        for i, tfm in enumerate(tta_tfms):
            print(f"  Crop {i+1}/{len(tta_tfms)}", flush=True)
            loader = make_loader(tfm)
            probs = run_inference_probs(model, loader, device, n_clips)
            ensemble_probs = probs if ensemble_probs is None else ensemble_probs + probs
        assert ensemble_probs is not None
        predictions: List[int] = ensemble_probs.argmax(dim=1).tolist()
    else:
        loader = make_loader(eval_transform)
        print(
            f"Starting inference: {n_clips} clips, batch_size={batch_size}, "
            f"{len(loader)} batches",
            flush=True,
        )
        predictions = run_inference(model, loader, device, total_videos=n_clips)

    print("Inference finished.", flush=True)

    if len(predictions) != len(video_names):
        raise RuntimeError(
            f"Prediction count {len(predictions)} != manifest length {len(video_names)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing submission CSV: {output_path}", flush=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "predicted_class"])
        for name, pred in zip(video_names, predictions):
            w.writerow([name, pred])

    print(f"Done. Wrote {len(predictions)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
