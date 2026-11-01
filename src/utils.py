"""
Small helpers: reproducibility, image transforms, and metric computation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def set_seed(seed: int) -> None:
    """Make runs reproducible (as far as CUDA allows)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_imagenet_norm: bool = True,
    random_crop: bool = False,
    color_jitter: bool = False,
    rand_augment: bool = False,
    random_erasing: bool = False,
    use_rrc: bool = False,
) -> transforms.Compose:
    if use_imagenet_norm:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    if is_training:
        aug = []
        if use_rrc:
            aug.append(transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)))
        else:
            resize_to = int(image_size * 256 / 224) if random_crop else image_size
            aug.append(transforms.Resize((resize_to, resize_to)))
            if random_crop:
                aug.append(transforms.RandomCrop(image_size))
        if rand_augment:
            aug.append(transforms.RandAugment(num_ops=2, magnitude=9))
        if color_jitter:
            aug.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                )
            )
        aug += [transforms.ToTensor(), normalize]
        if random_erasing:
            aug.append(transforms.RandomErasing(p=0.25))
        return transforms.Compose(aug)

    resize_to = int(image_size * 256 / 224) if random_crop else image_size
    eval_aug = [transforms.Resize((resize_to, resize_to))]
    if random_crop:
        eval_aug.append(transforms.CenterCrop(image_size))
    eval_aug += [transforms.ToTensor(), normalize]
    return transforms.Compose(eval_aug)


def build_tta_transforms(image_size: int, use_imagenet_norm: bool) -> List[transforms.Compose]:
    """3 spatial crops (left, center, right) for test-time augmentation."""
    scale = int(image_size * 256 / 224)
    normalize = (
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if use_imagenet_norm
        else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    )
    offsets = [0, (scale - image_size) // 2, scale - image_size]
    return [
        transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.Lambda(lambda img, off=off: TF.crop(img, 0, off, image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
        for off in offsets
    ]


@torch.no_grad()
def accuracy_topk(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Tuple[torch.Tensor, ...]:
    max_k = max(topk)
    batch_size = targets.size(0)

    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))

    accuracies = []
    for k in topk:
        accuracies.append(correct[:k].reshape(-1).float().sum() / batch_size)
    return tuple(accuracies)


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction="none")
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


def split_train_val(
    samples: List[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if val_ratio <= 0.0:
        return shuffled, []

    n_val = int(round(len(shuffled) * val_ratio))
    n_val = max(1, n_val) if len(shuffled) > 1 else 0

    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    if len(train_samples) == 0:
        train_samples = val_samples[:-1]
        val_samples = val_samples[-1:]

    return train_samples, val_samples
