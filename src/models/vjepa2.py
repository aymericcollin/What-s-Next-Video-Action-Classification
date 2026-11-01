from __future__ import annotations

import torch
import torch.nn as nn
from transformers import VJEPA2ForVideoClassification


class VJEPA2SSV2(nn.Module):
    """
    V-JEPA 2 ViT-L fine-tuned on SSv2, head replaced for our num_classes.

    Uses VJEPA2ForVideoClassification which stacks:
      encoder (24 ViT-L blocks) → attentive pooler (3 layers) → linear head.

    Exposes:
      classifier_parameters()   — head + pooler only (for head-only phase)
      layer_wise_lr_groups()    — LLRD over the 24 encoder blocks
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "facebook/vjepa2-vitl-fpc16-256-ssv2",
    ) -> None:
        super().__init__()
        self.model = VJEPA2ForVideoClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values_videos=video_batch).logits

    def classifier_parameters(self) -> list:
        # Pooler query is also task-specific — retrain alongside the linear head.
        return list(self.model.classifier.parameters()) + list(self.model.pooler.parameters())

    def layer_wise_lr_groups(
        self, lr_head: float, lr_backbone: float, lr_decay: float = 0.75
    ) -> list:
        layers = self.model.vjepa2.encoder.layer
        num_layers = len(layers)
        assigned: set = set()
        groups: list = []

        head_params = list(self.model.classifier.parameters()) + list(self.model.pooler.parameters())
        groups.append({"params": head_params, "lr": lr_head})
        assigned.update(id(p) for p in head_params)

        for i, block in enumerate(reversed(layers)):
            params = [p for p in block.parameters() if id(p) not in assigned]
            if params:
                groups.append({"params": params, "lr": lr_backbone * (lr_decay ** i)})
                assigned.update(id(p) for p in params)

        remaining = [p for p in self.model.vjepa2.encoder.parameters() if id(p) not in assigned]
        if remaining:
            groups.append({"params": remaining, "lr": lr_backbone * (lr_decay ** num_layers)})

        return groups
