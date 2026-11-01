from __future__ import annotations

from typing import List, Dict

import torch.nn as nn
from transformers import VivitForVideoClassification


class ViViTSSV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 33,
        pretrained_model: str = "google/vivit-b-16x2-kinetics400",
    ) -> None:
        super().__init__()
        self.model = VivitForVideoClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        return self.model(pixel_values=x).logits

    def classifier_parameters(self) -> List[nn.Parameter]:
        return list(self.model.classifier.parameters())

    def layer_wise_lr_groups(
        self, lr_head: float, lr_backbone: float, lr_decay: float = 0.75
    ) -> List[Dict]:
        blocks = self.model.vivit.encoder.layer
        n = len(blocks)
        groups = [{"params": list(self.model.classifier.parameters()), "lr": lr_head}]
        groups.append({"params": list(self.model.vivit.layernorm.parameters()), "lr": lr_head * lr_decay})
        for i, block in enumerate(reversed(blocks)):
            lr = lr_backbone * (lr_decay ** (n - 1 - i))
            groups.append({"params": list(block.parameters()), "lr": lr})
        groups.append({"params": list(self.model.vivit.embeddings.parameters()), "lr": lr_backbone * (lr_decay ** n)})
        return groups
