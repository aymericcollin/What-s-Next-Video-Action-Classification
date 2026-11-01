from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEForPreTraining


class VideoMAESSV2(nn.Module):
    """
    VideoMAE fine-tuned on SSv2 (174 classes), head replaced for our num_classes.
    Also supports loading from a locally pretrained VideoMAEForPreTraining checkpoint
    (closed world track: pass an absolute path as pretrained_model).
    Exposes layer_wise_lr_groups() for LLRD in Proposal 2.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "MCG-NJU/videomae-base-finetuned-ssv2",
    ) -> None:
        super().__init__()
        if Path(pretrained_model).exists():
            # Load encoder weights from locally pretrained VideoMAEForPreTraining checkpoint
            pretrain = VideoMAEForPreTraining.from_pretrained(pretrained_model)
            self.model = VideoMAEForVideoClassification(pretrain.config)
            self.model.classifier = nn.Linear(pretrain.config.hidden_size, num_classes)
            self.model.videomae.load_state_dict(pretrain.videomae.state_dict())
        else:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=video_batch).logits

    def classifier_parameters(self) -> list:
        return list(self.model.classifier.parameters())

    def layer_wise_lr_groups(
        self, lr_head: float, lr_backbone: float, lr_decay: float = 0.75
    ) -> list:
        layers = self.model.videomae.encoder.layer
        num_layers = len(layers)
        assigned: set = set()
        groups: list = []

        head_params = list(self.model.classifier.parameters())
        groups.append({"params": head_params, "lr": lr_head})
        assigned.update(id(p) for p in head_params)

        for i, block in enumerate(reversed(layers)):
            params = [p for p in block.parameters() if id(p) not in assigned]
            if params:
                groups.append({"params": params, "lr": lr_backbone * (lr_decay ** i)})
                assigned.update(id(p) for p in params)

        remaining = [p for p in self.model.videomae.parameters() if id(p) not in assigned]
        if remaining:
            groups.append({"params": remaining, "lr": lr_backbone * (lr_decay ** num_layers)})

        return groups
