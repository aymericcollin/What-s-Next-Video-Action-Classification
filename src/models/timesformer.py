from __future__ import annotations

import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification


class TimeSformerSSV2(nn.Module):
    """
    TimeSformer pretrained on SSv2 (174 classes), head replaced for our num_classes.
    Exposes classifier_parameters() / backbone_parameters() for phase-1/2 training.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_model: str = "facebook/timesformer-base-finetuned-ssv2",
    ) -> None:
        super().__init__()
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=video_batch).logits

    def classifier_parameters(self) -> list:
        return list(self.model.classifier.parameters())

    def backbone_parameter_groups(self, lr_head: float, lr_backbone: float) -> list:
        classifier_ids = {id(p) for p in self.model.classifier.parameters()}
        backbone_params = [
            p for p in self.parameters() if id(p) not in classifier_ids
        ]
        return [
            {"params": list(self.model.classifier.parameters()), "lr": lr_head},
            {"params": backbone_params, "lr": lr_backbone},
        ]
