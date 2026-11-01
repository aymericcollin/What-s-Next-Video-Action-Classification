from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetGRU(nn.Module):
    """
    EfficientNet-B0 per-frame feature extractor + Bidirectional GRU temporal model.
    ~7M parameters, designed for the closed-world (from-scratch) track.
    """

    def __init__(self, num_classes: int = 33, dropout: float = 0.3) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        feat_dim = backbone.classifier[1].in_features  # 1280
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(1024, num_classes)  # 512 * 2 directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.backbone(x.view(B * T, C, H, W))
        feats = feats.view(B, T, -1)
        _, hidden = self.gru(feats)
        out = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head(self.dropout(out))
