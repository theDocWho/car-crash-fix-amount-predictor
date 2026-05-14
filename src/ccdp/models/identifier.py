"""ResNet50-based make/model/year classifier for Stanford Cars 196.

Two-stage fine-tune pattern:
- Stage 1: freeze backbone, train only the classification head (fast warm-up).
- Stage 2: unfreeze ``layer3`` and ``layer4`` for full fine-tune at a lower LR.

Loading respects whatever stage the checkpoint was saved at — the trainer
restores optimizer state separately.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50_identifier(
    num_classes: int = 196,
    pretrained: bool = True,
) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return backbone


def set_finetune_stage(model: nn.Module, stage: int) -> None:
    """Stage 1 freezes backbone; stage 2 unfreezes ``layer3``/``layer4`` + head."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    for p in model.parameters():
        p.requires_grad = False

    if stage == 1:
        for p in model.fc.parameters():
            p.requires_grad = True
        return

    # stage == 2 — full fine-tune of upper backbone + head
    for name, p in model.named_parameters():
        if name.startswith("fc.") or name.startswith("layer3.") or name.startswith("layer4."):
            p.requires_grad = True


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable_parameters(model))
