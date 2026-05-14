"""ResNet50 multi-label damage-type classifier (Variant A).

Output layer is ``len(DAMAGE_TYPES) == 6`` logits — sigmoid activations applied
externally by the loss (`BCEWithLogitsLoss`) and at inference. Two-stage
fine-tune toggled via `set_finetune_stage`.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from ccdp.data.schema import DAMAGE_TYPES


def build_damage_classifier(
    num_classes: int = len(DAMAGE_TYPES),
    pretrained: bool = True,
) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = models.resnet50(weights=weights)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )
    return backbone


def set_finetune_stage(model: nn.Module, stage: int) -> None:
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    for p in model.parameters():
        p.requires_grad = False
    if stage == 1:
        for p in model.fc.parameters():
            p.requires_grad = True
        return
    for name, p in model.named_parameters():
        if name.startswith("fc.") or name.startswith("layer3.") or name.startswith("layer4."):
            p.requires_grad = True


def extract_features(model: nn.Module, x):
    """Forward through the backbone (everything up to but not including `fc`).

    Returns a (B, 2048) tensor — the 2048-d image embedding used by XGBoost(A).
    """
    import torch
    # ResNet forward, but skip fc
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
