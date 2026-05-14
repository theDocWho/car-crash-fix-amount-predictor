"""Batch-level MixUp and CutMix for the identifier.

Both produce *soft* labels (one-hot mixed by lambda), so the trainer must use a
loss that accepts target probabilities — `nn.CrossEntropyLoss` does as of
PyTorch 1.10. ``apply_mixup_or_cutmix`` alternates by coin flip per batch:
roughly half MixUp, half CutMix.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes=num_classes).to(torch.float32)


def mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convex-combine pairs of images and their one-hot labels."""
    if alpha <= 0.0 or x.size(0) < 2:
        return x, _one_hot(y, num_classes)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_oh = _one_hot(y, num_classes)
    y_mix = lam * y_oh + (1.0 - lam) * y_oh[perm]
    return x_mix, y_mix


def cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste a random rectangular patch from image B onto image A.

    Label is the area-weighted mix of the two one-hots.
    """
    if alpha <= 0.0 or x.size(0) < 2:
        return x, _one_hot(y, num_classes)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape

    # rectangle of area ratio (1-lam)
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_w = max(1, int(w * cut_ratio))
    cut_h = max(1, int(h * cut_ratio))
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)

    x_mix = x.clone()
    x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    # update lam to actual pasted area
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(w * h))

    y_oh = _one_hot(y, num_classes)
    y_mix = lam * y_oh + (1.0 - lam) * y_oh[perm]
    return x_mix, y_mix


def apply_mixup_or_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    prob: float = 1.0,
    cutmix_share: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp or CutMix to a batch with probability `prob`.

    With probability `cutmix_share`, CutMix is chosen; otherwise MixUp.
    When skipped (1 - prob), labels are returned as one-hots so the loss path
    is uniform regardless of branch.
    """
    if prob <= 0.0 or random.random() >= prob:
        return x, _one_hot(y, num_classes)
    if random.random() < cutmix_share:
        return cutmix(x, y, num_classes, alpha=cutmix_alpha)
    return mixup(x, y, num_classes, alpha=mixup_alpha)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against probability targets (B, C)."""
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
