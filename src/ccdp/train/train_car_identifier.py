"""Train the Stanford Cars make/model/year identifier.

Two-stage fine-tune (see ``ccdp.models.identifier.set_finetune_stage``):

- Stage 1: warm-up classification head with frozen backbone.
- Stage 2: unfreeze ``layer3``/``layer4`` for full fine-tune.

Checkpoints per epoch, plus last.pt / best.pt symlinks, with full resume
support (model, optimizer, scheduler, RNG, epoch counter).

CLI is wired in ``ccdp.cli`` under ``ccdp train identifier``.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from ccdp.data import stanford_cars as sc
from ccdp.models.identifier import (
    build_resnet50_identifier,
    n_trainable,
    set_finetune_stage,
)
from ccdp.registry import (
    create_run,
    load_checkpoint,
    save_checkpoint,
    update_metrics,
)
from ccdp.train.mixup import apply_mixup_or_cutmix, soft_cross_entropy

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    epochs_stage1: int = 3
    epochs_stage2: int = 12
    batch_size: int = 32
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    val_fraction: float = 0.1
    seed: int = 42
    tag: str = "identifier"
    # augmentation knobs
    randaug_num_ops: int = 2          # 0 disables RandAugment
    randaug_magnitude: int = 9
    mixup_alpha: float = 0.2          # 0.0 disables MixUp
    cutmix_alpha: float = 1.0         # 0.0 disables CutMix
    mix_prob: float = 0.8             # per-batch probability of applying mix
    cutmix_share: float = 0.5         # fraction of mixed batches using CutMix
    label_smoothing: float = 0.1


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _transforms(image_size: int, train: bool,
                randaug_num_ops: int = 2, randaug_magnitude: int = 9):
    if train:
        ops = [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if randaug_num_ops > 0:
            ops.append(transforms.RandAugment(
                num_ops=randaug_num_ops, magnitude=randaug_magnitude,
            ))
        else:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        return transforms.Compose(ops)
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _build_loaders(cfg: TrainConfig):
    classes = sc.load_classes()
    samples = sc.load_train_samples()
    train_samples, val_samples = sc.split_train_val(
        samples, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    train_tfm = _transforms(cfg.image_size, train=True,
                            randaug_num_ops=cfg.randaug_num_ops,
                            randaug_magnitude=cfg.randaug_magnitude)
    val_tfm = _transforms(cfg.image_size, train=False)
    train_ds = sc.build_torch_dataset(train_samples, train_tfm)
    val_ds = sc.build_torch_dataset(val_samples, val_tfm)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False, persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False, persistent_workers=cfg.num_workers > 0,
    )
    return classes, train_loader, val_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    train: bool,
    num_classes: int,
    cfg: Optional[TrainConfig] = None,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    model.train(train)
    val_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing if cfg else 0.0)
    total_loss = 0.0
    correct = 0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    t0 = time.time()
    with ctx:
        for i, (xb, yb) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if train and cfg is not None and cfg.mix_prob > 0:
                xb_mix, y_soft = apply_mixup_or_cutmix(
                    xb, yb, num_classes,
                    mixup_alpha=cfg.mixup_alpha,
                    cutmix_alpha=cfg.cutmix_alpha,
                    prob=cfg.mix_prob,
                    cutmix_share=cfg.cutmix_share,
                )
                logits = model(xb_mix)
                # apply label smoothing onto the soft target manually so the
                # smoothing is consistent regardless of mix branch
                eps = cfg.label_smoothing
                if eps > 0:
                    y_soft = (1.0 - eps) * y_soft + eps / num_classes
                loss = soft_cross_entropy(logits, y_soft)
            else:
                logits = model(xb)
                loss = val_criterion(logits, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            # accuracy still computed against the *original* hard labels
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
    elapsed = time.time() - t0
    print(f"      {'train' if train else 'val'}: loss={total_loss/max(n,1):.4f} "
          f"acc={correct/max(n,1):.4f} ({n} samples in {elapsed:.1f}s)")
    return total_loss / max(n, 1), correct / max(n, 1)


def train(
    cfg: TrainConfig,
    resume: Optional[Path] = None,
    smoke_batches: Optional[int] = None,
    training_catalog_id: Optional[str] = None,
) -> Path:
    """Run training and return the path of the best checkpoint."""
    _seed_everything(cfg.seed)
    device = _pick_device()
    print(f"[device] {device}")

    classes, train_loader, val_loader = _build_loaders(cfg)
    num_classes = len(classes)
    print(f"[data] {num_classes} classes, "
          f"train batches/epoch≈{len(train_loader)}, val batches≈{len(val_loader)}")

    model = build_resnet50_identifier(num_classes=num_classes, pretrained=True).to(device)
    set_finetune_stage(model, 1)
    print(f"[model] stage 1, trainable params: {n_trainable(model):,}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_stage1, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    start_epoch = 1
    best_val = 0.0
    stage = 1

    if resume is not None and resume.exists():
        ck = load_checkpoint(resume, map_location=str(device))
        model.load_state_dict(ck["model"])
        if ck.get("optimizer"):
            try:
                optimizer.load_state_dict(ck["optimizer"])
            except ValueError:
                print("[resume] optimizer state shape mismatch; reinitializing")
        if ck.get("scheduler"):
            try:
                scheduler.load_state_dict(ck["scheduler"])
            except Exception:  # noqa: BLE001
                pass
        start_epoch = ck.get("epoch", 0) + 1
        best_val = ck.get("best_val", 0.0)
        stage = ck.get("stage", 1)
        if stage == 2:
            set_finetune_stage(model, 2)
        try:
            rng = ck.get("rng_cpu")
            if rng is not None:
                torch.set_rng_state(rng.to(torch.uint8) if rng.dtype != torch.uint8 else rng)
        except (TypeError, RuntimeError):
            pass  # non-fatal; training still resumes
        run_dir = resume.parent
        print(f"[resume] {resume} (epoch={start_epoch}, stage={stage}, best_val={best_val:.4f})")
    else:
        run_dir = create_run(
            variant="identifier", tag=cfg.tag,
            training_catalog_id=training_catalog_id,
            notes="Stanford Cars 196 fine-tune (ResNet50, two-stage)",
        )
        with (run_dir / "config.yaml").open("w") as f:
            for k, v in asdict(cfg).items():
                f.write(f"{k}: {v}\n")
        print(f"[run] {run_dir}")

    total_epochs = cfg.epochs_stage1 + cfg.epochs_stage2

    for epoch in range(start_epoch, total_epochs + 1):
        # transition to stage 2 if needed
        if epoch == cfg.epochs_stage1 + 1 and stage == 1:
            stage = 2
            set_finetune_stage(model, 2)
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.lr_stage2, weight_decay=cfg.weight_decay,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            print(f"[stage 2] unfreezing layer3/layer4, trainable: {n_trainable(model):,}")

        print(f"\n[epoch {epoch}/{total_epochs}] stage={stage} lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss, train_acc = _run_epoch(
            model, train_loader, optimizer, device, train=True,
            num_classes=num_classes, cfg=cfg, max_batches=smoke_batches,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, None, device, train=False,
            num_classes=num_classes, cfg=cfg, max_batches=smoke_batches,
        )
        scheduler.step(val_loss)

        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "best_val": best_val,
            "config": asdict(cfg),
            "rng_cpu": torch.get_rng_state(),
            "num_classes": num_classes,
        }
        save_checkpoint(run_dir, state, epoch=epoch, is_best=is_best)
        update_metrics(run_dir.name.replace("run_", ""), {
            f"epoch_{epoch}": {
                "stage": stage,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss,     "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
            "best_val_acc": best_val,
        })

    print(f"\n[done] best val acc: {best_val:.4f}")
    return run_dir / "best.pt"
