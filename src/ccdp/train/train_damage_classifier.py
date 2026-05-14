"""Train the CarDD ResNet50 multi-label damage-type classifier (Variant A).

Same two-stage pattern as the car identifier. Loss = BCEWithLogitsLoss with
per-class pos_weight from inverse frequency. Metrics: per-class precision /
recall / F1, macro F1, micro F1, mAP. Saved per epoch with full resume.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_cardd
from ccdp.data.schema import DAMAGE_TYPES
from ccdp.models.damage_classifier import (
    build_damage_classifier,
    n_trainable,
    set_finetune_stage,
)
from ccdp.registry import create_run, load_checkpoint, save_checkpoint, update_metrics
from ccdp.utils import eval_transform, pick_device, seed_everything, train_transform


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
    seed: int = 42
    tag: str = "classifier"
    label_smoothing: float = 0.0


def _train_tfm(cfg: "TrainConfig"):
    return train_transform(image_size=cfg.image_size, randaug_num_ops=0)


def _load_records() -> tuple[list, list, list, list[float]]:
    """Stream CarDD, split, and compute pos_weight from train fold."""
    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    pw = dd.pos_weight(train)
    return train, val, test, pw


def _build_loaders(cfg: TrainConfig):
    train, val, test, pw = _load_records()
    train_ds = dd.build_torch_dataset(train, _train_tfm(cfg))
    val_tfm = eval_transform(cfg.image_size)
    val_ds = dd.build_torch_dataset(val, val_tfm)
    test_ds = dd.build_torch_dataset(test, val_tfm)
    common = dict(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        pin_memory=False, persistent_workers=cfg.num_workers > 0,
    )
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
        pw,
    )


# ----- metrics ----------------------------------------------------------


def _per_class_prf(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    """Return dict with per-class P/R/F1 + macro/micro F1."""
    preds = (probs >= threshold).astype(np.float32)
    tp = (preds * labels).sum(axis=0)
    fp = (preds * (1 - labels)).sum(axis=0)
    fn = ((1 - preds) * labels).sum(axis=0)
    prec = np.where(tp + fp > 0, tp / (tp + fp + 1e-9), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn + 1e-9), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-9), 0.0)
    macro_f1 = float(f1.mean())
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_prec = micro_tp / max(micro_tp + micro_fp, 1)
    micro_rec = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = (2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)) if (micro_prec + micro_rec) > 0 else 0.0
    return {
        "per_class": {DAMAGE_TYPES[i]: {"precision": float(prec[i]),
                                         "recall":    float(rec[i]),
                                         "f1":        float(f1[i]),
                                         "support":   float(labels[:, i].sum())}
                      for i in range(len(DAMAGE_TYPES))},
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    max_batches: Optional[int] = None,
) -> tuple[float, dict]:
    model.train(train)
    total_loss = 0.0
    n = 0
    probs_all = []
    labels_all = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    t0 = time.time()
    with ctx:
        for i, (xb, yb) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            probs_all.append(torch.sigmoid(logits).detach().float().cpu().numpy())
            labels_all.append(yb.detach().float().cpu().numpy())
    elapsed = time.time() - t0
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(DAMAGE_TYPES)))
    labels = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, len(DAMAGE_TYPES)))
    metrics = _per_class_prf(probs, labels) if n > 0 else {"macro_f1": 0.0, "micro_f1": 0.0, "per_class": {}}
    metrics["loss"] = total_loss / max(n, 1)
    print(f"      {'train' if train else 'val'}: loss={metrics['loss']:.4f} "
          f"macroF1={metrics['macro_f1']:.4f} microF1={metrics['micro_f1']:.4f} "
          f"({n} samples in {elapsed:.1f}s)")
    return metrics["loss"], metrics


def train(
    cfg: TrainConfig,
    resume: Optional[Path] = None,
    smoke_batches: Optional[int] = None,
    training_catalog_id: Optional[str] = None,
) -> Path:
    seed_everything(cfg.seed)
    device = pick_device()
    print(f"[device] {device}")

    train_loader, val_loader, _test_loader, pos_weights = _build_loaders(cfg)
    print(f"[data] {len(DAMAGE_TYPES)} classes, "
          f"train batches/epoch≈{len(train_loader)}, val batches≈{len(val_loader)}")
    print(f"[data] pos_weight (inv-freq): "
          f"{ {DAMAGE_TYPES[i]: round(pos_weights[i], 2) for i in range(len(DAMAGE_TYPES))} }")

    model = build_damage_classifier(num_classes=len(DAMAGE_TYPES), pretrained=True).to(device)
    set_finetune_stage(model, 1)
    print(f"[model] stage 1, trainable params: {n_trainable(model):,}")

    pw_tensor = torch.tensor(pos_weights, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr_stage1, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    start_epoch = 1
    best_val_f1 = 0.0
    stage = 1
    run_dir: Path

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
        best_val_f1 = ck.get("best_val_f1", 0.0)
        stage = ck.get("stage", 1)
        if stage == 2:
            set_finetune_stage(model, 2)
        try:
            rng = ck.get("rng_cpu")
            if rng is not None:
                torch.set_rng_state(rng.to(torch.uint8) if rng.dtype != torch.uint8 else rng)
        except (TypeError, RuntimeError):
            pass
        run_dir = resume.parent
        print(f"[resume] {resume} (epoch={start_epoch}, stage={stage}, best_val_f1={best_val_f1:.4f})")
    else:
        run_dir = create_run(
            variant="classifier", tag=cfg.tag,
            training_catalog_id=training_catalog_id,
            notes="CarDD ResNet50 multi-label damage-type classifier (Variant A)",
        )
        with (run_dir / "config.yaml").open("w") as f:
            for k, v in asdict(cfg).items():
                f.write(f"{k}: {v}\n")
        print(f"[run] {run_dir}")

    total_epochs = cfg.epochs_stage1 + cfg.epochs_stage2

    for epoch in range(start_epoch, total_epochs + 1):
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
        train_loss, train_m = _run_epoch(
            model, train_loader, optimizer, criterion, device, train=True, max_batches=smoke_batches,
        )
        val_loss, val_m = _run_epoch(
            model, val_loader, None, criterion, device, train=False, max_batches=smoke_batches,
        )
        scheduler.step(val_loss)

        is_best = val_m["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_m["macro_f1"]

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "best_val_f1": best_val_f1,
            "config": asdict(cfg),
            "rng_cpu": torch.get_rng_state(),
            "num_classes": len(DAMAGE_TYPES),
            "damage_types": list(DAMAGE_TYPES),
            "pos_weights": pos_weights,
        }
        save_checkpoint(run_dir, state, epoch=epoch, is_best=is_best)
        update_metrics(run_dir.name.replace("run_", ""), {
            f"epoch_{epoch}": {
                "stage": stage,
                "train_loss": train_loss, "train_macro_f1": train_m["macro_f1"],
                "val_loss": val_loss,     "val_macro_f1": val_m["macro_f1"],
                "val_micro_f1": val_m["micro_f1"],
                "val_per_class": val_m["per_class"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            "best_val_f1": best_val_f1,
        })

    print(f"\n[done] best val macro-F1: {best_val_f1:.4f}")
    return run_dir / "best.pt"
