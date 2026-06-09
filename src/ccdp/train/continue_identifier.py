"""Continue-train the make/model identifier on a larger dataset (Phase 6).

Warm-starts from the existing identifier checkpoint, swaps the final head to the
new label space, and two-stage fine-tunes on a bigger dataset (CompCars by
default) at a lower LR. Reuses the exact epoch loop + MixUp/CutMix recipe from
:mod:`ccdp.train.train_car_identifier` so behaviour matches the original trainer.

What transfers vs. re-inits (see progress/phase_5-8_plan.md):
- **transfer:** full ResNet-50 backbone + the ``Linear(2048->512)`` embedding.
- **re-init:** only the final ``Linear(512->N)`` for the new class count.

An optional make-level *forgetting anchor* checks, after training, that the model
still recognises Stanford-Cars makes (a catastrophic-forgetting proxy).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ccdp.data import compcars
from ccdp.models.identifier import build_resnet50_identifier, n_trainable, set_finetune_stage
from ccdp.registry import create_run, load_checkpoint, save_checkpoint, update_metrics
from ccdp.train.train_car_identifier import TrainConfig, _run_epoch
from ccdp.utils import eval_transform, pick_device, seed_everything, train_transform


@dataclass
class ContinueConfig:
    base_checkpoint: Optional[str] = None     # defaults to production identifier
    epochs_stage1: int = 2
    epochs_stage2: int = 8
    batch_size: int = 64
    lr_stage1: float = 5e-4                    # lower than scratch — gentle continue
    lr_stage2: float = 5e-5
    weight_decay: float = 1e-4
    num_workers: int = 2
    image_size: int = 224
    val_fraction: float = 0.1
    seed: int = 42
    tag: str = "identifier_compcars"
    anchor_eval: bool = True                   # make-level forgetting check on Stanford
    resume_from: Optional[str] = None          # path to epoch_NNN.pt / last.pt to resume
    resume_run_dir: Optional[str] = None       # reuse existing run dir instead of creating new
    # --- Option 3: car-bbox crop preprocessing for VMMRdb -------------------
    # When ``use_bbox_crop=True`` and ``bbox_cache_path`` points to a JSON
    # produced by ``scripts/precompute_vmmrdb_bboxes.py``, the VMMRdb
    # DataLoader crops to the per-image car bbox before transforms — closing
    # the preprocessing gap with Stanford. We also switch to a Stanford-like
    # recipe (full LR, more epochs) because the bbox crop makes this closer
    # to fresh fine-tuning than a gentle continue.
    use_bbox_crop: bool = False
    bbox_cache_path: Optional[str] = None


def _swap_head(model: nn.Module, new_num_classes: int) -> bool:
    """Re-initialise the final ``Linear(512 -> N)`` only when the label space
    actually changes. Returns ``True`` if a swap happened, ``False`` if the
    existing head was preserved.

    When continuing from a checkpoint that already has the same N (e.g. v0.2.1
    → Option 3 retrain on the same 1163 VMMRdb classes), discarding the head
    throws away a useful warm start. This guard keeps the head intact in that
    case and only resets it when the label space genuinely differs (e.g. v0.1.0
    Stanford 196 → VMMRdb 1163).
    """
    final = model.fc[-1]
    in_features = final.in_features
    if final.out_features == new_num_classes:
        return False
    model.fc[-1] = nn.Linear(in_features, new_num_classes)
    return True


def _load_warm_start(base_ckpt: Path, new_num_classes: int, device) -> nn.Module:
    ck = load_checkpoint(base_ckpt, map_location=str(device))
    old_classes = int(ck.get("num_classes") or 196)
    model = build_resnet50_identifier(num_classes=old_classes, pretrained=False)
    model.load_state_dict(ck["model"])
    swapped = _swap_head(model, new_num_classes)
    print(f"[warm-start] head: old={old_classes} new={new_num_classes} "
          f"→ {'swapped (fresh head)' if swapped else 'preserved (same label space)'}")
    return model.to(device)


def make_level_anchor_accuracy(model, class_names, device, max_samples: int = 500) -> Optional[float]:
    """Top-1 *make* accuracy on Stanford-Cars val — a forgetting proxy.

    Returns None when Stanford Cars isn't available locally. The new head predicts
    CompCars models, so we compare only the *make* token of the predicted class
    name against Stanford's ground-truth make.
    """
    try:
        from ccdp.data import stanford_cars as sc
        classes = {c.class_id: c for c in sc.load_classes()}
        samples = sc.load_train_samples()
        _, val = sc.split_train_val(samples, val_fraction=0.1, seed=42)
    except Exception:  # noqa: BLE001
        return None
    if not val or not class_names:
        return None

    pred_make = [n.split()[0] if n else "" for n in class_names]
    tfm = eval_transform(224)
    model.eval()
    correct, total = 0, 0
    from PIL import Image
    with torch.no_grad():
        for s in val[:max_samples]:
            try:
                img = Image.open(s.image_path).convert("RGB").crop(s.bbox)
            except Exception:  # noqa: BLE001
                continue
            x = tfm(img).unsqueeze(0).to(device)
            idx = int(model(x).argmax(1).item())
            gt_make = classes[s.class_id].make
            if 0 <= idx < len(pred_make) and pred_make[idx] == gt_make:
                correct += 1
            total += 1
    return (correct / total) if total else None


def _build_loaders(cfg: ContinueConfig, dataset=compcars):
    classes = dataset.load_classes()
    # Pipe the bbox cache through *only* for datasets whose load_train_samples
    # accepts it. Today that's VMMRdb (where Option 3 lives); Stanford/CompCars
    # don't need it (Stanford has GT bboxes built in; CompCars unaffected).
    if cfg.use_bbox_crop and cfg.bbox_cache_path and hasattr(dataset, "load_train_samples"):
        try:
            samples = dataset.load_train_samples(bbox_cache_path=cfg.bbox_cache_path)
            n_with_bbox = sum(1 for s in samples if getattr(s, "bbox", None) is not None)
            print(f"[bbox-crop] loaded cache {cfg.bbox_cache_path}; "
                  f"{n_with_bbox}/{len(samples)} samples have a car bbox "
                  f"({n_with_bbox / max(len(samples), 1):.1%})")
        except TypeError:
            # Older datasets without the kwarg: fall back to no-crop loading.
            print("[bbox-crop] dataset doesn't accept bbox_cache_path — running without crops")
            samples = dataset.load_train_samples()
    else:
        samples = dataset.load_train_samples()

    train_samples, val_samples = dataset.split_train_val(
        samples, val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    train_tfm = train_transform(image_size=cfg.image_size)
    val_tfm = eval_transform(cfg.image_size)
    train_ds = dataset.build_torch_dataset(train_samples, train_tfm)
    val_ds = dataset.build_torch_dataset(val_samples, val_tfm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    return classes, train_loader, val_loader


def train(
    cfg: ContinueConfig,
    dataset=compcars,
    training_catalog_id: Optional[str] = None,
    smoke_batches: Optional[int] = None,
) -> Path:
    from ccdp.registry import production_target

    seed_everything(cfg.seed)
    device = pick_device()
    print(f"[device] {device}")

    classes, train_loader, val_loader = _build_loaders(cfg, dataset)
    num_classes = len(classes)
    class_names = [c.raw_name for c in classes]
    print(f"[data] {num_classes} classes, train≈{len(train_loader)}, val≈{len(val_loader)}")

    base_ckpt = Path(cfg.base_checkpoint) if cfg.base_checkpoint else production_target("identifier")
    if base_ckpt is None or not Path(base_ckpt).exists():
        raise FileNotFoundError(
            "No base identifier checkpoint. Pass --base-checkpoint or promote one."
        )
    model = _load_warm_start(Path(base_ckpt), num_classes, device)
    set_finetune_stage(model, 1)
    print(f"[warm-start] {base_ckpt} -> head swapped to {num_classes} classes; "
          f"stage 1 trainable {n_trainable(model):,}")

    # epoch-loop config (reuses train_car_identifier recipe: MixUp/CutMix/smoothing)
    loop_cfg = TrainConfig(image_size=cfg.image_size, seed=cfg.seed)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr_stage1, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # ---- resume / fresh run ------------------------------------------------
    start_epoch = 1
    best_val, stage = 0.0, 1
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume-from path not found: {resume_path}")
        ck = load_checkpoint(resume_path, map_location=str(device))
        # Restore weights over the freshly head-swapped model. Class count must match.
        ckpt_classes = int(ck.get("num_classes") or num_classes)
        if ckpt_classes != num_classes:
            raise ValueError(
                f"resume checkpoint has {ckpt_classes} classes but current dataset has "
                f"{num_classes}. Refusing to resume across different label spaces.",
            )
        model.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        stage = int(ck.get("stage", 1))
        best_val = float(ck.get("best_val", 0.0))
        # If we resume already inside stage 2, replicate the unfreeze + LR jump that
        # would have happened at the stage boundary.
        if stage == 2:
            set_finetune_stage(model, 2)
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        print(f"[resume] {resume_path} -> start at epoch {start_epoch} "
              f"(stage {stage}, best_val {best_val:.4f})")

    if cfg.resume_run_dir:
        run_dir = Path(cfg.resume_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[resume] reusing run dir {run_dir}")
    else:
        run_dir = create_run(
            variant="identifier", tag=cfg.tag, training_catalog_id=training_catalog_id,
            notes=f"Continue-train identifier on {dataset.__name__} ({num_classes} classes)",
        )
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()))

    total_epochs = cfg.epochs_stage1 + cfg.epochs_stage2
    for epoch in range(start_epoch, total_epochs + 1):
        if epoch == cfg.epochs_stage1 + 1 and stage == 1:
            stage = 2
            set_finetune_stage(model, 2)
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg.lr_stage2, weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
            print(f"[stage 2] unfreeze layer3/layer4, trainable {n_trainable(model):,}")

        print(f"\n[epoch {epoch}/{total_epochs}] stage={stage} lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, device,
                                            train=True, num_classes=num_classes,
                                            cfg=loop_cfg, max_batches=smoke_batches)
        val_loss, val_acc = _run_epoch(model, val_loader, None, device, train=False,
                                       num_classes=num_classes, cfg=loop_cfg,
                                       max_batches=smoke_batches)
        scheduler.step(val_loss)
        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc
        save_checkpoint(run_dir, {
            "model": model.state_dict(), "epoch": epoch, "stage": stage,
            "best_val": best_val, "num_classes": num_classes,
            "class_names": class_names, "config": asdict(cfg),
        }, epoch=epoch, is_best=is_best)
        update_metrics(run_dir.name.replace("run_", ""), {
            f"epoch_{epoch}": {"stage": stage, "train_acc": train_acc, "val_acc": val_acc},
            "best_val_acc": best_val,
        })

    if cfg.anchor_eval:
        anchor = make_level_anchor_accuracy(model, class_names, device)
        if anchor is not None:
            print(f"[anchor] Stanford make-level accuracy: {anchor:.3f}")
            update_metrics(run_dir.name.replace("run_", ""), {"anchor_make_acc": anchor})

    print(f"\n[done] best val acc: {best_val:.4f} -> {run_dir / 'best.pt'}")
    return run_dir / "best.pt"
