"""Extract 2048-d ResNet50 features for every CarDD image and cache to parquet.

Used by XGBoost(A) as the image-feature input. Runs the trained classifier
backbone in inference mode; writes one row per image with:

    image_id, split, damage_types (comma-joined), f_0..f_2047
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_cardd
from ccdp.data.schema import DAMAGE_TYPES
from ccdp.models.damage_classifier import build_damage_classifier, extract_features
from ccdp.registry import load_checkpoint, production_target
from ccdp.utils import eval_transform, pick_device


def extract_all(
    checkpoint: Optional[Path] = None,
    out_path: Path = Path("data/processed/cardd_features.parquet"),
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    max_batches: Optional[int] = None,
) -> Path:
    """Extract features for every CarDD image using the given checkpoint.

    If `checkpoint` is None, falls back to ``production_target('classifier')``;
    if that's also unset, uses ImageNet-pretrained ResNet50 (so the function is
    still usable for smoke tests before a real classifier is trained).
    """
    import pandas as pd

    device = pick_device()
    print(f"[device] {device}")

    if checkpoint is None:
        checkpoint = production_target("classifier")
    model = build_damage_classifier(num_classes=len(DAMAGE_TYPES), pretrained=(checkpoint is None))
    if checkpoint is not None and Path(checkpoint).exists():
        ck = load_checkpoint(Path(checkpoint), map_location=str(device))
        model.load_state_dict(ck["model"])
        print(f"[ckpt] loaded {checkpoint}")
    else:
        print("[ckpt] none — using ImageNet-pretrained backbone (smoke mode)")
    model = model.to(device).eval()

    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    splits = {"train": train, "val": val, "test": test}

    rows: list[dict] = []
    t0 = time.time()
    for split_name, recs in splits.items():
        ds = dd.build_torch_dataset(recs, eval_transform(image_size))
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        offset = 0
        for batch_i, (xb, _yb) in enumerate(loader):
            if max_batches is not None and batch_i >= max_batches:
                break
            xb = xb.to(device, non_blocking=True)
            with torch.no_grad():
                feats = extract_features(model, xb).cpu().numpy()
            for i in range(feats.shape[0]):
                if offset + i >= len(recs):
                    break
                r = recs[offset + i]
                row = {
                    "image_id": r.image_id,
                    "image_path": str(r.image_path),
                    "split": split_name,
                    "damage_types": ",".join(sorted(r.damage_types)),
                }
                for j in range(feats.shape[1]):
                    row[f"f_{j}"] = float(feats[i, j])
                rows.append(row)
            offset += feats.shape[0]
        print(f"[{split_name}] {len(rows)} features so far ({time.time() - t0:.1f}s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] wrote {len(rows)} rows -> {out_path}")
    return out_path
