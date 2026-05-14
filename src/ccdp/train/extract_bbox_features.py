"""Variant B bbox-derived features.

For each CarDD image (across train/val/test splits) we run the trained YOLOv8
detector and emit:

    image_id, split, n_damage_regions, total_area_pct, largest_area_pct,
    area_<damage_type> for every damage type,
    count_<damage_type>

Joined to the 2048-d image features (Variant A's parquet) at XGBoost(B)
training time.

For smoke / no-detector runs we fall back to **ground-truth CarDD bboxes** from
the loader so the rest of the Variant B pipeline can be exercised end-to-end
without waiting for a full YOLOv8 fine-tune.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_cardd
from ccdp.data.schema import DAMAGE_TYPES, BBox, Record
from ccdp.registry import production_target

DEFAULT_OUT = Path("data/processed/cardd_bbox_features.parquet")


def bbox_stats(bboxes: Iterable[BBox]) -> dict:
    """Aggregate a list of (already-normalized) BBoxes into the Variant B features."""
    bboxes = list(bboxes)
    out: dict[str, float] = {
        "n_damage_regions": float(len(bboxes)),
        "total_area_pct": 0.0,
        "largest_area_pct": 0.0,
    }
    for dt in DAMAGE_TYPES:
        out[f"count_{dt}"] = 0.0
        out[f"area_{dt}"] = 0.0
    if not bboxes:
        return out
    areas = []
    for b in bboxes:
        area = max(0.0, float(b.width)) * max(0.0, float(b.height))
        areas.append(area)
        out[f"count_{b.damage_type}"] = out.get(f"count_{b.damage_type}", 0.0) + 1.0
        out[f"area_{b.damage_type}"] = out.get(f"area_{b.damage_type}", 0.0) + area
    out["total_area_pct"] = float(sum(areas))
    out["largest_area_pct"] = float(max(areas)) if areas else 0.0
    return out


def _stats_row(r: Record, split: str, bboxes: Iterable[BBox]) -> dict:
    s = bbox_stats(bboxes)
    s["image_id"] = r.image_id
    s["image_path"] = str(r.image_path)
    s["split"] = split
    s["damage_types"] = ",".join(sorted(r.damage_types))
    return s


# -------------------------------------------------------------------------
# Ground-truth-bbox fallback (no detector required)
# -------------------------------------------------------------------------


def extract_from_ground_truth(
    out_path: Path = DEFAULT_OUT,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    """Use CarDD's ground-truth bboxes (already YOLO-normalized at load time).

    This is the "no detector" path: useful for smoke runs and as an upper-bound
    eval of how much bbox info helps the XGBoost regressor.
    """
    import pandas as pd

    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    split_map = {"train": train, "val": val, "test": test}
    rows: list[dict] = []
    t0 = time.time()
    for split_name, recs in split_map.items():
        if split_name not in splits:
            continue
        for r in recs:
            rows.append(_stats_row(r, split_name, r.bboxes))
        print(f"[{split_name}] {len(recs)} records ({time.time()-t0:.1f}s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] {len(rows)} rows -> {out_path}")
    return out_path


# -------------------------------------------------------------------------
# Real YOLOv8 detector path (used when a trained detector is available)
# -------------------------------------------------------------------------


def extract_with_detector(
    weights: Optional[Path] = None,
    out_path: Path = DEFAULT_OUT,
    imgsz: int = 640,
    conf: float = 0.25,
    max_records_per_split: Optional[int] = None,
) -> Path:
    """Run YOLOv8 inference over every CarDD image and aggregate bbox stats."""
    import pandas as pd
    from ultralytics import YOLO

    if weights is None:
        weights = production_target("detector")
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(
            f"No detector weights at {weights}. "
            f"Either train one (`ccdp train detector`) or use --gt for ground-truth fallback."
        )
    model = YOLO(str(weights))
    print(f"[detector] {weights}")

    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)

    rows: list[dict] = []
    for split_name, recs in (("train", train), ("val", val), ("test", test)):
        n = 0
        for r in recs:
            if max_records_per_split and n >= max_records_per_split:
                break
            result = model.predict(str(r.image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
            h, w = result.orig_shape
            bboxes = []
            if result.boxes is not None and len(result.boxes) > 0:
                for cls_t, xyxy_t in zip(result.boxes.cls.cpu().tolist(),
                                          result.boxes.xyxy.cpu().tolist()):
                    cls_idx = int(cls_t)
                    if cls_idx >= len(DAMAGE_TYPES):
                        continue
                    x1, y1, x2, y2 = xyxy_t
                    bboxes.append(BBox(
                        damage_type=DAMAGE_TYPES[cls_idx],
                        x_center=(x1 + x2) / 2 / max(w, 1),
                        y_center=(y1 + y2) / 2 / max(h, 1),
                        width=(x2 - x1) / max(w, 1),
                        height=(y2 - y1) / max(h, 1),
                    ))
            rows.append(_stats_row(r, split_name, bboxes))
            n += 1
        print(f"[{split_name}] {n} images processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] {len(rows)} rows -> {out_path}")
    return out_path
