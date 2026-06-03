"""Variant C (seg) mask-area features from the YOLOv8-seg damage model.

Mirrors :mod:`ccdp.train.extract_bbox_features` but each region's *area* is the
true segmentation-mask fraction rather than the bounding-box area. The output
parquet shares the same schema + join keys (``image_id``, ``split``), so
XGBoost(C) joins it exactly like Variant B joins the bbox features.

Two paths:
- ``extract_from_ground_truth`` — CarDD COCO polygon ``area`` (damaged-pixel area
  / image area). No model, no rasterisation.
- ``extract_with_seg_model`` — runs the trained YOLOv8-seg model and measures the
  realised mask-pixel fraction per instance.

Splits reuse the deterministic 80/10/10 ``dd.split_records`` assignment so the
rows line up with ``cardd_features.parquet`` and ``cardd_cost_targets.parquet``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Optional

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import CARDD_ROOT, _cardd_label_to_canonical, iter_cardd
from ccdp.data.schema import DAMAGE_TYPES, Record
from ccdp.registry import production_target

DEFAULT_OUT = Path("data/processed/cardd_seg_features.parquet")

_CANON = set(DAMAGE_TYPES)
_SPLIT_FILES = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
    "test": "instances_test2017.json",
}


def seg_mask_stats(instances: Iterable[tuple[str, float]]) -> dict:
    """Aggregate ``(damage_type, area_fraction)`` pairs — same keys as bbox_stats."""
    instances = list(instances)
    out: dict[str, float] = {
        "n_damage_regions": float(len(instances)),
        "total_area_pct": 0.0,
        "largest_area_pct": 0.0,
    }
    for dt in DAMAGE_TYPES:
        out[f"count_{dt}"] = 0.0
        out[f"area_{dt}"] = 0.0
    if not instances:
        return out
    areas = []
    for dt, area in instances:
        area = max(0.0, float(area))
        areas.append(area)
        out[f"count_{dt}"] = out.get(f"count_{dt}", 0.0) + 1.0
        out[f"area_{dt}"] = out.get(f"area_{dt}", 0.0) + area
    out["total_area_pct"] = float(sum(areas))
    out["largest_area_pct"] = float(max(areas)) if areas else 0.0
    return out


def _stats_row(r: Record, split: str, instances: Iterable[tuple[str, float]]) -> dict:
    s = seg_mask_stats(instances)
    s["image_id"] = r.image_id
    s["image_path"] = str(r.image_path)
    s["split"] = split
    s["damage_types"] = ",".join(sorted(r.damage_types))
    return s


def _gt_masks_by_filename(root: Path = CARDD_ROOT) -> dict[str, list[tuple[str, float]]]:
    """filename -> [(damage_type, polygon_area_fraction)] from CarDD COCO."""
    out: dict[str, list[tuple[str, float]]] = {}
    for ann_name in _SPLIT_FILES.values():
        ann_path = Path(root) / "annotations" / ann_name
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text())
        cat = {c["id"]: _cardd_label_to_canonical(c["name"]) for c in coco["categories"]}
        dims = {im["id"]: (im.get("width") or 0, im.get("height") or 0, im["file_name"])
                for im in coco["images"]}
        for a in coco["annotations"]:
            dt = cat.get(a["category_id"])
            if dt not in _CANON:
                continue
            w, h, fname = dims.get(a["image_id"], (0, 0, None))
            if not fname or w <= 0 or h <= 0:
                continue
            out.setdefault(fname, []).append((dt, float(a.get("area", 0.0)) / float(w * h)))
    return out


def extract_from_ground_truth(
    out_path: Path = DEFAULT_OUT,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Path:
    """True damaged-area fractions from CarDD polygons (no model)."""
    import pandas as pd

    masks_by_file = _gt_masks_by_filename()
    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    split_map = {"train": train, "val": val, "test": test}

    rows: list[dict] = []
    t0 = time.time()
    for split_name, recs in split_map.items():
        if split_name not in splits:
            continue
        for r in recs:
            rows.append(_stats_row(r, split_name, masks_by_file.get(r.image_path.name, [])))
        print(f"[{split_name}] {len(recs)} records ({time.time()-t0:.1f}s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] {len(rows)} rows -> {out_path}")
    return out_path


def extract_with_seg_model(
    weights: Optional[Path] = None,
    out_path: Path = DEFAULT_OUT,
    conf: float = 0.25,
    max_records_per_split: Optional[int] = None,
) -> Path:
    """Run the trained YOLOv8-seg model and measure realised mask-pixel fractions."""
    import numpy as np
    import pandas as pd

    from ccdp.infer.seg_inference import SegModel

    if weights is None:
        weights = production_target("yoloseg")
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(
            f"No yoloseg weights at {weights}. Train one (`ccdp train detector --seg`) "
            f"and promote it, or use --gt for the ground-truth fallback."
        )
    model = SegModel(Path(weights), conf=conf)

    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)

    rows: list[dict] = []
    for split_name, recs in (("train", train), ("val", val), ("test", test)):
        n = 0
        for r in recs:
            if max_records_per_split and n >= max_records_per_split:
                break
            if not r.image_path.exists():
                continue
            instances = [
                (inst.name, float(np.count_nonzero(inst.mask)) / float(inst.mask.size or 1))
                for inst in model.predict(str(r.image_path))
                if inst.name in _CANON
            ]
            rows.append(_stats_row(r, split_name, instances))
            n += 1
        print(f"[{split_name}] {n} images processed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] {len(rows)} rows -> {out_path}")
    return out_path


__all__ = ["seg_mask_stats", "extract_from_ground_truth", "extract_with_seg_model"]
