"""Convert CarDD COCO annotations to Ultralytics YOLO directory layout.

Target layout (per Ultralytics convention)::

    data/processed/yolo/
    ├── data.yaml
    ├── train/
    │   ├── images/        (symlinks to CarDD train2017/*.jpg)
    │   └── labels/        (one .txt per image, YOLO normalized boxes)
    ├── val/   (same)
    └── test/  (same)

Splits reuse the deterministic 80/10/10 image split from
``ccdp.data.damage_dataset.split_records`` so Variant A and Variant B train and
evaluate on the same images. Per-split labels are derived from the existing
CarDD loader (which already converts COCO bboxes to normalized YOLO format).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import (
    CARDD_ROOT,
    _CARDD_SPLITS,
    _cardd_label_to_canonical,
    iter_cardd,
)
from ccdp.data.schema import DAMAGE_TYPES, Record

DEFAULT_ROOT = Path("data/processed/yolo")
DEFAULT_SEG_ROOT = Path("data/processed/yolo_seg")

CLASS_INDEX: dict[str, int] = {dt: i for i, dt in enumerate(DAMAGE_TYPES)}


def _write_label_file(path: Path, record: Record) -> None:
    lines = []
    for b in record.bboxes:
        cls = CLASS_INDEX.get(b.damage_type)
        if cls is None:
            continue
        lines.append(f"{cls} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _link_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    # use relative symlinks so the dataset is portable inside the repo
    rel = os.path.relpath(src.resolve(), dst.parent.resolve())
    dst.symlink_to(rel)


def _materialize_split(records: Iterable[Record], split_root: Path) -> int:
    n = 0
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    for r in records:
        if not r.bboxes:
            continue
        if not r.image_path.exists():
            continue
        _link_image(r.image_path, img_dir / r.image_path.name)
        _write_label_file(lbl_dir / f"{r.image_path.stem}.txt", r)
        n += 1
    return n


def write_data_yaml(root: Path) -> Path:
    """Write the Ultralytics-style data.yaml referencing this layout."""
    p = root / "data.yaml"
    # Ultralytics resolves these relative to data.yaml's parent.
    body = []
    body.append(f"path: {root.resolve()}")
    body.append("train: train/images")
    body.append("val: val/images")
    body.append("test: test/images")
    body.append("")
    body.append(f"nc: {len(DAMAGE_TYPES)}")
    body.append("names:")
    for i, name in enumerate(DAMAGE_TYPES):
        body.append(f"  {i}: {name}")
    p.write_text("\n".join(body) + "\n")
    return p


def build(root: Path = DEFAULT_ROOT) -> Path:
    """Materialize the YOLO directory tree and return the data.yaml path."""
    records = [r for r in iter_cardd() if r.damage_types and r.bboxes]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    n_train = _materialize_split(train, root / "train")
    n_val = _materialize_split(val, root / "val")
    n_test = _materialize_split(test, root / "test")
    data_yaml = write_data_yaml(root)
    print(f"[yolo] root={root}  train={n_train}  val={n_val}  test={n_test}")
    print(f"[yolo] data.yaml -> {data_yaml}")
    return data_yaml


# -------------------------------------------------------------------------
# YOLOv8-seg (instance segmentation) — polygon labels from CarDD's COCO masks
# -------------------------------------------------------------------------


def normalize_polygon(poly: list[float], w: int, h: int) -> list[float]:
    """COCO absolute polygon [x1,y1,...] -> YOLO-seg normalized [x1/w,y1/h,...]."""
    out: list[float] = []
    for i in range(0, len(poly) - 1, 2):
        out.append(poly[i] / w)
        out.append(poly[i + 1] / h)
    return out


def seg_polygons_by_filename(root: Path = CARDD_ROOT) -> dict[str, list[tuple[int, list[float]]]]:
    """Map image file_name -> list of (class_idx, normalized polygon) from CarDD.

    RLE-encoded masks (dict ``segmentation``) are skipped — CarDD ships polygons.
    """
    out: dict[str, list[tuple[int, list[float]]]] = {}
    for _, ann_name in _CARDD_SPLITS.values():
        ann_path = Path(root) / "annotations" / ann_name
        if not ann_path.exists():
            continue
        coco = json.loads(ann_path.read_text())
        cat = {c["id"]: _cardd_label_to_canonical(c["name"]) for c in coco["categories"]}
        dims = {im["id"]: (im.get("width") or 0, im.get("height") or 0, im["file_name"])
                for im in coco["images"]}
        for a in coco["annotations"]:
            cls = CLASS_INDEX.get(cat.get(a["category_id"]))
            if cls is None:
                continue
            w, h, fname = dims.get(a["image_id"], (0, 0, None))
            seg = a.get("segmentation")
            if not fname or w <= 0 or h <= 0 or not isinstance(seg, list):
                continue
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    out.setdefault(fname, []).append((cls, normalize_polygon(poly, w, h)))
    return out


def _write_seg_label_file(path: Path, instances: list[tuple[int, list[float]]]) -> None:
    lines = [f"{cls} " + " ".join(f"{v:.6f}" for v in pts) for cls, pts in instances]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _materialize_seg_split(records, seg_map, split_root: Path) -> int:
    n = 0
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    for r in records:
        insts = seg_map.get(r.image_path.name)
        if not insts or not r.image_path.exists():
            continue
        _link_image(r.image_path, img_dir / r.image_path.name)
        _write_seg_label_file(lbl_dir / f"{r.image_path.stem}.txt", insts)
        n += 1
    return n


def build_seg(root: Path = DEFAULT_SEG_ROOT) -> Path:
    """Materialize CarDD as a YOLOv8-**seg** dataset (polygon labels) + data.yaml.

    Uses the same deterministic 80/10/10 image split as :func:`build` so the
    segmentation run is comparable to the Variant B detector.
    """
    records = [r for r in iter_cardd() if r.damage_types]
    train, val, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=42)
    seg_map = seg_polygons_by_filename()
    n_train = _materialize_seg_split(train, seg_map, root / "train")
    n_val = _materialize_seg_split(val, seg_map, root / "val")
    n_test = _materialize_seg_split(test, seg_map, root / "test")
    data_yaml = write_data_yaml(root)
    print(f"[yolo-seg] root={root}  train={n_train}  val={n_val}  test={n_test}")
    print(f"[yolo-seg] data.yaml -> {data_yaml}")
    return data_yaml
