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

import os
from pathlib import Path
from typing import Iterable

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_cardd
from ccdp.data.schema import DAMAGE_TYPES, Record

DEFAULT_ROOT = Path("data/processed/yolo")

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
