"""Per-dataset loaders that yield canonical ``Record`` instances.

Three loaders, one per real dataset on disk:

- ``iter_cardd``        — CarDD (nasimetemadi/car-damage-detection): damage-type
                          labels + COCO bboxes (Variant A/B training).
- ``iter_comprehensive``— samwash94/comprehensive-car-damage-detection:
                          folder-encoded ``F_/R_`` × ``Normal/Crushed/Breakage``.
- ``iter_iaai``         — rebrowser/iaai-dataset: metadata-only (cost paywalled).

Each loader is a generator so callers can stream millions of records without
materializing them all in memory.
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
from typing import Iterator, Optional

from .schema import BBox, Record

# -------------------------------------------------------------------------
# Dataset roots — paths relative to the project root. Override per-call.
# -------------------------------------------------------------------------

RAW_ROOT = Path("data/raw")
CARDD_ROOT = RAW_ROOT / "car-damage-detection" / "CarDD_release" / "CarDD_COCO"
COMPREHENSIVE_ROOT = RAW_ROOT / "comprehensive-car-damage-detection" / "dataset"
IAAI_ROOT = RAW_ROOT / "iaai-dataset" / "auction-listings" / "data"

# -------------------------------------------------------------------------
# CarDD
# -------------------------------------------------------------------------

_CARDD_SPLITS = {
    "train": ("train2017", "instances_train2017.json"),
    "val":   ("val2017",   "instances_val2017.json"),
    "test":  ("test2017",  "instances_test2017.json"),
}


def _cardd_label_to_canonical(name: str) -> str:
    """Normalize CarDD's free-form category names to canonical ``DAMAGE_TYPES``."""
    return name.strip().lower().replace(" ", "_")


def iter_cardd(
    splits: tuple[str, ...] = ("train", "val", "test"),
    root: Path = CARDD_ROOT,
) -> Iterator[Record]:
    """Yield one ``Record`` per CarDD image with damage_types and normalized bboxes."""
    for split in splits:
        img_dir_name, ann_name = _CARDD_SPLITS[split]
        img_dir = root / img_dir_name
        ann_path = root / "annotations" / ann_name
        if not ann_path.exists():
            continue
        with ann_path.open() as f:
            coco = json.load(f)

        cat_by_id = {c["id"]: _cardd_label_to_canonical(c["name"]) for c in coco["categories"]}
        anns_by_image: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)

        for img in coco["images"]:
            w, h = img.get("width") or 0, img.get("height") or 0
            anns = anns_by_image.get(img["id"], [])
            bboxes: list[BBox] = []
            types: set[str] = set()
            for a in anns:
                cat = cat_by_id.get(a["category_id"])
                if cat is None:
                    continue
                types.add(cat)
                if w > 0 and h > 0 and a.get("bbox"):
                    x, y, bw, bh = a["bbox"]
                    bboxes.append(BBox(
                        damage_type=cat,
                        x_center=(x + bw / 2) / w,
                        y_center=(y + bh / 2) / h,
                        width=bw / w,
                        height=bh / h,
                    ))
            yield Record(
                image_path=img_dir / img["file_name"],
                dataset="cardd",
                damage_types=sorted(types),
                bboxes=bboxes,
                extras={"split": split, "image_id": img["id"]},
            )


# -------------------------------------------------------------------------
# Comprehensive (samwash94) — front/rear × normal/crushed/breakage folders
# -------------------------------------------------------------------------

_COMPREHENSIVE_FOLDERS = {
    "F_Normal":   ("front", "normal"),
    "F_Crushed":  ("front", "crushed"),
    "F_Breakage": ("front", "breakage"),
    "R_Normal":   ("rear",  "normal"),
    "R_Crushed":  ("rear",  "crushed"),
    "R_Breakage": ("rear",  "breakage"),
}


def iter_comprehensive(root: Path = COMPREHENSIVE_ROOT) -> Iterator[Record]:
    """Yield one ``Record`` per image with damage_location + damage_condition."""
    for folder, (loc, cond) in _COMPREHENSIVE_FOLDERS.items():
        d = root / folder
        if not d.exists():
            continue
        for img_path in sorted(d.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            yield Record(
                image_path=img_path,
                dataset="comprehensive",
                damage_location=loc,
                damage_condition=cond,
                extras={"folder": folder},
            )


# -------------------------------------------------------------------------
# Stanford Cars as "no damage" negatives for the multi-label classifier
#
# CarDD only contains damaged cars, so a model trained on it alone has no
# concept of "no damage" and falsely fires on undamaged inputs. Stanford
# Cars images are by-and-large undamaged, so we re-use them as the negative
# class for the damage classifier. Same Record schema, just with empty
# damage_types so encode_labels() emits an all-zero target vector.
# -------------------------------------------------------------------------


def iter_negatives(
    img_dir: Path = Path("data/raw/stanford-cars-dataset/cars_train/cars_train"),
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Iterator[Record]:
    """Yield ``Record(damage_types=[])`` for every image under ``img_dir``.

    This is intentionally agnostic to Stanford Cars' .mat metadata — we don't
    need bbox crops or class IDs for negatives, just the raw photo. So the
    loader is a simple recursive scan over the image directory, which makes
    it easy to swap in any other "undamaged car" image folder later.
    """
    if not img_dir.exists():
        return
    for path in sorted(img_dir.rglob("*")):
        if path.suffix.lower() not in extensions:
            continue
        yield Record(
            image_path=path,
            dataset="stanford_cars_negative",
            damage_types=[],
            bboxes=[],
        )


# -------------------------------------------------------------------------
# IAAI — metadata-only (cost is paywalled in the free sample)
# -------------------------------------------------------------------------

_IAAI_PREMIUM = "[PREMIUM]"


def _maybe(v) -> Optional[str]:
    if v is None or v == "" or v == _IAAI_PREMIUM:
        return None
    # pandas NaN equals nothing including itself
    if isinstance(v, float) and v != v:
        return None
    return v


def _maybe_int(v) -> Optional[int]:
    s = _maybe(v)
    if s is None:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _maybe_float(v) -> Optional[float]:
    s = _maybe(v)
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# Body-style normalization to canonical body_type names used by the reference
# table and segment classifier.
_IAAI_BODY_NORMALIZE = {
    "sedan": "sedan", "sedan 4 door": "sedan", "saloon": "sedan",
    "hatchback": "hatchback",
    "coupe": "coupe",
    "convertible": "convertible",
    "wagon": "wagon",
    "crew cab": "pickup", "extended cab": "pickup", "double cab": "pickup",
    "quad cab": "pickup", "regular cab": "pickup", "supercrew": "pickup",
    "standard cab": "pickup", "regular cab styleside": "pickup",
    "pickup": "pickup", "truck": "pickup",
    "sport utility": "suv", "suv": "suv",
}


def _normalize_body_style(raw: Optional[str]) -> str:
    if not raw:
        return "unknown"
    return _IAAI_BODY_NORMALIZE.get(raw.strip().lower(), "unknown")


def iter_iaai(
    root: Path = IAAI_ROOT,
    use_parquet: bool = True,
) -> Iterator[Record]:
    """Yield one Record per IAAI auction row.

    These records have NO images on local disk (imageUrl is paywalled) — they
    carry metadata only and feed the reference table and metadata-distribution
    builders.
    """
    if use_parquet:
        try:
            import pandas as pd  # type: ignore
            files = sorted(glob.glob(str(root / "*.parquet")))
            if files:
                yield from _iter_iaai_parquet(files)
                return
        except ImportError:
            pass
    yield from _iter_iaai_csv(sorted(glob.glob(str(root / "*.csv"))))


def _iter_iaai_parquet(files: list[str]) -> Iterator[Record]:
    import pandas as pd  # type: ignore
    for f in files:
        df = pd.read_parquet(f)
        # to_dict('records') preserves the leading-underscore column names
        # (e.g. `_primaryKey`) that itertuples mangles into NamedTuple-safe names.
        for row in df.to_dict("records"):
            yield _iaai_row_to_record(row)


def _iter_iaai_csv(files: list[str]) -> Iterator[Record]:
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                yield _iaai_row_to_record(row)


def _iaai_row_to_record(row: dict) -> Record:
    make = _maybe(row.get("make"))
    model = _maybe(row.get("model"))
    year = _maybe_int(row.get("year"))
    body_type = _normalize_body_style(_maybe(row.get("bodyStyle")))
    primary_damage = _maybe(row.get("primaryDamage")) or ""
    secondary_damage = _maybe(row.get("secondaryDamage")) or ""
    cost_raw = _maybe_float(row.get("estimatedRepairCost"))
    return Record(
        image_path=Path(f"iaai/{_maybe(row.get('_primaryKey')) or 'unknown'}.unknown"),
        dataset="iaai",
        # iaai gives us damage *location* phrases in text; reuse the location field
        damage_location=_classify_iaai_damage_location(primary_damage),
        # iaai has no per-image cost in the free sample — leave None
        cost=cost_raw,
        cost_currency="USD" if cost_raw is not None else None,
        cost_usd=cost_raw,
        cost_source="iaai" if cost_raw is not None else None,
        make=make.lower() if make else None,
        model=model.lower() if model else None,
        year=year,
        body_type=body_type,
        extras={
            "primaryDamage": primary_damage,
            "secondaryDamage": secondary_damage,
            "vehicleClass": _maybe(row.get("vehicleClass")),
            "exteriorColor": _maybe(row.get("exteriorColor")),
            "mileage": _maybe(row.get("mileage")),
            "lossType": _maybe(row.get("lossType")),
            "primaryKey": _maybe(row.get("_primaryKey")),
        },
    )


# Map free-text IAAI primaryDamage to {front, rear, unknown} for the reference
# table. We keep the original phrase in `extras['primaryDamage']` for audit.
_IAAI_FRONT_RE = re.compile(r"front", re.I)
_IAAI_REAR_RE = re.compile(r"rear", re.I)


def _classify_iaai_damage_location(phrase: str) -> str:
    if not phrase:
        return "unknown"
    has_front = bool(_IAAI_FRONT_RE.search(phrase))
    has_rear = bool(_IAAI_REAR_RE.search(phrase))
    if has_front and not has_rear:
        return "front"
    if has_rear and not has_front:
        return "rear"
    return "unknown"
