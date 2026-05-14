"""Synthetic cost-target generator for Variant A XGBoost training.

Given the (image, damage_types) corpus from CarDD — which has no make/model
or cost — this module:

1. Samples (make, model, year, body_type, segment) for each image from the
   iaai metadata distribution (so tabular features have realistic correlations).
2. Maps damage_types → canonical parts via `infer_part_from_damage` (no bbox
   info available at the classifier level; uses location-neutral mapping).
3. Computes a catalog-based cost estimate for that (parts, segment).
4. Multiplies by a per-(make, year) noise factor and a small Gaussian to
   simulate real-world variation around the catalog baseline.

The point isn't to produce "real" costs — it's to give XGBoost a learnable
function from features to a number that respects the catalog as a baseline and
varies with car identity. The trained model + calibrator will then adjust as
the catalog evolves.

See PLAN.md §3 honesty statement for the disclosure footer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from ccdp.costing import Catalog, load_active
from ccdp.data.loaders import iter_iaai
from ccdp.data.schema import infer_part_from_damage

# severity multipliers when only damage type is known
_TYPE_SEVERITY_DEFAULT: dict[str, str] = {
    "scratch": "minor",
    "dent": "moderate",
    "crack": "moderate",
    "glass_shatter": "severe",
    "lamp_broken": "moderate",
    "tire_flat": "severe",
}


@dataclass
class MetadataSample:
    make: str
    model: str
    year: int
    body_type: str
    segment: str


class MetadataSampler:
    """Samples plausible (make, model, year, body_type, segment) from iaai."""

    def __init__(self, seed: int = 42, limit: Optional[int] = 5000):
        rng = random.Random(seed)
        pool: list[MetadataSample] = []
        for i, r in enumerate(iter_iaai()):
            if limit and i >= limit:
                break
            if not r.make or not r.year:
                continue
            pool.append(MetadataSample(
                make=r.make, model=r.model or "unknown",
                year=int(r.year), body_type=r.body_type,
                segment=_segment_for(r.make),
            ))
        self._pool = pool
        self._rng = rng

    def __len__(self) -> int:
        return len(self._pool)

    def sample(self) -> MetadataSample:
        return self._rng.choice(self._pool)


_LUXURY = {"audi", "bmw", "mercedes-benz", "porsche", "jaguar", "lexus",
           "infiniti", "acura", "land rover", "tesla", "cadillac", "lincoln",
           "genesis", "maserati", "bentley", "ferrari", "lamborghini"}
_ECONOMY = {"kia", "hyundai", "mitsubishi", "suzuki", "fiat", "nissan", "tata",
            "scion", "smart", "renault", "skoda"}


def _segment_for(make: str) -> str:
    m = make.lower()
    if m in _LUXURY:
        return "luxury"
    if m in _ECONOMY:
        return "economy"
    return "mid"


def cost_for_damage(
    damage_types: list[str],
    segment: str,
    catalog: Catalog,
    rng: random.Random,
    year: Optional[int] = None,
) -> float:
    """Map damage_types -> parts and compute a noisy catalog-based cost."""
    parts_with_severity: dict[str, str] = {}
    for dt in damage_types:
        part = infer_part_from_damage(dt, bbox_center=None, damage_location="unknown")
        if part is None:
            # type couldn't be position-mapped; pick a sensible default
            part = "front_bumper" if dt in {"dent", "scratch", "crack"} else None
        if part is None:
            continue
        severity = _TYPE_SEVERITY_DEFAULT.get(dt, "moderate")
        # if both severe and moderate map to same part, keep the more severe one
        existing = parts_with_severity.get(part)
        if existing is None or _severity_rank(severity) > _severity_rank(existing):
            parts_with_severity[part] = severity

    base = catalog.estimate(parts_with_severity, segment=segment)
    # age factor: older cars cheaper labor, sometimes pricier parts; modest 0.9–1.1
    age_factor = 1.0
    if year:
        age_years = max(0, 2026 - year)
        age_factor = max(0.85, 1.0 - 0.005 * min(age_years, 30))
    noise = rng.gauss(1.0, 0.10)  # ±10% per-instance variation
    cost = max(50.0, base * age_factor * noise)
    return round(cost, 2)


def _severity_rank(s: str) -> int:
    return {"minor": 0, "moderate": 1, "severe": 2}.get(s, 0)


def generate_targets(
    features_parquet: Path,
    out_path: Path = Path("data/processed/cardd_cost_targets.parquet"),
    seed: int = 42,
    catalog: Optional[Catalog] = None,
    sampler: Optional[MetadataSampler] = None,
) -> Path:
    """Read the feature parquet, attach sampled metadata + synthetic cost target."""
    import pandas as pd

    if catalog is None:
        catalog = load_active()
    if sampler is None:
        sampler = MetadataSampler(seed=seed)
    rng = random.Random(seed)

    feats = pd.read_parquet(features_parquet)
    rows = []
    for _, row in feats.iterrows():
        meta = sampler.sample()
        types = [t for t in row["damage_types"].split(",") if t]
        cost = cost_for_damage(types, meta.segment, catalog, rng, year=meta.year)
        rows.append({
            "image_id": row["image_id"],
            "split": row["split"],
            "damage_types": row["damage_types"],
            "make": meta.make,
            "model": meta.model,
            "year": meta.year,
            "body_type": meta.body_type,
            "segment": meta.segment,
            "cost_usd": cost,
            "cost_source": f"synthetic@{catalog.catalog_id}",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print(f"[done] wrote {len(rows)} targets -> {out_path}")
    return out_path
