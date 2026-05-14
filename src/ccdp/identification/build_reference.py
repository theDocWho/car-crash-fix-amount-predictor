"""Build the canonical reference table from the iaai metadata loader.

The free-sample iaai dataset has no usable cost values, so the resulting table
captures car-metadata *distributions* (year × make × model × body_type) with
NaN ``avg_cost_usd``. Tier-2 cost estimates fall through to catalog-based
pricing via the fallback estimator's no-scaling path.

When real cost data becomes available (e.g., the un-paywalled iaai slice from
Rebrowser's research-access program, or any authoritative repair table), rerun
this builder with `--with-cost` and the table will be re-aggregated with real
costs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ccdp.data.loaders import iter_iaai
from ccdp.identification import reference_table as reftab
from ccdp.identification.car_identifier import infer_segment


def build_from_iaai(
    out_path: Path = reftab.DEFAULT_PATH,
    limit: int | None = None,
) -> Path:
    """Stream the iaai loader into the reference-table builder."""
    rows = _iaai_rows(limit=limit)
    return reftab.build(rows, out_path=out_path)


def _iaai_rows(limit: int | None = None) -> Iterator[dict]:
    n = 0
    for r in iter_iaai():
        if not r.make:
            continue
        yield {
            "make": r.make,
            "model": r.model or "",
            "year": r.year,
            "body_type": r.body_type,
            "segment": infer_segment(r.make),
            "cost_usd": r.cost_usd,        # always None in free sample
            "dataset": r.dataset,
        }
        n += 1
        if limit and n >= limit:
            return
