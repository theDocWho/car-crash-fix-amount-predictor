"""Three-tier cost estimator that ties the identification, reference table,
and Tier-3 catalog together. This is the only function inference code should
call to turn a damage prediction into a cost number; everything else is wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ccdp.costing import Catalog, load_active
from ccdp.identification import reference_table as reftab
from ccdp.identification.car_identifier import IdentificationResult, infer_segment

HIGH_CONFIDENCE = 0.6  # car-id confidence threshold for Tier-1 path


@dataclass
class CostEstimate:
    cost_usd: float
    tier: str                      # "exact" | "nearest_class" | "category_only"
    provenance: str
    catalog_id: Optional[str] = None
    match_how: Optional[str] = None
    n_samples: int = 0
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def estimate(
    parts_with_severity: dict[str, str],
    identification: Optional[IdentificationResult] = None,
    catalog: Optional[Catalog] = None,
    high_confidence: float = HIGH_CONFIDENCE,
) -> CostEstimate:
    """Run the three-tier degradation chain to produce a USD cost estimate.

    `parts_with_severity` is the canonical-part -> severity map produced by
    the damage model. The result is always populated; ambiguity is encoded in
    `tier`, `match_how`, and `warning` for honest downstream reporting.

    Tier-1 (exact) and Tier-2 (nearest class) currently fall through to the
    catalog-based estimate scaled by the reference-table cost ratio. Once a
    trained XGBoost is wired in (Phase 2), this function is replaced by the
    XGBoost call for tiers 1 and 2; only Tier-3 stays catalog-only.
    """
    if catalog is None:
        catalog = load_active()
    segment_for_catalog = (
        identification.segment if identification and identification.segment != "unknown"
        else "mid"
    )
    catalog_estimate = catalog.estimate(parts_with_severity, segment=segment_for_catalog)

    # No identification at all → Tier 3
    if identification is None or not identification.make:
        return CostEstimate(
            cost_usd=catalog_estimate,
            tier="category_only",
            provenance="catalog-only fallback; no car identification",
            catalog_id=catalog.catalog_id,
            warning="Car model unidentified. Estimate based on damage severity × catalog only.",
        )

    # Try the reference table
    try:
        match = reftab.nearest(
            make=identification.make,
            model=identification.model,
            year=identification.year,
            body_type=identification.body_type if identification.body_type != "unknown" else None,
            segment=identification.segment if identification.segment != "unknown" else
                    infer_segment(identification.make),
        )
    except FileNotFoundError:
        match = None

    if match is None:
        return CostEstimate(
            cost_usd=catalog_estimate,
            tier="category_only",
            provenance=(
                f"catalog-only; no reference-table match for "
                f"{identification.make}/{identification.model}/{identification.year}"
            ),
            catalog_id=catalog.catalog_id,
            warning="No matching car class in reference table. Catalog-only estimate.",
        )

    tier = "exact" if (
        match["match_how"] == "exact" and identification.confidence >= high_confidence
    ) else "nearest_class"

    return CostEstimate(
        cost_usd=catalog_estimate * _scale_to_reference(match, catalog),
        tier=tier,
        provenance=(
            f"{tier} match via {match['match_how']}; "
            f"reference example: {match['example_model']}; "
            f"n_samples={match['n_samples']}"
        ),
        catalog_id=catalog.catalog_id,
        match_how=match["match_how"],
        n_samples=match["n_samples"],
        warning=None if tier == "exact" else (
            f"Specific car model unidentified or low-confidence; cost approximated "
            f"from nearest class ({match['body_type']}/{match['segment']})."
        ),
    )


def _scale_to_reference(match: dict, catalog: Catalog) -> float:
    """Scale a catalog Tier-3 estimate toward the reference-table average.

    Heuristic: if reference-table average is meaningfully higher/lower than the
    catalog median (controlled for segment), nudge the catalog estimate halfway
    toward it. Returns 1.0 (no scaling) when reference cost is unavailable —
    expected when the reference table was built from iaai's free sample which
    has no usable cost rows.
    """
    cat_median = catalog.median_cost()
    avg = match.get("avg_cost_usd")
    if cat_median <= 0 or avg is None:
        return 1.0
    # pandas/NaN safety
    if isinstance(avg, float) and avg != avg:
        return 1.0
    if avg <= 0:
        return 1.0
    ratio = avg / cat_median
    return (1.0 + ratio) / 2.0
