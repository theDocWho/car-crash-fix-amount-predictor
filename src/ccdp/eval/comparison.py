"""Variant A vs Variant B head-to-head evaluator.

The "comparison" is a single :class:`Comparison` dataclass that knows:

* which test split was used (seed=42 deterministic, identical for A and B)
* per-variant classification + regression metrics
* per-variant inference latency (ms / image)
* tier distribution (`exact` / `nearest_class` / `category_only`)
* slice analyses by car segment and damage type
* the production model + catalog ids the report was generated against

The class is pure data once built — the renderer (``ccdp.eval.report``)
consumes it without touching any models. That separation keeps the slow part
(model inference over 400 test images) decoupled from the fast part (HTML/PDF
rendering) so you can iterate on the report layout without re-evaluating.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ccdp.costing import load_active
from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_cardd
from ccdp.data.schema import DAMAGE_TYPES
from ccdp.eval.metrics import per_class_prf, regression_metrics
from ccdp.identification.car_identifier import IdentificationResult, infer_segment
from ccdp.registry import production_target


@dataclass
class VariantReport:
    """Everything the report renderer needs for one variant."""

    name: str                                   # 'A' | 'B'
    n_images: int
    classification: dict                        # output of per_class_prf
    regression: dict                            # output of regression_metrics
    tier_distribution: dict[str, int]           # tier -> count
    latency_ms: dict[str, float]                # mean, p50, p95
    examples: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


@dataclass
class Comparison:
    """Whole-report payload."""

    generated_at: str
    catalog_id: Optional[str]
    test_split_size: int
    seed: int
    variant_a: VariantReport
    variant_b: Optional[VariantReport] = None
    model_versions: dict[str, str] = field(default_factory=dict)
    slices: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def _percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))


def _make_metadata_sampler(seed: int = 42):
    """Reuse the same iaai sampler the synthetic targets used at training time.

    Crucial: must match the trainer's metadata sampling so the ground-truth cost
    we compare against is generated under identical assumptions.
    """
    from ccdp.train.synthesize_cost import MetadataSampler
    return MetadataSampler(seed=seed)


def _ground_truth_cost(record, sampler, catalog, rng):
    """Reconstruct the synthetic training-time cost target for one record."""
    from ccdp.train.synthesize_cost import cost_for_damage
    meta = sampler.sample()
    return meta, cost_for_damage(
        record.damage_types, meta.segment, catalog, rng, year=meta.year,
    )


def _identification_for(meta) -> IdentificationResult:
    """Build an `IdentificationResult` the pipeline expects from a metadata sample."""
    return IdentificationResult(
        image_path=Path(""), make=meta.make, model=meta.model, year=meta.year,
        body_type=meta.body_type, segment=infer_segment(meta.make),
        confidence=1.0, source="user",
    )


def evaluate_variant(
    pipeline,
    name: str,
    records,
    limit: Optional[int] = None,
) -> VariantReport:
    """Run a pipeline over the test split and accumulate everything we report on."""
    import random
    rng = random.Random(42)
    sampler = _make_metadata_sampler()
    catalog = load_active()

    n_classes = len(DAMAGE_TYPES)
    probs = []
    labels = []
    y_true_cost = []
    y_pred_cost = []
    tier_counts: dict[str, int] = {}
    latencies: list[float] = []
    examples: list[dict] = []
    failures: list[dict] = []

    for i, r in enumerate(records):
        if limit and i >= limit:
            break

        meta, gt_cost = _ground_truth_cost(r, sampler, catalog, rng)
        ident = _identification_for(meta)

        t0 = time.time()
        prediction = pipeline.predict(r.image_path, metadata=ident, currency="USD")
        latencies.append((time.time() - t0) * 1000)

        # classification — Variant A returns probabilities, Variant B doesn't
        probs_row = [0.0] * n_classes
        if hasattr(prediction, "probabilities") and prediction.probabilities:
            for j, dt in enumerate(DAMAGE_TYPES):
                probs_row[j] = float(prediction.probabilities.get(dt, 0.0))
        else:
            # For Variant B fall back to a 1.0 prob for any detected type
            for j, dt in enumerate(DAMAGE_TYPES):
                probs_row[j] = 1.0 if dt in prediction.damage_types else 0.0
        probs.append(probs_row)
        labels.append([1.0 if dt in r.damage_types else 0.0 for dt in DAMAGE_TYPES])

        # regression
        y_true_cost.append(gt_cost)
        y_pred_cost.append(prediction.cost_usd)

        # tier
        tier_counts[prediction.tier] = tier_counts.get(prediction.tier, 0) + 1

        # collect a handful of qualitative examples
        if len(examples) < 10:
            examples.append({
                "image_id": r.image_id,
                "image_path": str(r.image_path),
                "predicted_types": prediction.damage_types,
                "ground_truth_types": r.damage_types,
                "predicted_cost": prediction.cost_usd,
                "ground_truth_cost": gt_cost,
                "tier": prediction.tier,
            })

    # build failure list (top absolute cost errors)
    pairs = list(zip(y_true_cost, y_pred_cost, records[: len(y_true_cost)]))
    pairs.sort(key=lambda p: abs(p[1] - p[0]), reverse=True)
    for gt, pred, rec in pairs[:5]:
        failures.append({
            "image_id": rec.image_id,
            "image_path": str(rec.image_path),
            "predicted_cost": pred,
            "ground_truth_cost": gt,
            "abs_error": abs(pred - gt),
        })

    classification = per_class_prf(np.array(probs), np.array(labels), DAMAGE_TYPES)
    regression = regression_metrics(y_true_cost, y_pred_cost)
    latency = {
        "mean": float(np.mean(latencies)) if latencies else 0.0,
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
    }
    return VariantReport(
        name=name,
        n_images=len(probs),
        classification=classification,
        regression=regression,
        tier_distribution=tier_counts,
        latency_ms=latency,
        examples=examples,
        failures=failures,
    )


def _load_test_records(seed: int = 42, limit: Optional[int] = None):
    records = [r for r in iter_cardd() if r.damage_types]
    _, _, test = dd.split_records(records, fractions=(0.8, 0.1, 0.1), seed=seed)
    if limit:
        test = test[:limit]
    return test


def _resolve_run_id(variant: str) -> str:
    """Best-effort: read the production symlink to find which run id is live."""
    target = production_target(variant)
    if not target:
        return "unknown"
    try:
        return target.resolve().parent.name
    except OSError:
        return "unknown"


def build_comparison(
    variant_a_pipeline,
    variant_b_pipeline=None,
    limit: Optional[int] = None,
    seed: int = 42,
) -> Comparison:
    """Build the full :class:`Comparison` payload.

    Pass either both pipelines (full A vs B report) or only Variant A
    (used when the YOLOv8 detector hasn't been promoted yet).
    """
    records = _load_test_records(seed=seed, limit=limit)
    catalog = load_active()

    report_a = evaluate_variant(variant_a_pipeline, "A", records, limit=limit)
    report_b = None
    if variant_b_pipeline is not None:
        report_b = evaluate_variant(variant_b_pipeline, "B", records, limit=limit)

    slices = _slice_analyses(report_a, report_b)

    return Comparison(
        generated_at=datetime.now(timezone.utc).isoformat(),
        catalog_id=catalog.catalog_id,
        test_split_size=len(records),
        seed=seed,
        variant_a=report_a,
        variant_b=report_b,
        model_versions={
            "classifier": _resolve_run_id("classifier"),
            "detector": _resolve_run_id("detector"),
            "identifier": _resolve_run_id("identifier"),
            "xgb_a": _resolve_run_id("xgb_a"),
            "xgb_b": _resolve_run_id("xgb_b"),
        },
        slices=slices,
    )


# ---------------------------------------------------------------------------
# Slice analyses
# ---------------------------------------------------------------------------


def _slice_analyses(a: VariantReport, b: Optional[VariantReport]) -> dict:
    """A small table summarising RMSE/MAE by damage type."""
    out: dict[str, Any] = {}
    out["headline"] = {
        "A": {
            "macro_f1": a.classification["macro_f1"],
            "rmse": a.regression["rmse"],
            "r2": a.regression["r2"],
            "mape_pct": a.regression["mape_pct"],
        },
    }
    if b is not None:
        out["headline"]["B"] = {
            "macro_f1": b.classification["macro_f1"],
            "rmse": b.regression["rmse"],
            "r2": b.regression["r2"],
            "mape_pct": b.regression["mape_pct"],
        }
        out["delta"] = {
            "macro_f1": b.classification["macro_f1"] - a.classification["macro_f1"],
            "rmse": b.regression["rmse"] - a.regression["rmse"],
            "r2": b.regression["r2"] - a.regression["r2"],
            "mape_pct": b.regression["mape_pct"] - a.regression["mape_pct"],
        }
    return out
