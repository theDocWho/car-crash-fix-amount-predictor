"""Variant D — parts-aware cost: YOLOv8-seg damage + YOLOv8-seg parts → catalog.

Runs two segmentation models on the image, overlaps the damage masks with the
part masks to get **real** ``(part, damage, severity)`` assignments (instead of
the bbox-centre heuristic), and prices them through the existing three-tier
catalog estimator. No XGBoost, no retraining — interpretable and catalog-driven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ccdp.costing import Catalog, fx as fxmod, load_active
from ccdp.identification.car_identifier import IdentificationResult
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.infer.parts_map import assign_damage_to_parts
from ccdp.infer.seg_inference import SegModel
from ccdp.registry import production_target


@dataclass
class PredictionD:
    damage_types: list[str]
    parts: list[str]
    parts_with_severity: dict[str, str]
    assignments: list[dict]
    detections: list[dict]
    cost_usd: float
    currency: str
    cost: float
    tier: str
    provenance: str
    catalog_id: Optional[str] = None
    fx_snapshot: dict = field(default_factory=dict)
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class VariantDPipeline:
    """Damage-seg ∩ parts-seg → catalog cost (parts-aware)."""

    def __init__(
        self,
        damage_ckpt: Optional[Path] = None,
        parts_ckpt: Optional[Path] = None,
        conf: float = 0.25,
        min_overlap: float = 0.15,
        damage_model: Optional[SegModel] = None,
        parts_model: Optional[SegModel] = None,
    ):
        self.min_overlap = min_overlap

        # Resolve + validate weights up front so a missing-weights Space fails
        # HERE (caught by the demo's try/except → "Variant D unavailable") rather
        # than later inside predict() as an uncaught FileNotFoundError. SegModel
        # is lazy, so without this check construction would deceptively succeed.
        if damage_model is None:
            dmg = damage_ckpt or production_target("yoloseg")
            if dmg is None or not Path(dmg).exists():
                raise FileNotFoundError(
                    "Variant D needs the YOLOv8-seg damage weights. Train + promote "
                    "(`ccdp train detector --seg`) or add `yoloseg.pt` to the release."
                )
            damage_model = SegModel(dmg, conf=conf)
        if parts_model is None:
            prt = parts_ckpt or production_target("parts")
            if prt is None or not Path(prt).exists():
                raise FileNotFoundError(
                    "Variant D needs the car-parts seg weights. Train + promote "
                    "(`ccdp train parts`) or add `parts.pt` to the release."
                )
            parts_model = SegModel(prt, conf=conf)
        self.damage = damage_model
        self.parts = parts_model

    def predict(
        self,
        image,
        metadata: Optional[IdentificationResult] = None,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
        conf: Optional[float] = None,
    ) -> PredictionD:
        catalog = catalog or load_active()

        damages = self.damage.predict(image, conf=conf)
        parts = self.parts.predict(image, conf=conf)
        parts_with_severity, assignments = assign_damage_to_parts(
            damages, parts, min_overlap=self.min_overlap)

        est = fallback_estimate(parts_with_severity, metadata, catalog)
        amount, fx_snapshot = self._convert(est.cost_usd, currency)

        warning = est.warning
        if damages and not parts_with_severity:
            warning = ("damage found but none overlapped a recognised part — "
                       "cost is a category-only estimate")

        return PredictionD(
            damage_types=sorted({d.name for d in damages}),
            parts=sorted(parts_with_severity.keys()),
            parts_with_severity=parts_with_severity,
            assignments=assignments,
            detections=self._detections(damages),
            cost_usd=round(est.cost_usd, 2),
            currency=currency.upper(),
            cost=round(amount, 2),
            tier=est.tier,
            provenance=f"variant_d(parts-aware); {est.provenance}",
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snapshot,
            warning=warning,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _convert(cost_usd: float, currency: str) -> tuple[float, dict]:
        amount, fr = fxmod.convert(cost_usd, "USD", currency)
        if fr is None:
            return amount, {}
        return amount, {"rate": fr.rate, "base": fr.base, "target": fr.target,
                        "source": fr.source, "fetched_at": fr.fetched_at}

    @staticmethod
    def _detections(damages) -> list[dict]:
        """Damage boxes (normalised) for the annotated overlay."""
        out = []
        for d in damages:
            h, w = d.mask.shape
            x1, y1, x2, y2 = d.bbox
            out.append({
                "damage_type": d.name,
                "confidence": round(d.score, 3),
                "xywh_norm": ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                              (x2 - x1) / w, (y2 - y1) / h),
                "area_frac": round(d.area_frac, 4),
            })
        return out


__all__ = ["VariantDPipeline", "PredictionD"]
