"""Multi-car mode: detect every vehicle, identify each, and group damage per car.

Reuses the pieces that already exist:
- ``CarGate.detect_all`` → all vehicles (box + mask), not just the dominant one.
- the per-part overlap logic from :mod:`ccdp.infer.parts_map`.
- the catalog estimator (:func:`ccdp.identification.fallback_estimator.estimate`).

Each damage instance is assigned to the vehicle whose mask (or box) it overlaps
most; per-vehicle damages are priced independently and summed. Damage that
overlaps no vehicle goes to an ``unassigned`` bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ccdp.costing import Catalog, fx as fxmod, load_active
from ccdp.identification.car_gate import CarGate, VehicleInstance, pad_box
from ccdp.identification.car_identifier import (
    DEFAULT_MIN_CONFIDENCE,
    IdentificationResult,
    infer_segment,
)
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.identification.ml_identifier import MLIdentifier
from ccdp.infer.parts_map import assign_damage_to_parts
from ccdp.infer.seg_inference import SegModel
from ccdp.registry import production_target

ImageLike = Union[str, Path, "object"]

# Distinct overlay colours, one per car (cycles if more cars than colours).
CAR_COLORS: list[tuple[int, int, int]] = [
    (0, 200, 0), (255, 99, 71), (30, 144, 255), (255, 215, 0),
    (186, 146, 255), (255, 140, 0), (60, 220, 200), (255, 105, 180),
]


def _bbox_overlap(dbox, vbox) -> float:
    """Fraction of the damage box area that falls inside the vehicle box."""
    ix1, iy1 = max(dbox[0], vbox[0]), max(dbox[1], vbox[1])
    ix2, iy2 = min(dbox[2], vbox[2]), min(dbox[3], vbox[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    darea = max(1e-6, (dbox[2] - dbox[0]) * (dbox[3] - dbox[1]))
    return inter / darea


def _overlap(damage, vehicle) -> float:
    """Damage↔vehicle overlap: mask-based when compatible, else bbox-based."""
    dm = getattr(damage, "mask", None)
    vm = getattr(vehicle, "mask", None)
    if dm is not None and vm is not None and getattr(dm, "shape", None) == getattr(vm, "shape", None):
        import numpy as np
        d = int(np.count_nonzero(dm))
        if d == 0:
            return 0.0
        return float(np.count_nonzero(np.logical_and(dm, vm))) / float(d)
    return _bbox_overlap(damage.bbox, vehicle.box)


def group_damages_by_vehicle(
    damages,
    vehicles,
    min_overlap: float = 0.1,
) -> tuple[list[list[int]], list[int]]:
    """Assign each damage to the best-overlapping vehicle.

    Returns ``(groups, unassigned)`` where ``groups[i]`` is the list of damage
    indices for ``vehicles[i]``, and ``unassigned`` is damage indices that didn't
    overlap any vehicle above ``min_overlap``.
    """
    groups: list[list[int]] = [[] for _ in vehicles]
    unassigned: list[int] = []
    for di, d in enumerate(damages):
        best_v, best_ov = -1, 0.0
        for vi, v in enumerate(vehicles):
            ov = _overlap(d, v)
            if ov > best_ov:
                best_ov, best_v = ov, vi
        if best_v >= 0 and best_ov >= min_overlap:
            groups[best_v].append(di)
        else:
            unassigned.append(di)
    return groups, unassigned


@dataclass
class CarResult:
    index: int
    label: str                              # car | truck | bus
    box: tuple[float, float, float, float]  # xyxy px
    color: tuple[int, int, int]
    make: Optional[str]
    model: Optional[str]
    confidence: float
    damage_types: list[str]
    parts: list[str]
    assignments: list[dict]
    cost_usd: float
    cost: float
    tier: str
    detections: list[dict] = field(default_factory=list)   # damage boxes (norm) for overlay

    def label_text(self) -> str:
        who = f"{self.make} {self.model}".strip() if self.make else f"{self.label} (unknown)"
        return f"Car {self.index + 1}: {who}"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["box"] = list(self.box)
        return d


@dataclass
class MultiCarPrediction:
    cars: list[CarResult]
    unassigned_damage: list[str]
    total_cost_usd: float
    total_cost: float
    currency: str
    catalog_id: Optional[str] = None
    fx_snapshot: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "n_cars": len(self.cars),
            "cars": [c.to_dict() for c in self.cars],
            "unassigned_damage": self.unassigned_damage,
            "total_cost_usd": self.total_cost_usd,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "catalog_id": self.catalog_id,
            "fx_snapshot": self.fx_snapshot,
            "note": self.note,
        }


class MultiCarPipeline:
    """Detect all vehicles → identify each → group damage per car → per-car cost."""

    def __init__(
        self,
        gate: Optional[CarGate] = None,
        identifier: Optional[MLIdentifier] = None,
        damage_model: Optional[SegModel] = None,
        parts_model: Optional[SegModel] = None,
        conf: float = 0.25,
        min_overlap: float = 0.1,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ):
        self.gate = gate or CarGate()
        self.identifier = identifier or MLIdentifier()
        if damage_model is None:
            dmg = production_target("yoloseg")
            if dmg is None or not Path(dmg).exists():
                raise FileNotFoundError(
                    "Multi-car mode needs the YOLOv8-seg damage weights "
                    "(`ccdp train detector --seg` → promote, or add yoloseg.pt to the release)."
                )
            damage_model = SegModel(dmg, conf=conf)
        if parts_model is None:
            prt = production_target("parts")
            if prt is None or not Path(prt).exists():
                raise FileNotFoundError(
                    "Multi-car mode needs the car-parts seg weights "
                    "(`ccdp train parts` → promote, or add parts.pt to the release)."
                )
            parts_model = SegModel(prt, conf=conf)
        self.damage = damage_model
        self.parts = parts_model
        self.min_overlap = min_overlap
        self.min_confidence = min_confidence

    def predict(
        self,
        image: ImageLike,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
    ) -> MultiCarPrediction:
        from PIL import Image

        catalog = catalog or load_active()
        pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        w, h = pil.size

        vehicles: list[VehicleInstance] = self.gate.detect_all(pil)
        damages = self.damage.predict(pil)
        parts = self.parts.predict(pil)
        groups, unassigned = group_damages_by_vehicle(damages, vehicles, self.min_overlap)

        cars: list[CarResult] = []
        total_usd = 0.0
        for idx, veh in enumerate(vehicles):
            crop = pil.crop(pad_box(veh.box, w, h, pad_frac=0.05))
            ml = self.identifier.predict(crop)
            confident = ml.confidence >= self.min_confidence and ml.make not in (None, "unknown")
            meta = IdentificationResult(
                image_path=Path(""),
                make=(ml.make if confident else None),
                model=(ml.model if confident else None),
                year=(ml.year if confident else None),
                body_type=(ml.body_type if confident else "unknown"),
                segment=infer_segment(ml.make) if confident else "unknown",
                confidence=ml.confidence, source="ml" if confident else "none",
            )
            car_damages = [damages[i] for i in groups[idx]]
            pws, assignments = assign_damage_to_parts(car_damages, parts, min_overlap=0.15)
            est = fallback_estimate(pws, meta, catalog)
            amount, _ = self._convert(est.cost_usd, currency)
            cars.append(CarResult(
                index=idx, label=veh.label, box=veh.box,
                color=CAR_COLORS[idx % len(CAR_COLORS)],
                make=(ml.make if confident else None),
                model=(ml.model if confident else None),
                confidence=round(ml.confidence, 3),
                damage_types=sorted({d.name for d in car_damages}),
                parts=sorted(pws.keys()), assignments=assignments,
                cost_usd=round(est.cost_usd, 2), cost=round(amount, 2), tier=est.tier,
                detections=[self._det(d, w, h) for d in car_damages],
            ))
            total_usd += est.cost_usd

        total_amount, fx_snapshot = self._convert(total_usd, currency)
        return MultiCarPrediction(
            cars=cars,
            unassigned_damage=sorted({damages[i].name for i in unassigned}),
            total_cost_usd=round(total_usd, 2),
            total_cost=round(total_amount, 2),
            currency=currency.upper(),
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snapshot,
            note=(f"{len(vehicles)} vehicle(s) detected" if vehicles else "no vehicle detected"),
        )

    @staticmethod
    def _det(d, w: int, h: int) -> dict:
        x1, y1, x2, y2 = d.bbox
        return {
            "damage_type": d.name, "confidence": round(d.score, 3),
            "xywh_norm": ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                          (x2 - x1) / w, (y2 - y1) / h),
        }

    @staticmethod
    def _convert(cost_usd: float, currency: str) -> tuple[float, dict]:
        amount, fr = fxmod.convert(cost_usd, "USD", currency)
        if fr is None:
            return amount, {}
        return amount, {"rate": fr.rate, "base": fr.base, "target": fr.target,
                        "source": fr.source, "fetched_at": fr.fetched_at}


__all__ = ["MultiCarPipeline", "MultiCarPrediction", "CarResult",
           "group_damages_by_vehicle", "CAR_COLORS"]
