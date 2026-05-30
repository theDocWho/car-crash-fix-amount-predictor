"""Image → (is-it-a-car? → which car?) orchestration.

Chains the two new stages so the serving layer has a single call:

    gate (COCO Mask R-CNN)  →  no car? stop and say so.
                            →  car?    crop to it → ML identifier → make/model.

Returns an :class:`AutoIdentifyResult` carrying both the gate verdict (for the
"no car detected" UI path and the car-box overlay) and an
:class:`IdentificationResult` (the metadata the cost pipelines already accept).

Default gate/identifier instances are cached module-level so the heavy models
load once per process; both are injectable for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from ccdp.identification.car_gate import CarGate, GateResult
from ccdp.identification.car_identifier import IdentificationResult, infer_segment
from ccdp.identification.ml_identifier import MLIdentification, MLIdentifier

ImageLike = Union[str, Path, "object"]

_DEFAULT_GATE: Optional[CarGate] = None
_DEFAULT_IDENTIFIER: Optional[MLIdentifier] = None


def get_default_gate() -> CarGate:
    global _DEFAULT_GATE
    if _DEFAULT_GATE is None:
        _DEFAULT_GATE = CarGate()
    return _DEFAULT_GATE


def get_default_identifier() -> MLIdentifier:
    global _DEFAULT_IDENTIFIER
    if _DEFAULT_IDENTIFIER is None:
        _DEFAULT_IDENTIFIER = MLIdentifier()
    return _DEFAULT_IDENTIFIER


@dataclass
class AutoIdentifyResult:
    has_car: bool
    gate: GateResult
    identification: Optional[IdentificationResult] = None
    ml: Optional[MLIdentification] = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "has_car": self.has_car,
            "gate": self.gate.to_dict(),
            "identification": self.identification.to_dict() if self.identification else None,
            "ml": self.ml.to_dict() if self.ml else None,
            "note": self.note,
        }


def auto_identify(
    image: ImageLike,
    *,
    gate: Optional[CarGate] = None,
    identifier: Optional[MLIdentifier] = None,
    run_gate: bool = True,
    run_ml: bool = True,
    min_confidence: float = 0.0,
    image_path: Optional[Path] = None,
) -> AutoIdentifyResult:
    """Gate the image, then identify the car if one is present.

    - ``run_gate=False`` skips the presence check (assume a car) — useful when an
      upstream stage already cropped to a vehicle.
    - ``min_confidence`` drops a low-confidence ML guess back to make=None so the
      cost pipeline falls back to the catalog rather than trusting a weak label.
    """
    img_path = image_path or (Path(image) if isinstance(image, (str, Path)) else Path(""))

    # --- Stage 0: presence gate ------------------------------------------
    if run_gate:
        gate = gate or get_default_gate()
        g = gate.detect(image)
        if not g.has_car:
            return AutoIdentifyResult(
                has_car=False, gate=g, identification=None, ml=None,
                note="No car detected — upload a photo containing a car.",
            )
        crop = gate.crop_to_car(image, g)
    else:
        g = GateResult(has_car=True, note="gate skipped")
        crop = image

    if not run_ml:
        return AutoIdentifyResult(has_car=True, gate=g, identification=None, ml=None,
                                  note="car present; identification skipped")

    # --- Stage 1: ML make/model identification ---------------------------
    identifier = identifier or get_default_identifier()
    ml = identifier.predict(crop)

    confident = ml.confidence >= min_confidence and ml.make not in (None, "unknown")
    ident = IdentificationResult(
        image_path=img_path,
        make=(ml.make if confident else None),
        model=(ml.model if confident else None),
        year=(ml.year if confident else None),
        body_type=(ml.body_type if confident else "unknown"),
        segment=infer_segment(ml.make) if confident else "unknown",
        confidence=ml.confidence,
        source="ml" if confident else "none",
        stages_tried=["gate", "ml"],
    )
    note = (
        f"{ml.make} {ml.model} ({ml.year}) — {ml.confidence:.0%}"
        if confident else
        f"low-confidence guess ({ml.confidence:.0%}); using catalog fallback"
    )
    return AutoIdentifyResult(has_car=True, gate=g, identification=ident, ml=ml, note=note)


__all__ = ["auto_identify", "AutoIdentifyResult", "get_default_gate", "get_default_identifier"]
