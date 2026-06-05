"""Stage 0 of the pipeline: "is there a car, and where?".

A COCO-pretrained **Mask R-CNN (ResNet-50-FPN)** detector answers two questions
before any make/model or damage model runs:

1. Is a vehicle present at all? If not, we stop and tell the user — the
   downstream identifier and damage models presume a car and would otherwise
   hallucinate make/model or emit meaningless damage boxes.
2. Where is it? We return the highest-scoring vehicle box so callers can crop
   to it, which sharpens both the identifier and the damage detector.

The COCO mask head is computed but **ignored** — we only use boxes + labels +
scores here. This keeps the gate on the exact same architecture family
(`maskrcnn_resnet50_fpn`) as the CarDD damage detector: one model class, two
checkpoints (COCO vs CarDD).

The pure decision logic (:func:`decide`) is separated from the heavy model so it
can be unit-tested without downloading weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

# COCO (91-class indexing used by torchvision detection models) → our label.
# We treat car / bus / truck as "a vehicle is present". Motorcycle is excluded:
# this project costs car body damage, not two-wheelers.
VEHICLE_LABELS: dict[int, str] = {3: "car", 6: "bus", 8: "truck"}

ImageLike = Union[str, Path, "object"]  # path-like or PIL.Image.Image


@dataclass
class GateResult:
    """Outcome of the car-presence gate."""

    has_car: bool
    box: Optional[tuple[float, float, float, float]] = None  # xyxy, original px
    score: float = 0.0
    label: str = "none"
    n_vehicles: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "has_car": self.has_car,
            "box": list(self.box) if self.box else None,
            "score": self.score,
            "label": self.label,
            "n_vehicles": self.n_vehicles,
            "note": self.note,
        }


@dataclass
class VehicleInstance:
    """One detected vehicle (for multi-car grouping): label, score, box, mask."""

    label: str                                       # car | truck | bus
    score: float
    box: tuple[float, float, float, float]           # xyxy, original px
    mask: "object" = None                            # np.ndarray[bool] (H, W) or None

    @property
    def area_frac(self) -> float:
        import numpy as np
        if self.mask is None:
            return 0.0
        return float(np.count_nonzero(self.mask)) / float(self.mask.size or 1)


def decide(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    score_threshold: float = 0.5,
) -> GateResult:
    """Pure gate decision over raw detector outputs.

    Picks the highest-scoring *vehicle* detection above ``score_threshold``.
    Returns ``has_car=False`` when none qualifies. No torch/torchvision import
    here so it is trivially unit-testable.
    """
    best: Optional[tuple[float, tuple[float, float, float, float], str]] = None
    n_vehicles = 0
    for box, label, score in zip(boxes, labels, scores):
        label = int(label)
        score = float(score)
        if label not in VEHICLE_LABELS or score < score_threshold:
            continue
        n_vehicles += 1
        cand = (score, (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                VEHICLE_LABELS[label])
        if best is None or score > best[0]:
            best = cand

    if best is None:
        return GateResult(
            has_car=False, n_vehicles=0,
            note=f"no vehicle detected above score {score_threshold:.2f}",
        )
    return GateResult(
        has_car=True, box=best[1], score=best[0], label=best[2],
        n_vehicles=n_vehicles,
        note=f"{best[2]} ({best[0]:.0%}) — {n_vehicles} vehicle(s) found",
    )


def _box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _containment(a, b) -> float:
    """Fraction of box ``a`` that lies inside box ``b``."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = _box_area(a)
    return inter / aa if aa > 0 else 0.0


def nms_vehicles(vehicles: list, iou_thr: float = 0.6, contain_thr: float = 0.8) -> list:
    """Greedy NMS (highest score first) so one car isn't counted several times.

    Suppresses a candidate when it overlaps a kept box (IoU ≥ ``iou_thr``) **or**
    is mostly inside one (a sub-detection) **or** mostly contains one (a
    "whole-scene" super-box) — both common COCO failure modes on busy scenes.
    """
    kept: list = []
    for v in sorted(vehicles, key=lambda x: x.score, reverse=True):
        dup = False
        for k in kept:
            if (_box_iou(v.box, k.box) >= iou_thr
                    or _containment(v.box, k.box) >= contain_thr
                    or _containment(k.box, v.box) >= contain_thr):
                dup = True
                break
        if not dup:
            kept.append(v)
    return kept


def pad_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Expand a box by ``pad_frac`` of its size and clamp to image bounds."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad_frac
    x2 += bw * pad_frac
    y1 -= bh * pad_frac
    y2 += bh * pad_frac
    return (
        max(0, int(round(x1))), max(0, int(round(y1))),
        min(width, int(round(x2))), min(height, int(round(y2))),
    )


class CarGate:
    """COCO Mask R-CNN car-presence gate (boxes only; masks ignored).

    The torchvision model is lazy-loaded on first :meth:`detect` so importing
    this module (and the test suite) stays cheap. Inject ``model`` to bypass the
    download entirely in tests — any callable taking ``list[Tensor]`` and
    returning ``list[dict]`` with ``boxes``/``labels``/``scores`` works.
    """

    def __init__(
        self,
        model=None,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
    ):
        self._model = model
        self._device = device
        self.score_threshold = score_threshold

    # -- model loading -----------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_Weights,
            maskrcnn_resnet50_fpn,
        )
        from ccdp.utils import pick_device

        if self._device is None:
            self._device = str(pick_device())
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
        model.eval().to(self._device)
        self._model = model
        return model

    # -- public API --------------------------------------------------------

    def detect(self, image: ImageLike) -> GateResult:
        """Run the detector and reduce its output to a single gate decision."""
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        model = self._ensure_model()
        device = self._device or "cpu"
        x = to_tensor(pil).to(device)
        with torch.no_grad():
            out = model([x])[0]

        boxes = out["boxes"].cpu().tolist() if hasattr(out["boxes"], "cpu") else out["boxes"]
        labels = out["labels"].cpu().tolist() if hasattr(out["labels"], "cpu") else out["labels"]
        scores = out["scores"].cpu().tolist() if hasattr(out["scores"], "cpu") else out["scores"]
        return decide(boxes, labels, scores, score_threshold=self.score_threshold)

    def detect_all(self, image: ImageLike, score_threshold: Optional[float] = None,
                   min_area_frac: float = 0.01, nms_iou: float = 0.6) -> list:
        """Return EVERY distinct vehicle (car/truck/bus) with its mask, for
        multi-car grouping. Tiny boxes (< ``min_area_frac`` of the image) are
        dropped and near-duplicate boxes are merged via NMS so one car isn't
        counted several times. Each item is a :class:`VehicleInstance`."""
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        model = self._ensure_model()
        device = self._device or "cpu"
        x = to_tensor(pil).to(device)
        with torch.no_grad():
            out = model([x])[0]

        thr = self.score_threshold if score_threshold is None else score_threshold
        boxes = out["boxes"].cpu().tolist()
        labels = out["labels"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        masks = out["masks"].cpu().numpy() if "masks" in out else None  # (N,1,H,W) soft, orig size

        vehicles: list[VehicleInstance] = []
        for i, (box, lab, sc) in enumerate(zip(boxes, labels, scores)):
            lab = int(lab)
            if lab not in VEHICLE_LABELS or float(sc) < thr:
                continue
            mask = (masks[i, 0] >= 0.5) if masks is not None else None
            vehicles.append(VehicleInstance(
                label=VEHICLE_LABELS[lab], score=float(sc),
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                mask=mask,
            ))
        # drop tiny boxes, then merge near-duplicates of the same car
        img_area = float(pil.size[0] * pil.size[1]) or 1.0
        vehicles = [v for v in vehicles if _box_area(v.box) / img_area >= min_area_frac]
        vehicles = nms_vehicles(vehicles, nms_iou)
        vehicles.sort(key=lambda v: v.score, reverse=True)
        return vehicles

    def crop_to_car(self, image: ImageLike, result: GateResult, pad_frac: float = 0.05):
        """Return a PIL crop around the detected car, or the full image if none."""
        from PIL import Image

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        if not result.has_car or result.box is None:
            return pil
        w, h = pil.size
        return pil.crop(pad_box(result.box, w, h, pad_frac=pad_frac))


__all__ = ["CarGate", "GateResult", "VehicleInstance", "VEHICLE_LABELS", "decide", "pad_box"]
