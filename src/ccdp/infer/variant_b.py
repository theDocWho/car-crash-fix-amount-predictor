"""End-to-end Variant B inference.

Pipeline: image → YOLOv8 detector → bounding boxes + classes + areas → parts
(via :func:`infer_part_from_damage`) → 2048-d image features from the
classifier backbone → XGBoost(B) cost prediction (with calibrator + FX) →
:class:`PredictionB`.

Falls back to the catalog three-tier estimator if XGBoost or metadata is
missing, same as Variant A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ccdp.costing import Catalog, load_active
from ccdp.data.schema import DAMAGE_TYPES, BBox, infer_part_from_damage
from ccdp.identification.car_identifier import IdentificationResult
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.infer.base import BaseVariantPipeline
from ccdp.models.damage_classifier import build_damage_classifier, extract_features
from ccdp.registry import load_checkpoint, production_target
from ccdp.train.extract_bbox_features import bbox_stats
from ccdp.utils import eval_transform, pick_device


@dataclass
class DetectedBox:
    """One detection from YOLOv8 with the inferred canonical part (if any)."""

    damage_type: str
    confidence: float
    xywh_norm: tuple[float, float, float, float]
    part: Optional[str] = None


@dataclass
class PredictionB:
    """Structured response from :meth:`VariantBPipeline.predict`."""

    damage_types: list[str]
    detections: list[DetectedBox]
    parts: list[str]
    cost_usd: float
    currency: str
    cost: float
    tier: str
    provenance: str
    catalog_id: Optional[str] = None
    fx_snapshot: dict = field(default_factory=dict)
    bundle_run_id: Optional[str] = None
    detector_run_id: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["detections"] = [det.__dict__ for det in self.detections]
        return d


class VariantBPipeline(BaseVariantPipeline):
    """Detector-driven damage recognition + cost regression (with localization)."""

    xgb_variant_name = "xgb_b"

    def __init__(
        self,
        detector_ckpt: Optional[Path] = None,
        classifier_ckpt: Optional[Path] = None,
        xgb_bundle_dir: Optional[Path] = None,
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        super().__init__(xgb_bundle_dir=xgb_bundle_dir)
        self.device = pick_device()

        self.detector_ckpt = detector_ckpt or production_target("detector")
        if self.detector_ckpt is None or not Path(self.detector_ckpt).exists():
            raise FileNotFoundError(
                "No detector weights available. Train + promote one via "
                "`ccdp train detector` then `ccdp registry promote <run_id> detector`."
            )
        self.detector = YOLO(str(self.detector_ckpt))
        self.conf = conf
        self.imgsz = imgsz

        self.classifier_ckpt = classifier_ckpt or production_target("classifier")
        self.classifier = build_damage_classifier(
            num_classes=len(DAMAGE_TYPES),
            pretrained=(self.classifier_ckpt is None),
        )
        if self.classifier_ckpt and Path(self.classifier_ckpt).exists():
            ck = load_checkpoint(Path(self.classifier_ckpt), map_location=str(self.device))
            self.classifier.load_state_dict(ck["model"])
        self.classifier = self.classifier.to(self.device).eval()
        self.transform = eval_transform(224)

    # -- public API --------------------------------------------------------

    def predict(
        self,
        image,
        metadata: Optional[IdentificationResult] = None,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
        conf: Optional[float] = None,
    ) -> PredictionB:
        """Detect damages, score them, return calibrated cost + provenance.

        ``image`` accepts a path-like or an already-opened ``PIL.Image``.
        ``conf`` overrides the detector confidence threshold for this call only;
        leave as ``None`` to use the pipeline default (set in __init__).
        """
        catalog = catalog or load_active()

        detections, stats = self._detect(image, metadata, conf=conf)
        damage_types = sorted({d.damage_type for d in detections})
        parts = sorted({d.part for d in detections if d.part})

        image_features = self._image_features(image)

        if self._can_use_xgb(metadata):
            cost_usd, tier, provenance = self._predict_via_xgb(
                image_features, stats, metadata, catalog,
            )
            warning: Optional[str] = None
        else:
            cost_usd, tier, provenance, warning = self._predict_via_catalog(
                parts, metadata, catalog,
            )

        amount, fx_snapshot = self.convert_currency(cost_usd, currency)

        return PredictionB(
            damage_types=damage_types,
            detections=detections,
            parts=parts,
            cost_usd=round(cost_usd, 2),
            currency=currency.upper(),
            cost=round(amount, 2),
            tier=tier,
            provenance=provenance,
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snapshot,
            bundle_run_id=(Path(self.xgb_bundle_dir).name if self.xgb_bundle_dir else None),
            detector_run_id=(Path(self.detector_ckpt).parent.name if self.detector_ckpt else None),
            warning=warning,
        )

    # -- internals --------------------------------------------------------

    def _detect(
        self,
        image,
        metadata: Optional[IdentificationResult],
        conf: Optional[float] = None,
    ) -> tuple[list[DetectedBox], dict]:
        """Run YOLOv8 and convert raw boxes into ``DetectedBox`` + per-image stats.

        Ultralytics accepts paths, PIL.Image, numpy arrays, or tensors — we
        just pass through whatever the caller gave us.
        """
        source = str(image) if isinstance(image, (str, Path)) else image
        effective_conf = self.conf if conf is None else float(conf)
        result = self.detector.predict(
            source, imgsz=self.imgsz, conf=effective_conf, verbose=False,
        )[0]
        h, w = result.orig_shape
        location_hint = metadata.body_type if metadata else "unknown"

        detections: list[DetectedBox] = []
        bboxes_for_stats: list[BBox] = []
        if result.boxes is not None and len(result.boxes) > 0:
            for cls_t, conf_t, xyxy_t in zip(
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.xyxy.cpu().tolist(),
            ):
                cls_idx = int(cls_t)
                if cls_idx >= len(DAMAGE_TYPES):
                    continue
                x1, y1, x2, y2 = xyxy_t
                xc, yc = (x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1)
                bw, bh = (x2 - x1) / max(w, 1), (y2 - y1) / max(h, 1)
                damage_type = DAMAGE_TYPES[cls_idx]
                part = infer_part_from_damage(damage_type, (xc, yc), location_hint)
                detections.append(DetectedBox(
                    damage_type=damage_type,
                    confidence=float(conf_t),
                    xywh_norm=(xc, yc, bw, bh),
                    part=part,
                ))
                bboxes_for_stats.append(BBox(
                    damage_type=damage_type,
                    x_center=xc, y_center=yc, width=bw, height=bh,
                ))
        return detections, bbox_stats(bboxes_for_stats)

    def _image_features(self, image):
        """2048-d backbone features shared with Variant A's XGBoost feature schema."""
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return extract_features(self.classifier, x).cpu().numpy().flatten()

    def _can_use_xgb(self, metadata: Optional[IdentificationResult]) -> bool:
        """XGBoost is usable only when a bundle is loaded AND we have a make."""
        if self.xgb_model is None or self.xgb_bundle is None:
            return False
        return metadata is not None and metadata.make is not None

    def _predict_via_xgb(
        self,
        features,
        stats: dict,
        metadata: IdentificationResult,
        catalog: Catalog,
    ) -> tuple[float, str, str]:
        if self.xgb_model is None or self.xgb_bundle is None:
            raise RuntimeError("XGBoost not available; check `_can_use_xgb` upstream.")
        row = {f"f_{i}": float(v) for i, v in enumerate(features)}
        row.update({
            "year": metadata.year or 2015,
            "make": metadata.make or "unknown",
            "body_type": metadata.body_type or "unknown",
            "segment": metadata.segment or "unknown",
        })
        for k, v in stats.items():
            row[k] = float(v)
        return self.run_xgb(row, confidence=(metadata.confidence or 0.0), catalog=catalog)

    @staticmethod
    def _predict_via_catalog(
        parts: list[str],
        metadata: Optional[IdentificationResult],
        catalog: Catalog,
    ) -> tuple[float, str, str, Optional[str]]:
        est = fallback_estimate(
            parts_with_severity={p: "moderate" for p in parts},
            identification=metadata,
            catalog=catalog,
        )
        return est.cost_usd, est.tier, est.provenance, est.warning
