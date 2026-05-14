"""End-to-end Variant B inference: image -> YOLOv8 detector -> bbox stats +
parts (via `infer_part_from_damage`) -> XGBoost(B) cost.

Shares the classifier-based 2048-d feature extraction with Variant A so the
image-feature column schema in the XGBoost bundle stays consistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ccdp.costing import Calibrator, Catalog, load_active
from ccdp.costing import fx as fxmod
from ccdp.data.schema import DAMAGE_TYPES, BBox, infer_part_from_damage
from ccdp.identification.car_identifier import IdentificationResult
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.models.damage_classifier import build_damage_classifier, extract_features
from ccdp.models.xgb_regressor import XGBRegressorBundle, make_feature_matrix
from ccdp.registry import load_checkpoint, production_target
from ccdp.train.extract_bbox_features import bbox_stats

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DetectedBox:
    damage_type: str
    confidence: float
    xywh_norm: tuple[float, float, float, float]   # x_center, y_center, w, h (0..1)
    part: Optional[str] = None


@dataclass
class PredictionB:
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


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class VariantBPipeline:
    def __init__(
        self,
        detector_ckpt: Optional[Path] = None,
        classifier_ckpt: Optional[Path] = None,
        xgb_bundle_dir: Optional[Path] = None,
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO
        self.device = _pick_device()
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
        self.classifier = build_damage_classifier(num_classes=len(DAMAGE_TYPES),
                                                  pretrained=(self.classifier_ckpt is None))
        if self.classifier_ckpt and Path(self.classifier_ckpt).exists():
            ck = load_checkpoint(Path(self.classifier_ckpt), map_location=str(self.device))
            self.classifier.load_state_dict(ck["model"])
        self.classifier = self.classifier.to(self.device).eval()
        self.transform = _eval_transform(224)

        self.xgb_bundle_dir = xgb_bundle_dir
        if self.xgb_bundle_dir is None:
            prod = production_target("xgb_b")
            self.xgb_bundle_dir = prod.parent if prod else None
        self.xgb_model = None
        self.xgb_bundle: Optional[XGBRegressorBundle] = None
        if self.xgb_bundle_dir and (Path(self.xgb_bundle_dir) / "bundle.json").exists():
            self._load_xgb()

    def _load_xgb(self) -> None:
        import xgboost as xgb
        bundle_dir = Path(self.xgb_bundle_dir)
        booster_path = bundle_dir / "best.ubj"
        if not booster_path.exists():
            return
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(booster_path))
        with (bundle_dir / "bundle.json").open() as f:
            self.xgb_bundle = XGBRegressorBundle.from_dict(json.load(f))

    # -----------------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
        metadata: Optional[IdentificationResult] = None,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
    ) -> PredictionB:
        catalog = catalog or load_active()
        # --- YOLO detection ---
        result = self.detector.predict(str(image_path), imgsz=self.imgsz, conf=self.conf,
                                       verbose=False)[0]
        h, w = result.orig_shape
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
                xc = (x1 + x2) / 2 / max(w, 1)
                yc = (y1 + y2) / 2 / max(h, 1)
                bw = (x2 - x1) / max(w, 1)
                bh = (y2 - y1) / max(h, 1)
                dt = DAMAGE_TYPES[cls_idx]
                loc = (metadata.body_type if metadata else "unknown")
                part = infer_part_from_damage(dt, bbox_center=(xc, yc), damage_location=loc)
                detections.append(DetectedBox(
                    damage_type=dt, confidence=float(conf_t),
                    xywh_norm=(xc, yc, bw, bh), part=part,
                ))
                bboxes_for_stats.append(BBox(
                    damage_type=dt, x_center=xc, y_center=yc, width=bw, height=bh,
                ))
        stats = bbox_stats(bboxes_for_stats)
        damage_types = sorted({d.damage_type for d in detections})
        parts = sorted({d.part for d in detections if d.part})

        # --- image features from classifier backbone ---
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = extract_features(self.classifier, x).cpu().numpy().flatten()

        # --- cost: XGBoost(B) if loaded, else fallback estimator ---
        if self.xgb_model is not None and self.xgb_bundle is not None and metadata and metadata.make:
            cost_usd, tier, provenance = self._xgb_predict(feats, stats, metadata, catalog)
            warning = None
        else:
            est = fallback_estimate(
                parts_with_severity={p: "moderate" for p in parts},
                identification=metadata,
                catalog=catalog,
            )
            cost_usd, tier, provenance = est.cost_usd, est.tier, est.provenance
            warning = est.warning

        out_amount, fx_used = fxmod.convert(cost_usd, "USD", currency)
        fx_snap = {} if fx_used is None else {
            "rate": fx_used.rate, "base": fx_used.base, "target": fx_used.target,
            "source": fx_used.source, "fetched_at": fx_used.fetched_at,
        }

        return PredictionB(
            damage_types=damage_types,
            detections=detections,
            parts=parts,
            cost_usd=round(cost_usd, 2),
            currency=currency.upper(),
            cost=round(out_amount, 2),
            tier=tier,
            provenance=provenance,
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snap,
            bundle_run_id=(Path(self.xgb_bundle_dir).name if self.xgb_bundle_dir else None),
            detector_run_id=Path(self.detector_ckpt).parent.name if self.detector_ckpt else None,
            warning=warning,
        )

    def _xgb_predict(self, feats, stats, metadata, catalog):
        import pandas as pd
        bundle = self.xgb_bundle
        row = {f"f_{i}": float(v) for i, v in enumerate(feats)}
        row.update({
            "year": metadata.year or 2015,
            "make": metadata.make or "unknown",
            "body_type": metadata.body_type or "unknown",
            "segment": metadata.segment or "unknown",
        })
        # include all bbox-stat columns the bundle expects
        for k, v in stats.items():
            row[k] = float(v)
        df = pd.DataFrame([row])
        X = make_feature_matrix(df, bundle)
        import xgboost as xgb
        pred = float(self.xgb_model.predict(xgb.DMatrix(X.values, feature_names=list(X.columns)))[0])
        if bundle.training_median:
            cal = Calibrator(training_catalog_id=bundle.training_catalog_id or "unknown",
                             training_median=bundle.training_median)
            scaled = cal.scale(pred, catalog)
        else:
            scaled = pred
        tier = "exact" if (metadata.confidence or 0) >= 0.6 else "nearest_class"
        provenance = (f"xgb_b({tier}); training_catalog={bundle.training_catalog_id}; "
                      f"calibrated to {catalog.catalog_id}")
        return scaled, tier, provenance
