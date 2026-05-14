"""End-to-end Variant A inference: image -> damage types -> parts -> cost.

Loads the promoted classifier checkpoint + the promoted XGBoost(A) bundle.
Falls back gracefully to catalog-only Tier-3 if either is missing.
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
from ccdp.data.schema import DAMAGE_TYPES, infer_part_from_damage
from ccdp.identification.car_identifier import IdentificationResult
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.models.damage_classifier import build_damage_classifier, extract_features
from ccdp.models.xgb_regressor import XGBRegressorBundle, make_feature_matrix
from ccdp.registry import load_checkpoint, production_target

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PredictionA:
    damage_types: list[str]
    parts: list[str]
    cost_usd: float
    currency: str
    cost: float
    tier: str                                # 'exact' | 'nearest_class' | 'category_only' | 'xgb_a'
    provenance: str
    catalog_id: Optional[str] = None
    fx_snapshot: dict = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    bundle_run_id: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
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


class VariantAPipeline:
    """Stateful loader so the FastAPI/Gradio paths don't reload weights per call."""

    def __init__(
        self,
        classifier_ckpt: Optional[Path] = None,
        xgb_bundle_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or _pick_device()
        self.classifier_ckpt = classifier_ckpt or production_target("classifier")
        self.xgb_bundle_dir = xgb_bundle_dir
        if self.xgb_bundle_dir is None:
            prod = production_target("xgb_a")
            self.xgb_bundle_dir = prod.parent if prod else None

        self.transform = _eval_transform(224)
        self.model = build_damage_classifier(num_classes=len(DAMAGE_TYPES),
                                             pretrained=(self.classifier_ckpt is None))
        if self.classifier_ckpt and Path(self.classifier_ckpt).exists():
            ck = load_checkpoint(Path(self.classifier_ckpt), map_location=str(self.device))
            self.model.load_state_dict(ck["model"])
        self.model = self.model.to(self.device).eval()

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
        threshold: float = 0.5,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
    ) -> PredictionA:
        catalog = catalog or load_active()
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            feats = extract_features(self.model, x).cpu().numpy().flatten()

        damage_types = [DAMAGE_TYPES[i] for i, p in enumerate(probs) if p >= threshold]
        prob_dict = {DAMAGE_TYPES[i]: float(probs[i]) for i in range(len(DAMAGE_TYPES))}

        # parts derived from damage_types (no bbox info in Variant A)
        loc = (metadata.body_type if metadata else "unknown")
        parts: list[str] = []
        for dt in damage_types:
            p = infer_part_from_damage(dt, bbox_center=None, damage_location="unknown")
            if p:
                parts.append(p)

        # Tier-1/2 via XGBoost if available and we have metadata
        if self.xgb_model is not None and self.xgb_bundle is not None and metadata and metadata.make:
            cost_usd, tier, provenance = self._xgb_predict(feats, metadata, catalog)
            warning = None
        else:
            est = fallback_estimate(
                parts_with_severity={p: "moderate" for p in parts},
                identification=metadata,
                catalog=catalog,
            )
            cost_usd, tier, provenance = est.cost_usd, est.tier, est.provenance
            warning = est.warning

        # currency conversion
        out_amount, fx_used = fxmod.convert(cost_usd, "USD", currency)
        fx_snap = {} if fx_used is None else {
            "rate": fx_used.rate, "base": fx_used.base, "target": fx_used.target,
            "source": fx_used.source, "fetched_at": fx_used.fetched_at,
        }

        return PredictionA(
            damage_types=damage_types,
            parts=parts,
            cost_usd=round(cost_usd, 2),
            currency=currency.upper(),
            cost=round(out_amount, 2),
            tier=tier,
            provenance=provenance,
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snap,
            probabilities=prob_dict,
            bundle_run_id=(Path(self.xgb_bundle_dir).name if self.xgb_bundle_dir else None),
            warning=warning,
        )

    def _xgb_predict(
        self,
        feats: np.ndarray,
        metadata: IdentificationResult,
        catalog: Catalog,
    ) -> tuple[float, str, str]:
        import pandas as pd
        bundle = self.xgb_bundle
        row = {f"f_{i}": float(v) for i, v in enumerate(feats)}
        row.update({
            "year": metadata.year or 2015,
            "make": metadata.make or "unknown",
            "body_type": metadata.body_type or "unknown",
            "segment": metadata.segment or "unknown",
        })
        df = pd.DataFrame([row])
        X = make_feature_matrix(df, bundle)
        pred = float(self.xgb_model.predict(self._dmatrix(X))[0])

        # Catalog drift calibration
        if bundle.training_median:
            cal = Calibrator(training_catalog_id=bundle.training_catalog_id or "unknown",
                             training_median=bundle.training_median)
            scaled = cal.scale(pred, catalog)
        else:
            scaled = pred

        tier = "exact" if (metadata.confidence or 0) >= 0.6 else "nearest_class"
        provenance = (
            f"xgb_a({tier}); training_catalog={bundle.training_catalog_id}; "
            f"calibrated to {catalog.catalog_id}"
        )
        return scaled, tier, provenance

    def _dmatrix(self, X):
        import xgboost as xgb
        return xgb.DMatrix(X.values, feature_names=list(X.columns))
