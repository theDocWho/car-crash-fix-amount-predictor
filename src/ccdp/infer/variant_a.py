"""End-to-end Variant A inference.

Pipeline: image → ResNet50 multi-label classifier → damage types → optional
XGBoost(A) cost prediction (with calibrator + FX) → :class:`PredictionA`.

Falls back to the catalog-based three-tier estimator (Tier 3 "category only")
when no XGBoost bundle or no identification metadata is available, so the
pipeline is always usable end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ccdp.costing import Catalog, load_active
from ccdp.data.schema import DAMAGE_TYPES, infer_part_from_damage
from ccdp.identification.car_identifier import IdentificationResult
from ccdp.identification.fallback_estimator import estimate as fallback_estimate
from ccdp.infer.base import BaseVariantPipeline
from ccdp.models.damage_classifier import build_damage_classifier, extract_features
from ccdp.registry import load_checkpoint, production_target
from ccdp.utils import eval_transform, pick_device


@dataclass
class PredictionA:
    """Structured response from :meth:`VariantAPipeline.predict`."""

    damage_types: list[str]
    parts: list[str]
    cost_usd: float
    currency: str
    cost: float
    tier: str                                # 'exact' | 'nearest_class' | 'category_only'
    provenance: str
    catalog_id: Optional[str] = None
    fx_snapshot: dict = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    bundle_run_id: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class VariantAPipeline(BaseVariantPipeline):
    """Image-only damage recognition + cost regression (no localization)."""

    xgb_variant_name = "xgb_a"

    def __init__(
        self,
        classifier_ckpt: Optional[Path] = None,
        xgb_bundle_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(xgb_bundle_dir=xgb_bundle_dir)
        self.device = device or pick_device()
        self.classifier_ckpt = classifier_ckpt or production_target("classifier")
        self.classifier = build_damage_classifier(
            num_classes=len(DAMAGE_TYPES),
            pretrained=(self.classifier_ckpt is None),
        )
        if self.classifier_ckpt and Path(self.classifier_ckpt).exists():
            checkpoint = load_checkpoint(Path(self.classifier_ckpt), map_location=str(self.device))
            self.classifier.load_state_dict(checkpoint["model"])
        self.classifier = self.classifier.to(self.device).eval()
        self.transform = eval_transform(224)

    # -- public API --------------------------------------------------------

    def predict(
        self,
        image,
        metadata: Optional[IdentificationResult] = None,
        threshold: float = 0.5,
        currency: str = "USD",
        catalog: Optional[Catalog] = None,
    ) -> PredictionA:
        """Run end-to-end inference on a single image.

        ``image`` accepts a path-like (``str`` / ``Path``) or an already-opened
        ``PIL.Image`` — useful from the API where the bytes are in memory.

        ``threshold`` is the sigmoid-probability cutoff above which a damage
        class is reported. Default 0.5; raise to ~0.7 if you're seeing false
        positives on undamaged or out-of-distribution images.

        Steps:
          1. Forward the image through the classifier; threshold sigmoid probs.
          2. Forward through the backbone-only path to extract 2048-d features.
          3. If we have XGBoost + identification metadata, predict cost with it;
             otherwise call the three-tier catalog fallback.
          4. Currency-convert + assemble the response with full provenance.
        """
        catalog = catalog or load_active()

        damage_types, probabilities, image_features = self._forward(image, threshold=threshold)
        parts = self._infer_parts(damage_types)

        if self._can_use_xgb(metadata):
            cost_usd, tier, provenance = self._predict_via_xgb(
                image_features, metadata, catalog,
            )
            warning: Optional[str] = None
        else:
            cost_usd, tier, provenance, warning = self._predict_via_catalog(
                parts, metadata, catalog,
            )

        amount, fx_snapshot = self.convert_currency(cost_usd, currency)

        return PredictionA(
            damage_types=damage_types,
            parts=parts,
            cost_usd=round(cost_usd, 2),
            currency=currency.upper(),
            cost=round(amount, 2),
            tier=tier,
            provenance=provenance,
            catalog_id=catalog.catalog_id,
            fx_snapshot=fx_snapshot,
            probabilities=probabilities,
            bundle_run_id=(Path(self.xgb_bundle_dir).name if self.xgb_bundle_dir else None),
            warning=warning,
        )

    # -- internals --------------------------------------------------------

    def _forward(self, image, threshold: float = 0.5):
        """Single forward pass; returns (damage_types, prob_dict, 2048-d features).

        ``image`` may be a path-like *or* an already-opened PIL.Image.
        ``threshold`` controls which classes are surfaced in ``damage_types``;
        the full per-class probability dict is always returned regardless.
        """
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            features = extract_features(self.classifier, x).cpu().numpy().flatten()
        damage_types = [
            DAMAGE_TYPES[i] for i, p in enumerate(probs) if p >= threshold
        ]
        prob_dict = {DAMAGE_TYPES[i]: float(probs[i]) for i in range(len(DAMAGE_TYPES))}
        return damage_types, prob_dict, features

    @staticmethod
    def _infer_parts(damage_types: list[str]) -> list[str]:
        """Map damage types to likely parts without bbox info (Variant A has none).

        Returns an empty list when no mapping is possible — caller treats the
        empty list as "no localization data; defer to fallback estimator".
        """
        parts: list[str] = []
        for dt in damage_types:
            part = infer_part_from_damage(dt, bbox_center=None, damage_location="unknown")
            if part:
                parts.append(part)
        return parts

    def _can_use_xgb(self, metadata: Optional[IdentificationResult]) -> bool:
        """XGBoost is usable only when a bundle is loaded AND we have a make."""
        if self.xgb_model is None or self.xgb_bundle is None:
            return False
        return metadata is not None and metadata.make is not None

    def _predict_via_xgb(
        self,
        features,
        metadata: IdentificationResult,
        catalog: Catalog,
    ) -> tuple[float, str, str]:
        """Build the XGBoost(A) feature row and delegate to BaseVariantPipeline."""
        if self.xgb_model is None or self.xgb_bundle is None:
            raise RuntimeError("XGBoost not available; check `_can_use_xgb` upstream.")
        row = {f"f_{i}": float(v) for i, v in enumerate(features)}
        row.update({
            "year": metadata.year or 2015,
            "make": metadata.make or "unknown",
            "body_type": metadata.body_type or "unknown",
            "segment": metadata.segment or "unknown",
        })
        return self.run_xgb(row, confidence=(metadata.confidence or 0.0), catalog=catalog)

    @staticmethod
    def _predict_via_catalog(
        parts: list[str],
        metadata: Optional[IdentificationResult],
        catalog: Catalog,
    ) -> tuple[float, str, str, Optional[str]]:
        """Fallback path: three-tier estimator using only the catalog."""
        est = fallback_estimate(
            parts_with_severity={p: "moderate" for p in parts},
            identification=metadata,
            catalog=catalog,
        )
        return est.cost_usd, est.tier, est.provenance, est.warning
