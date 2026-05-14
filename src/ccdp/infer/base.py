"""Shared building blocks for the Variant A and Variant B inference pipelines.

The two variants do different vision work (whole-image classifier vs. detector)
but their **downstream** path is identical:

1. Compute a cost estimate, either via XGBoost (if a bundle is loaded and the
   user supplied identification metadata) or via the catalog fallback estimator.
2. Calibrate the XGBoost output to the *currently active* parts-cost catalog
   (so price drift is corrected without retraining).
3. Convert to the requested currency via the FX module.
4. Bundle full provenance (catalog id, FX snapshot, tier, training-time
   catalog id, bundle run id) into the response.

The :class:`BaseVariantPipeline` captures that shared downstream and exposes
two small hooks for subclasses:

* :meth:`load_xgb_bundle`   — load `bundle.json` + `best.ubj` if available.
* :meth:`run_xgb`           — produce a calibrated cost prediction from a
                              feature row plus identification metadata.

Subclasses (`VariantAPipeline`, `VariantBPipeline`) layer their vision-specific
forward pass on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ccdp.costing import Calibrator, Catalog, fx as fxmod
from ccdp.models.xgb_regressor import XGBRegressorBundle, make_feature_matrix
from ccdp.registry import production_target


class BaseVariantPipeline:
    """Behaviour shared by all inference variants: XGBoost handling, FX, provenance."""

    #: Registry variant name used to discover the production XGBoost bundle.
    #: Subclasses override this (``"xgb_a"`` or ``"xgb_b"``).
    xgb_variant_name: str = "xgb_a"

    def __init__(self, xgb_bundle_dir: Optional[Path] = None):
        # Resolve the XGBoost bundle directory. Either:
        # - explicit dir passed in (used by tests / Phase 3 demo), OR
        # - the run dir reached by *resolving* `production/<variant>.pt` so we
        #   follow the symlink chain to its real location (where `bundle.json`
        #   actually lives — the production symlink itself does not).
        if xgb_bundle_dir is None:
            prod = production_target(self.xgb_variant_name)
            xgb_bundle_dir = prod.resolve().parent if prod else None
        self.xgb_bundle_dir = xgb_bundle_dir
        self.xgb_model = None
        self.xgb_bundle: Optional[XGBRegressorBundle] = None
        if self.xgb_bundle_dir and (Path(self.xgb_bundle_dir) / "bundle.json").exists():
            self._load_xgb_bundle()

    # -- loading -----------------------------------------------------------

    def _load_xgb_bundle(self) -> None:
        """Read the booster (``best.ubj``) and the feature-schema sidecar."""
        import xgboost as xgb
        bundle_dir = Path(self.xgb_bundle_dir)
        booster_path = bundle_dir / "best.ubj"
        if not booster_path.exists():
            return
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(booster_path))
        with (bundle_dir / "bundle.json").open() as f:
            self.xgb_bundle = XGBRegressorBundle.from_dict(json.load(f))

    # -- inference helpers -------------------------------------------------

    def run_xgb(
        self,
        feature_row: dict,
        confidence: float,
        catalog: Catalog,
    ) -> tuple[float, str, str]:
        """Predict cost in USD and return ``(cost, tier, provenance)``.

        ``feature_row`` is a flat dict of ``{column_name: value}``. Categorical
        columns get one-hot encoded against the bundle's training schema so the
        feature vector at inference matches what XGBoost saw during training.

        Tier is ``"exact"`` if ``confidence >= 0.6`` else ``"nearest_class"``.
        """
        import pandas as pd
        import xgboost as xgb

        bundle = self.xgb_bundle
        if bundle is None or self.xgb_model is None:
            raise RuntimeError("XGBoost bundle not loaded; call _load_xgb_bundle().")

        df = pd.DataFrame([feature_row])
        x = make_feature_matrix(df, bundle)
        raw = float(self.xgb_model.predict(
            xgb.DMatrix(x.values, feature_names=list(x.columns)),
        )[0])

        if bundle.training_median:
            cal = Calibrator(
                training_catalog_id=bundle.training_catalog_id or "unknown",
                training_median=bundle.training_median,
            )
            cost = cal.scale(raw, catalog)
        else:
            cost = raw

        tier = "exact" if confidence >= 0.6 else "nearest_class"
        provenance = (
            f"{self.xgb_variant_name}({tier}); "
            f"training_catalog={bundle.training_catalog_id}; "
            f"calibrated to {catalog.catalog_id}"
        )
        return cost, tier, provenance

    # -- output construction ----------------------------------------------

    @staticmethod
    def convert_currency(cost_usd: float, currency: str) -> tuple[float, dict]:
        """Convert USD → ``currency`` and return ``(amount, fx_snapshot)``.

        ``fx_snapshot`` is ``{}`` when no conversion happens (target was USD).
        """
        amount, fx_used = fxmod.convert(cost_usd, "USD", currency)
        if fx_used is None:
            return amount, {}
        return amount, {
            "rate": fx_used.rate,
            "base": fx_used.base,
            "target": fx_used.target,
            "source": fx_used.source,
            "fetched_at": fx_used.fetched_at,
        }
