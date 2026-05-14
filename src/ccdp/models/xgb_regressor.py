"""XGBoost regressor that learns image features + tabular metadata -> cost.

Variant A consumes only image features (no bbox stats); Variant B (Phase 2B)
will append box-derived features. Both share this class.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class XGBRegressorBundle:
    """Saved alongside the model so inference can rebuild the same feature row."""
    feature_columns: list[str]
    categorical_columns: list[str]
    target_column: str
    variant: str                       # 'a' or 'b'
    training_catalog_id: Optional[str] = None
    training_median: Optional[float] = None
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    metrics: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "XGBRegressorBundle":
        return cls(**d)


def make_feature_matrix(df, bundle: XGBRegressorBundle):
    """Return (X, feature_names) aligned to `bundle.feature_columns`.

    Categorical columns are one-hot encoded using `pd.get_dummies` with
    `dummy_na=False`, then re-indexed against `bundle.feature_columns` so the
    columns match training time exactly (missing categories -> 0).
    """
    import pandas as pd
    df = df.copy()
    cat = pd.get_dummies(df[bundle.categorical_columns], dummy_na=False,
                         drop_first=False, dtype=float)
    numeric_cols = [c for c in bundle.feature_columns if c not in cat.columns]
    other_cols = [c for c in numeric_cols if c in df.columns]
    base = df[other_cols].astype(float).fillna(0.0) if other_cols else None
    X = pd.concat([base, cat], axis=1) if base is not None else cat
    X = X.reindex(columns=bundle.feature_columns, fill_value=0.0)
    return X
