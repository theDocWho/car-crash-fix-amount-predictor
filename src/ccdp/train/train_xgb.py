"""Train XGBoost(A) on the synthetic cost target.

Inputs:
    data/processed/cardd_features.parquet         # image features (f_0..f_2047)
    data/processed/cardd_cost_targets.parquet     # metadata + cost_usd

Output:
    checkpoints/xgb_a/run_<ts>_<tag>/best.ubj      # booster
    checkpoints/xgb_a/run_<ts>_<tag>/bundle.json   # feature schema + training catalog id

Metrics: RMSE, MAE, MAPE, R² on val + test splits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ccdp.costing import load_active
from ccdp.models.xgb_regressor import XGBRegressorBundle, make_feature_matrix
from ccdp.registry import create_run, update_metrics

FEATURES_DEFAULT = Path("data/processed/cardd_features.parquet")
TARGETS_DEFAULT = Path("data/processed/cardd_cost_targets.parquet")
BBOX_FEATURES_DEFAULT = Path("data/processed/cardd_bbox_features.parquet")
SEG_FEATURES_DEFAULT = Path("data/processed/cardd_seg_features.parquet")


@dataclass
class XGBConfig:
    n_estimators: int = 600
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.7
    min_child_weight: float = 3.0
    early_stopping_rounds: int = 25
    tag: str = "xgb_a"
    variant: str = "a"
    seed: int = 42


def _metrics(y_true, y_pred) -> dict:
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2 = float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "r2": r2}


def train(
    cfg: XGBConfig,
    features_path: Path = FEATURES_DEFAULT,
    targets_path: Path = TARGETS_DEFAULT,
    bbox_features_path: Optional[Path] = None,
    seg_features_path: Optional[Path] = None,
) -> Path:
    import pandas as pd
    import xgboost as xgb

    feats = pd.read_parquet(features_path)
    targs = pd.read_parquet(targets_path)
    df = feats.merge(targs, on=["image_id", "split", "damage_types"], how="inner")
    if df.empty:
        raise RuntimeError("merge produced empty df — features and targets out of sync")

    # Variant B joins bbox-derived region features; Variant C joins the same
    # schema computed from YOLOv8-seg masks (true damaged-area fraction).
    region_feat_cols: list[str] = []
    _REGION = {
        "b": ("bbox", bbox_features_path, BBOX_FEATURES_DEFAULT,
              "ccdp train extract-bbox-features [--gt]"),
        "c": ("seg", seg_features_path, SEG_FEATURES_DEFAULT,
              "ccdp train extract-seg-features [--gt]"),
    }
    if cfg.variant in _REGION:
        kind, path, default, how = _REGION[cfg.variant]
        path = path or default
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Variant {cfg.variant.upper()} needs {kind} features at {path}. Run `{how}` first."
            )
        reg = pd.read_parquet(path)
        drop = [c for c in ("image_path", "damage_types") if c in reg.columns]
        reg = reg.drop(columns=drop)
        df = df.merge(reg, on=["image_id", "split"], how="inner")
        region_feat_cols = [c for c in reg.columns
                            if c not in ("image_id", "split") and not c.startswith("f_")]
        print(f"[variant {cfg.variant}] joined {len(region_feat_cols)} {kind} features")

    image_feat_cols = [c for c in df.columns if c.startswith("f_")]
    categorical_cols = ["make", "body_type", "segment"]
    numeric_extra = ["year"] + region_feat_cols
    target_col = "cost_usd"

    import pandas as pd
    cat = pd.get_dummies(df[categorical_cols], dummy_na=False, dtype=float)
    feature_columns = image_feat_cols + numeric_extra + list(cat.columns)

    catalog = load_active()
    bundle = XGBRegressorBundle(
        feature_columns=feature_columns,
        categorical_columns=categorical_cols,
        target_column=target_col,
        variant=cfg.variant,
        training_catalog_id=catalog.catalog_id,
        training_median=catalog.median_cost(),
    )

    def _split(name: str):
        sub = df[df["split"] == name]
        X = make_feature_matrix(sub, bundle)
        y = sub[target_col].astype(float).values
        return X, y

    X_train, y_train = _split("train")
    X_val, y_val = _split("val")
    X_test, y_test = _split("test")
    print(f"[data] train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
          f"features={X_train.shape[1]}")

    model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=cfg.seed,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    val_metrics = _metrics(y_val, val_pred)
    test_metrics = _metrics(y_test, test_pred)
    bundle.n_train, bundle.n_val = len(X_train), len(X_val)
    bundle.metrics = {"val": val_metrics, "test": test_metrics}
    print(f"[val ] {val_metrics}")
    print(f"[test] {test_metrics}")

    run_dir = create_run(
        variant=f"xgb_{cfg.variant}", tag=cfg.tag,
        training_catalog_id=catalog.catalog_id,
        notes=f"Variant {cfg.variant.upper()} XGBoost on CarDD features + synthetic cost target",
    )
    booster_path = run_dir / "best.ubj"
    model.get_booster().save_model(str(booster_path))
    (run_dir / "bundle.json").write_text(json.dumps(bundle.to_dict(), indent=2))
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()))
    update_metrics(run_dir.name.replace("run_", ""), {
        "val": val_metrics, "test": test_metrics,
        "n_train": bundle.n_train, "n_val": bundle.n_val,
    })

    # also write last.pt-style symlink so registry promotion stays uniform
    last = run_dir / "best.pt"   # naming kept consistent with other variants
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(booster_path.name)
    print(f"[done] saved -> {run_dir}")
    return run_dir / "best.pt"
