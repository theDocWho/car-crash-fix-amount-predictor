"""Pure metric functions used by the comparison report.

Two families:

* **Multi-label classification metrics** for the damage-type classifier
  (per-class P/R/F1, macro/micro F1).
* **Regression metrics** for the XGBoost cost head (RMSE, MAE, MAPE, R²).

We deliberately avoid pulling in sklearn for these — the formulas are short,
the test suite covers them, and we already depend on numpy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def per_class_prf(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    threshold: float = 0.5,
) -> dict:
    """Per-class precision/recall/F1 + macro/micro F1 from sigmoid probabilities.

    ``probs`` and ``labels`` are both ``(N, C)`` arrays where ``N`` is the
    number of images and ``C == len(class_names)``. The threshold turns
    probabilities into hard predictions.
    """
    preds = (probs >= threshold).astype(np.float32)
    tp = (preds * labels).sum(axis=0)
    fp = (preds * (1 - labels)).sum(axis=0)
    fn = ((1 - preds) * labels).sum(axis=0)
    precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-9), 0.0)
    recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1e-9), 0.0)
    f1 = np.where(precision + recall > 0,
                  2 * precision * recall / np.maximum(precision + recall, 1e-9),
                  0.0)

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(labels[:, i].sum()),
        }
        for i in range(len(class_names))
    }
    macro_f1 = float(f1.mean())
    micro_tp, micro_fp, micro_fn = float(tp.sum()), float(fp.sum()), float(fn.sum())
    micro_p = micro_tp / max(micro_tp + micro_fp, 1)
    micro_r = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = (
        2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
        if (micro_p + micro_r) > 0 else 0.0
    )
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
    }


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def multiclass_prf(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Per-class precision/recall/F1 + macro/micro F1 for a single-label
    multiclass problem (e.g. the ResNet-50 identifier).

    Args:
        preds: integer array of shape ``(N,)`` — predicted class index per sample.
        labels: integer array of shape ``(N,)`` — ground-truth class index.
        class_names: length-``C`` sequence — used as keys in the per-class dict.

    Same return schema as :func:`per_class_prf` plus ``"accuracy"`` and
    ``"confusion"`` (a ``(C, C)`` int ndarray, rows=true, cols=pred). We avoid
    sklearn so this stays import-light; the formulas are textbook.
    """
    preds = np.asarray(preds, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    C = len(class_names)
    confusion = np.zeros((C, C), dtype=np.int64)
    for t, p in zip(labels, preds):
        if 0 <= t < C and 0 <= p < C:
            confusion[t, p] += 1

    tp = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)              # true count per class
    predicted = confusion.sum(axis=0).astype(np.float64)            # predicted count per class
    precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    f1 = np.where(
        precision + recall > 0,
        2 * precision * recall / np.maximum(precision + recall, 1e-9),
        0.0,
    )

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(C)
    }
    n = int(labels.size)
    accuracy = float(tp.sum() / max(n, 1))
    macro_f1 = float(f1.mean())
    # micro-F1 == accuracy for single-label multiclass; include for symmetry.
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "micro_f1": accuracy,
        "accuracy": accuracy,
        "confusion": confusion,
        "class_names": list(class_names),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, MAPE (percent), R² on a 1-D vector of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"rmse": 0.0, "mae": 0.0, "mape_pct": 0.0, "r2": 0.0, "n": 0}
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    safe_true = np.where(y_true == 0, 1, y_true)
    mape = float(np.mean(np.abs(diff / safe_true)) * 100)
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1.0
    r2 = float(1.0 - ss_res / ss_tot)
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "r2": r2, "n": int(len(y_true))}
