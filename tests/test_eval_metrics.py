"""Tests for the pure metric functions used by the comparison report."""

from __future__ import annotations

import numpy as np

from ccdp.eval.metrics import per_class_prf, regression_metrics


def test_per_class_prf_perfect_predictions():
    classes = ["a", "b", "c"]
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    probs = labels.copy()  # perfect
    m = per_class_prf(probs, labels, classes)
    assert m["macro_f1"] == 1.0
    assert m["micro_f1"] == 1.0
    for c in classes:
        assert m["per_class"][c]["precision"] == 1.0
        assert m["per_class"][c]["recall"] == 1.0


def test_per_class_prf_handles_no_predictions():
    classes = ["a", "b"]
    labels = np.array([[1, 0], [0, 1]], dtype=float)
    probs = np.zeros_like(labels)
    m = per_class_prf(probs, labels, classes)
    assert m["macro_f1"] == 0.0
    assert m["per_class"]["a"]["recall"] == 0.0


def test_per_class_prf_support_counts_match_labels():
    classes = ["a", "b"]
    labels = np.array([[1, 0], [1, 1], [0, 1]], dtype=float)
    probs = np.zeros_like(labels)
    m = per_class_prf(probs, labels, classes)
    assert m["per_class"]["a"]["support"] == 2
    assert m["per_class"]["b"]["support"] == 2


def test_regression_metrics_perfect_zero_error():
    y_true = [100, 200, 300]
    y_pred = [100, 200, 300]
    m = regression_metrics(y_true, y_pred)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["mape_pct"] == 0.0
    assert m["r2"] == 1.0
    assert m["n"] == 3


def test_regression_metrics_known_values():
    y_true = [100, 200]
    y_pred = [110, 180]   # errors: +10, -20
    m = regression_metrics(y_true, y_pred)
    # MAE = (10+20)/2 = 15
    assert abs(m["mae"] - 15.0) < 1e-6
    # RMSE = sqrt((100+400)/2) = sqrt(250) ≈ 15.81
    assert abs(m["rmse"] - np.sqrt(250)) < 1e-6
    # MAPE = ((10/100) + (20/200))/2 * 100 = 10.0
    assert abs(m["mape_pct"] - 10.0) < 1e-6


def test_regression_metrics_empty_input_safe():
    m = regression_metrics([], [])
    assert m["n"] == 0
    assert m["rmse"] == 0.0
