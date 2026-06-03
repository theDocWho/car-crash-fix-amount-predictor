"""Tests for Variant C seg mask-area feature aggregation."""

from __future__ import annotations

from ccdp.data.schema import DAMAGE_TYPES
from ccdp.train.extract_bbox_features import bbox_stats
from ccdp.train.extract_seg_features import seg_mask_stats


def test_seg_stats_schema_matches_bbox_stats():
    # XGBoost(C) joins seg features exactly like XGBoost(B) joins bbox features,
    # so the column schema must be identical.
    from ccdp.data.schema import BBox
    bb = bbox_stats([BBox(damage_type="dent", x_center=0.5, y_center=0.5, width=0.2, height=0.2)])
    sg = seg_mask_stats([("dent", 0.04)])
    assert set(bb.keys()) == set(sg.keys())


def test_seg_stats_aggregates_area_and_counts():
    s = seg_mask_stats([("dent", 0.1), ("dent", 0.05), ("scratch", 0.2)])
    assert s["n_damage_regions"] == 3
    assert s["count_dent"] == 2
    assert s["count_scratch"] == 1
    assert abs(s["area_dent"] - 0.15) < 1e-9
    assert abs(s["total_area_pct"] - 0.35) < 1e-9
    assert abs(s["largest_area_pct"] - 0.2) < 1e-9


def test_seg_stats_empty():
    s = seg_mask_stats([])
    assert s["n_damage_regions"] == 0
    assert s["total_area_pct"] == 0.0
    for dt in DAMAGE_TYPES:
        assert s[f"count_{dt}"] == 0.0
