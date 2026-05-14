"""Tests for the CarDD -> YOLO dataset conversion and bbox-stats features."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.data import cardd_yolo
from ccdp.data.schema import DAMAGE_TYPES, BBox, Record
from ccdp.train.extract_bbox_features import bbox_stats


def test_class_index_aligned_with_canonical_taxonomy():
    assert list(cardd_yolo.CLASS_INDEX.keys()) == list(DAMAGE_TYPES)
    for i, dt in enumerate(DAMAGE_TYPES):
        assert cardd_yolo.CLASS_INDEX[dt] == i


def test_write_data_yaml_contents(tmp_path: Path):
    p = cardd_yolo.write_data_yaml(tmp_path)
    text = p.read_text()
    assert "nc: 6" in text
    assert "train: train/images" in text
    assert "val: val/images" in text
    assert "test: test/images" in text
    for dt in DAMAGE_TYPES:
        assert dt in text


def test_write_label_file(tmp_path: Path):
    r = Record(image_path=Path("/x/img.jpg"), dataset="cardd",
               damage_types=["dent", "scratch"],
               bboxes=[
                   BBox(damage_type="dent", x_center=0.5, y_center=0.5, width=0.2, height=0.1),
                   BBox(damage_type="scratch", x_center=0.3, y_center=0.7, width=0.05, height=0.05),
                   BBox(damage_type="not_a_real_type", x_center=0.1, y_center=0.1, width=0.05, height=0.05),
               ])
    out = tmp_path / "labels" / "img.txt"
    cardd_yolo._write_label_file(out, r)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2   # the bogus type is filtered
    for line in lines:
        parts = line.split()
        assert len(parts) == 5
        cls = int(parts[0])
        assert 0 <= cls < len(DAMAGE_TYPES)
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0


def test_bbox_stats_empty():
    s = bbox_stats([])
    assert s["n_damage_regions"] == 0
    assert s["total_area_pct"] == 0
    assert s["largest_area_pct"] == 0
    for dt in DAMAGE_TYPES:
        assert s[f"count_{dt}"] == 0
        assert s[f"area_{dt}"] == 0


def test_bbox_stats_multi_class():
    bbs = [
        BBox(damage_type="dent", x_center=0.5, y_center=0.5, width=0.4, height=0.5),  # area 0.20
        BBox(damage_type="dent", x_center=0.2, y_center=0.2, width=0.1, height=0.1),  # area 0.01
        BBox(damage_type="scratch", x_center=0.7, y_center=0.7, width=0.2, height=0.1),  # area 0.02
    ]
    s = bbox_stats(bbs)
    assert s["n_damage_regions"] == 3
    assert s["count_dent"] == 2
    assert s["count_scratch"] == 1
    assert abs(s["area_dent"] - 0.21) < 1e-6
    assert abs(s["area_scratch"] - 0.02) < 1e-6
    assert abs(s["total_area_pct"] - 0.23) < 1e-6
    assert abs(s["largest_area_pct"] - 0.20) < 1e-6
