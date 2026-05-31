"""Tests for the YOLOv8-seg quick-win path: CarDD polygon labels + trainer config."""

from __future__ import annotations

from pathlib import Path

from ccdp.data.cardd_yolo import (
    DEFAULT_SEG_ROOT,
    _write_seg_label_file,
    normalize_polygon,
)
from ccdp.train.train_yolov8 import YoloConfig, _resolve_data


def test_normalize_polygon():
    out = normalize_polygon([10, 20, 30, 40, 50, 60], w=100, h=200)
    assert out == [0.1, 0.1, 0.3, 0.2, 0.5, 0.3]


def test_write_seg_label_format(tmp_path: Path):
    p = tmp_path / "lbl.txt"
    _write_seg_label_file(p, [(1, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), (0, [0.0, 0.0, 1.0, 1.0, 0.5, 0.9])])
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    # "<class> x y x y ..." — class is int, coords are floats
    first = lines[0].split()
    assert first[0] == "1"
    assert all(0.0 <= float(v) <= 1.0 for v in first[1:])
    assert (len(first) - 1) % 2 == 0


def test_config_seg_and_variant_defaults():
    det = YoloConfig()
    assert det.seg is False and det.variant == "detector"
    seg = YoloConfig(seg=True, variant="yoloseg", model="yolov8n-seg.pt")
    assert seg.seg is True and seg.variant == "yoloseg"


def test_resolve_data_passthrough_named_dataset():
    # a bare ultralytics name (not a local file) is passed through verbatim
    cfg = YoloConfig(seg=True)
    assert _resolve_data(cfg, "carparts-seg.yaml") == "carparts-seg.yaml"


def test_resolve_data_local_file_absolutised(tmp_path: Path):
    f = tmp_path / "data.yaml"
    f.write_text("path: x")
    cfg = YoloConfig()
    out = _resolve_data(cfg, f)
    assert out == str(f.resolve())


def test_seg_root_distinct_from_detect_root():
    from ccdp.data.cardd_yolo import DEFAULT_ROOT
    assert DEFAULT_SEG_ROOT != DEFAULT_ROOT
