"""Tests for the non-ML car identification stages."""

from __future__ import annotations

from pathlib import Path

from ccdp.identification.car_identifier import (
    from_filename,
    identify,
    infer_segment,
)


def test_filename_extracts_make_year_body_type():
    out = from_filename(Path("/data/raw/honda_civic_sedan_2018_damaged.jpg"))
    assert out.get("make") == "honda"
    assert out.get("year") == 2018
    assert out.get("body_type") == "sedan"


def test_filename_handles_folder_hints():
    out = from_filename(Path("/data/raw/Toyota/Camry/2020/img_042.jpg"))
    assert out.get("make") == "toyota"
    assert out.get("year") == 2020


def test_filename_no_match():
    out = from_filename(Path("/tmp/IMG_1234.jpg"))
    assert "make" not in out
    assert "year" not in out


def test_segment_inference():
    assert infer_segment("bmw") == "luxury"
    assert infer_segment("kia") == "economy"
    assert infer_segment("honda") == "mid"
    assert infer_segment(None) == "unknown"


def test_identify_pipeline_no_image(tmp_path: Path):
    # Use a non-existent path; identify() should fall through stages cleanly.
    p = tmp_path / "audi_a4_2019_dented.jpg"
    res = identify(p, use_ocr=False)
    assert res.make == "audi"
    assert res.year == 2019
    assert res.source == "filename"
    assert res.segment == "luxury"
    assert "filename" in res.stages_tried


def test_identify_unmatched_filename(tmp_path: Path):
    p = tmp_path / "IMG_0001.jpg"
    res = identify(p, use_ocr=False)
    assert res.make is None
    assert res.source == "none"
    assert res.confidence == 0.0
