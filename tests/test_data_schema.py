"""Tests for the canonical record schema and label mapping."""

from __future__ import annotations

from pathlib import Path

from ccdp.data.schema import CANONICAL_PARTS, Record, map_to_canonical_part


def test_canonical_parts_match_seed_catalog():
    from ccdp.costing import build_seed_catalog
    cat = build_seed_catalog()
    # every canonical part must exist in the seed catalog so cost lookups never miss
    missing = [p for p in CANONICAL_PARTS if p not in cat.parts]
    assert not missing, f"Catalog missing canonical parts: {missing}"


def test_label_mapping_known_cases():
    assert map_to_canonical_part("Front Bumper") == "front_bumper"
    assert map_to_canonical_part("rear-bumper") == "rear_bumper"
    assert map_to_canonical_part("hood") == "hood"
    assert map_to_canonical_part("bonnet") == "hood"
    assert map_to_canonical_part("Head Light") == "headlight"
    assert map_to_canonical_part("WindScreen") == "windshield"
    assert map_to_canonical_part("Boot") == "trunk"
    assert map_to_canonical_part("rim") == "wheel"


def test_label_mapping_unknown_returns_none():
    assert map_to_canonical_part("flux capacitor") is None
    assert map_to_canonical_part("") is None


def test_record_image_id_stable():
    r = Record(image_path=Path("/tmp/foo/img_001.jpg"), dataset="iaai")
    assert r.image_id == "iaai/img_001.jpg"
    assert r.is_identified is False
    r.make, r.model = "honda", "civic"
    assert r.is_identified is True
