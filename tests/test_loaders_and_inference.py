"""Tests for damage_type/location schema, the part-inference heuristic, and
the dataset loaders (smoke-only against real data when present)."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pytest

from ccdp.data import DAMAGE_TYPES, infer_part_from_damage
from ccdp.data.loaders import (
    CARDD_ROOT,
    COMPREHENSIVE_ROOT,
    IAAI_ROOT,
    _classify_iaai_damage_location,
    _normalize_body_style,
    iter_cardd,
    iter_comprehensive,
    iter_iaai,
)


# ---------- inference rules --------------------------------------------


def test_damage_types_canonical():
    assert set(DAMAGE_TYPES) == {
        "dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat",
    }


def test_infer_part_position_independent_types():
    # tire_flat → wheel regardless of position
    assert infer_part_from_damage("tire_flat", (0.1, 0.5)) == "wheel"
    assert infer_part_from_damage("tire_flat", None) == "wheel"

    # glass_shatter → windshield
    assert infer_part_from_damage("glass_shatter", (0.5, 0.2)) == "windshield"

    # lamp_broken → headlight by default, taillight if rear
    assert infer_part_from_damage("lamp_broken", (0.5, 0.3)) == "headlight"
    assert infer_part_from_damage("lamp_broken", None, damage_location="rear") == "taillight"


def test_infer_part_position_dependent_types():
    # dent in front-top → hood
    assert infer_part_from_damage("dent", (0.7, 0.2), "front") == "hood"
    # dent in rear-top → trunk
    assert infer_part_from_damage("dent", (0.3, 0.2), "rear") == "trunk"
    # dent in front-bottom → front_bumper
    assert infer_part_from_damage("dent", (0.7, 0.85), "front") == "front_bumper"
    # dent in front-mid → front_door
    assert infer_part_from_damage("dent", (0.6, 0.5), "front") == "front_door"
    # ambiguous → None
    assert infer_part_from_damage("dent", None) is None


def test_infer_part_unknown_type():
    assert infer_part_from_damage("foobar", (0.5, 0.5)) is None


# ---------- iaai helpers ------------------------------------------------


def test_normalize_body_style():
    assert _normalize_body_style("SEDAN") == "sedan"
    assert _normalize_body_style("CREW CAB") == "pickup"
    assert _normalize_body_style("Sport Utility") == "suv"
    assert _normalize_body_style(None) == "unknown"
    assert _normalize_body_style("Spaceship") == "unknown"


def test_iaai_damage_location_classifier():
    assert _classify_iaai_damage_location("FRONT END") == "front"
    assert _classify_iaai_damage_location("REAR") == "rear"
    assert _classify_iaai_damage_location("LEFT FRONT") == "front"
    assert _classify_iaai_damage_location("FRONT & REAR") == "unknown"  # both → ambiguous
    assert _classify_iaai_damage_location("") == "unknown"
    assert _classify_iaai_damage_location("ALL OVER") == "unknown"


# ---------- loader smoke tests (only when real data present) -----------


@pytest.mark.skipif(
    not (CARDD_ROOT / "annotations" / "instances_val2017.json").exists(),
    reason="CarDD val split not on disk",
)
def test_cardd_loader_real_data_shape():
    records = list(islice(iter_cardd(splits=("val",)), 50))
    assert records, "expected at least one CarDD record"
    r = records[0]
    assert r.dataset == "cardd"
    for d in r.damage_types:
        assert d in DAMAGE_TYPES, f"unexpected damage_type {d!r}"
    for b in r.bboxes:
        assert 0.0 <= b.x_center <= 1.0
        assert 0.0 <= b.y_center <= 1.0


@pytest.mark.skipif(
    not COMPREHENSIVE_ROOT.exists(),
    reason="comprehensive dataset not on disk",
)
def test_comprehensive_loader_real_data():
    records = list(islice(iter_comprehensive(), 20))
    assert records
    locs = {r.damage_location for r in records}
    conds = {r.damage_condition for r in records}
    assert locs <= {"front", "rear"}
    assert conds <= {"normal", "crushed", "breakage"}


@pytest.mark.skipif(
    not IAAI_ROOT.exists(),
    reason="iaai dataset not on disk",
)
def test_iaai_loader_real_data_metadata_only():
    records = list(islice(iter_iaai(), 100))
    assert records
    # at least some rows should have real make/year (free fields)
    assert any(r.make for r in records)
    assert any(r.year for r in records)
    # cost columns are paywalled in the free sample → all should be None
    assert all(r.cost is None for r in records)
    assert all(r.cost_usd is None for r in records)
    # iaai records have no on-disk image
    for r in records:
        assert r.image_path.parts[0] == "iaai"
