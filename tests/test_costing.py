"""Unit tests for ccdp.costing.catalog and calibrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.costing import catalog as catmod
from ccdp.costing.calibrator import Calibrator


def test_build_seed_catalog_is_self_consistent():
    cat = catmod.build_seed_catalog()
    assert cat.currency == "USD"
    assert len(cat.parts) >= 10
    assert cat.median_cost() > 0
    for name, pc in cat.parts.items():
        assert set(pc.base_cost) >= {"economy", "mid", "luxury"}, name
        assert set(pc.severity_multiplier) >= {"minor", "moderate", "severe"}, name
        assert set(pc.labor_hours) >= {"minor", "moderate", "severe"}, name


def test_save_load_roundtrip(tmp_path: Path):
    cat = catmod.build_seed_catalog()
    catmod.save(cat, tmp_path)
    loaded = catmod.load(cat.catalog_id, tmp_path)
    assert loaded.catalog_id == cat.catalog_id
    assert set(loaded.parts) == set(cat.parts)
    assert loaded.median_cost() == cat.median_cost()


def test_activate_and_load_active(tmp_path: Path):
    cat = catmod.build_seed_catalog()
    catmod.save(cat, tmp_path)
    catmod.activate(cat.catalog_id, tmp_path)
    active = catmod.load_active(tmp_path)
    assert active.catalog_id == cat.catalog_id

    listed = catmod.list_catalogs(tmp_path)
    assert any(r["is_active"] and r["catalog_id"] == cat.catalog_id for r in listed)


def test_estimate_with_known_parts(tmp_path: Path):
    cat = catmod.build_seed_catalog()
    # known seed parts
    cost = cat.estimate(["front_bumper", "hood"], segment="mid")
    assert cost > 0
    # unknown part is silently skipped (tier-3 fallback is best-effort)
    cost2 = cat.estimate(["front_bumper", "hood", "made_up_part"], segment="mid")
    assert cost2 == cost

    # luxury > mid > economy
    economy = cat.estimate(["front_bumper"], segment="economy")
    mid = cat.estimate(["front_bumper"], segment="mid")
    luxury = cat.estimate(["front_bumper"], segment="luxury")
    assert economy < mid < luxury


def test_diff_detects_changes(tmp_path: Path):
    a = catmod.build_seed_catalog(tag="a")
    catmod.save(a, tmp_path)
    b = catmod.build_seed_catalog(tag="b")
    # bump prices on b
    for pc in b.parts.values():
        pc.base_cost["mid"] *= 1.10
    catmod.save(b, tmp_path)

    d = catmod.diff(a.catalog_id, b.catalog_id, tmp_path)
    assert all(info["status"] == "changed" for info in d.values())
    assert all(abs(info["pct_change"] - 10.0) < 1e-6 for info in d.values())


def test_calibrator_scales_linearly():
    a = catmod.build_seed_catalog(tag="train")
    cal = Calibrator.from_catalog(a)
    assert cal.scale(100.0, a) == pytest.approx(100.0)

    b = catmod.build_seed_catalog(tag="active")
    for pc in b.parts.values():
        pc.base_cost["mid"] *= 2.0
        pc.base_cost["economy"] *= 2.0
        pc.base_cost["luxury"] *= 2.0
    factor = cal.scale_factor(b)
    assert factor > 1.0
    assert cal.scale(100.0, b) == pytest.approx(100.0 * factor)
