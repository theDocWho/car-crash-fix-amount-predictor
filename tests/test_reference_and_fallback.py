"""Tests for the reference table builder, nearest lookup, and three-tier estimator."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from ccdp.costing import build_seed_catalog
from ccdp.identification import fallback_estimator as fb
from ccdp.identification import reference_table as reftab
from ccdp.identification.car_identifier import IdentificationResult


def _sample_rows():
    return [
        {"make": "honda", "model": "civic", "year": 2018, "body_type": "sedan",
         "segment": "mid", "cost_usd": 2800, "dataset": "iaai"},
        {"make": "honda", "model": "civic", "year": 2018, "body_type": "sedan",
         "segment": "mid", "cost_usd": 3000, "dataset": "iaai"},
        {"make": "honda", "model": "civic", "year": 2019, "body_type": "sedan",
         "segment": "mid", "cost_usd": 3200, "dataset": "iaai"},
        {"make": "toyota", "model": "camry", "year": 2019, "body_type": "sedan",
         "segment": "mid", "cost_usd": 3400, "dataset": "iaai"},
        {"make": "bmw", "model": "3-series", "year": 2020, "body_type": "sedan",
         "segment": "luxury", "cost_usd": 7200, "dataset": "iaai"},
    ]


def test_build_and_lookup_exact(tmp_path: Path):
    out = tmp_path / "ref.parquet"
    reftab.build(_sample_rows(), out_path=out)
    r = reftab.nearest(make="honda", model="civic", year=2018, path=out)
    assert r is not None
    assert r["match_how"] == "exact"
    assert r["avg_cost_usd"] == pytest.approx(2900.0)
    assert r["n_samples"] == 2


def test_lookup_make_model_any_year(tmp_path: Path):
    out = tmp_path / "ref.parquet"
    reftab.build(_sample_rows(), out_path=out)
    r = reftab.nearest(make="honda", model="civic", year=2099, path=out)
    assert r["match_how"] == "make_model_any_year"


def test_lookup_segment_body_type_fallback(tmp_path: Path):
    out = tmp_path / "ref.parquet"
    reftab.build(_sample_rows(), out_path=out)
    r = reftab.nearest(make="lamborghini", model="huracan", year=2022,
                       body_type="sedan", segment="luxury", path=out)
    assert r["match_how"] in {"segment_body_type", "body_type", "segment"}


def test_lookup_returns_none_on_empty(tmp_path: Path):
    out = tmp_path / "ref.parquet"
    reftab.build([], out_path=out)
    assert reftab.nearest(make="x", model="y", year=2000, path=out) is None


def test_three_tier_estimator(tmp_path: Path, monkeypatch):
    out = tmp_path / "ref.parquet"
    reftab.build(_sample_rows(), out_path=out)
    monkeypatch.setattr(reftab, "DEFAULT_PATH", out)
    monkeypatch.setattr(fb.reftab, "DEFAULT_PATH", out)

    cat = build_seed_catalog()

    # Tier 1: identified + reference match
    ident = IdentificationResult(
        image_path=Path("/x.jpg"), make="honda", model="civic", year=2018,
        body_type="sedan", segment="mid", confidence=0.9, source="filename",
    )
    est = fb.estimate({"front_bumper": "moderate"}, identification=ident, catalog=cat)
    assert est.tier == "exact"
    assert est.cost_usd > 0
    assert "honda" in est.provenance

    # Tier 3: no identification
    est3 = fb.estimate({"front_bumper": "moderate"}, identification=None, catalog=cat)
    assert est3.tier == "category_only"
    assert est3.warning is not None
