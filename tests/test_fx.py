"""Unit tests for ccdp.costing.fx (no network calls — cache + manual paths only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ccdp.costing import fx as fxmod


def test_manual_set_and_get(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fx_cache.json"
    fxmod.manual_set("USD", "INR", 83.5, cache_path=cache)
    fr = fxmod.get_rate("USD", "INR", cache_path=cache)
    assert fr.rate == 83.5
    assert fr.source == "manual_override"


def test_convert_identity_is_noop(tmp_path: Path):
    out, fr = fxmod.convert(100.0, "USD", "USD")
    assert out == 100.0
    assert fr is None


def test_convert_uses_explicit_rate(tmp_path: Path):
    out, fr = fxmod.convert(10.0, "USD", "INR", rate=80.0)
    assert out == 800.0
    assert fr is not None
    assert fr.source == "manual_override"


def test_offline_without_cache_raises(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fx_cache.json"
    monkeypatch.setenv("FX_OFFLINE", "1")
    with pytest.raises(RuntimeError):
        fxmod.get_rate("USD", "INR", cache_path=cache)


def test_offline_with_cache_returns_cached(tmp_path: Path, monkeypatch):
    cache = tmp_path / "fx_cache.json"
    fxmod.manual_set("USD", "INR", 82.0, cache_path=cache)
    monkeypatch.setenv("FX_OFFLINE", "1")
    fr = fxmod.get_rate("USD", "INR", cache_path=cache)
    assert fr.rate == 82.0


def test_cache_file_format(tmp_path: Path):
    cache = tmp_path / "fx_cache.json"
    fxmod.manual_set("USD", "INR", 83.0, cache_path=cache)
    data = json.loads(cache.read_text())
    assert "USD_INR" in data
    entry = data["USD_INR"]
    assert set(entry) >= {"base", "target", "rate", "fetched_at", "source"}
