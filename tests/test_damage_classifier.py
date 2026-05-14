"""Tests for the CarDD multi-label classifier dataset utilities and model."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.data import damage_dataset as dd
from ccdp.data.schema import DAMAGE_TYPES, Record


def _mk(types: list[str], path: str = "x.jpg", ds: str = "cardd") -> Record:
    return Record(image_path=Path(path), dataset=ds, damage_types=types)


def test_encode_labels_canonical():
    v = dd.encode_labels(["dent", "scratch"])
    assert len(v) == len(DAMAGE_TYPES)
    assert v[DAMAGE_TYPES.index("dent")] == 1.0
    assert v[DAMAGE_TYPES.index("scratch")] == 1.0
    assert sum(v) == 2.0


def test_encode_labels_unknown_dropped():
    v = dd.encode_labels(["dent", "foobar"])
    assert sum(v) == 1.0


def test_split_records_deterministic():
    recs = [_mk(["dent"], path=f"{i}.jpg") for i in range(100)]
    a = dd.split_records(recs, seed=42)
    b = dd.split_records(recs, seed=42)
    assert [r.image_path for r in a[0]] == [r.image_path for r in b[0]]
    n_train, n_val, n_test = len(a[0]), len(a[1]), len(a[2])
    assert n_train + n_val + n_test == 100
    assert abs(n_train - 80) <= 1
    assert abs(n_val - 10) <= 1


def test_pos_weight_inverse_frequency():
    recs = []
    # 90 with 'dent', 10 with 'tire_flat' — tire_flat should be weighted up
    for _ in range(90):
        recs.append(_mk(["dent"]))
    for _ in range(10):
        recs.append(_mk(["tire_flat"]))
    pw = dd.pos_weight(recs)
    i_dent = DAMAGE_TYPES.index("dent")
    i_tf = DAMAGE_TYPES.index("tire_flat")
    # rare class weight > common class weight
    assert pw[i_tf] > pw[i_dent]
    # neg/pos ratio: tire_flat: (100-10)/10 = 9; dent: (100-90)/90 ≈ 0.11 -> floored to 1
    assert abs(pw[i_tf] - 9.0) < 0.01
    assert pw[i_dent] >= 1.0


def test_pos_weight_empty_class_safe():
    recs = [_mk(["dent"]) for _ in range(10)]
    pw = dd.pos_weight(recs)
    assert all(w >= 1.0 for w in pw)
    assert len(pw) == len(DAMAGE_TYPES)
