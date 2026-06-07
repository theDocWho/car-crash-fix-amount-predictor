"""Tests for the VMMRdb loader using a synthetic folder tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.data import vmmrdb


def _make_tree(root: Path, spec: dict[str, int]):
    # spec: folder_name -> number of images. Nest under an extra dir to mimic the
    # real unzip layout (loader must auto-detect the class folders).
    base = root / "vmmrdb-dataset" / "VMMRdb"
    for folder, n in spec.items():
        d = base / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"img{i}.jpg").write_bytes(b"x")
    return base


@pytest.fixture(autouse=True)
def _reset_topn():
    vmmrdb.set_top_n(None)
    yield
    vmmrdb.set_top_n(None)


def test_parse_folder():
    assert vmmrdb.parse_folder("honda_accord_2007") == ("honda", "accord", 2007)
    assert vmmrdb.parse_folder("Mercedes Benz C Class 2014") == ("mercedes", "benz c class", 2014)
    make, model, year = vmmrdb.parse_folder("weird_no_year")
    assert make == "weird" and year is None


def test_top_n_keeps_largest_classes(tmp_path: Path):
    base = _make_tree(tmp_path, {"honda_accord_2007": 5, "toyota_camry_2010": 3, "bmw_3_series_2012": 1})
    classes = vmmrdb.load_classes(top_n=2, root=base)
    folders = {c.folder.split("/")[-1] for c in classes}
    assert len(classes) == 2
    assert "honda_accord_2007" in folders and "toyota_camry_2010" in folders
    assert "bmw_3_series_2012" not in folders          # smallest dropped


def test_samples_consistent_with_classes(tmp_path: Path):
    base = _make_tree(tmp_path, {"honda_accord_2007": 2, "toyota_camry_2010": 2})
    classes = vmmrdb.load_classes(root=base)
    samples = vmmrdb.load_train_samples(root=base)
    assert len(samples) == 4
    valid_ids = {c.class_id for c in classes}
    assert all(s.class_id in valid_ids for s in samples)
    # parsed fields populated
    honda = next(c for c in classes if c.make == "honda")
    assert honda.model == "accord" and honda.year == 2007 and honda.raw_name == "honda accord 2007"


def test_module_top_n_used_by_noarg_calls(tmp_path: Path):
    base = _make_tree(tmp_path, {"a_x_2001": 3, "b_y_2002": 2, "c_z_2003": 1})
    vmmrdb.set_top_n(1)
    # continue_identifier calls these with no top_n -> must honour module TOP_N
    assert len(vmmrdb.load_classes(root=base)) == 1
    assert {s.class_id for s in vmmrdb.load_train_samples(root=base)} == {0}


def test_stratified_split(tmp_path: Path):
    base = _make_tree(tmp_path, {"a_x_2001": 4, "b_y_2002": 4})
    samples = vmmrdb.load_train_samples(root=base)
    train, val = vmmrdb.split_train_val(samples, val_fraction=0.5, seed=0)
    assert len(train) + len(val) == len(samples)
    assert len(val) >= 2
