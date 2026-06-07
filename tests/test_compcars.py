"""Tests for the CompCars loader using a synthetic on-disk tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.data import compcars


@pytest.fixture
def fake_compcars(tmp_path: Path, monkeypatch):
    split_dir = tmp_path / "train_test_split" / "classification"
    image_dir = tmp_path / "image"
    split_dir.mkdir(parents=True)
    # two makes, three models, multiple images per model
    rels = [
        "1/10/2012/a.jpg", "1/10/2012/b.jpg", "1/10/2011/c.jpg",   # make 1, model 10
        "1/11/2012/d.jpg", "1/11/2013/e.jpg",                       # make 1, model 11
        "2/20/2014/f.jpg", "2/20/2014/g.jpg",                       # make 2, model 20
    ]
    (split_dir / "train.txt").write_text("\n".join(rels) + "\n")
    monkeypatch.setattr(compcars, "SPLIT_DIR", split_dir)
    monkeypatch.setattr(compcars, "IMAGE_DIR", image_dir)
    monkeypatch.setattr(compcars, "MAKE_MODEL_MAT", tmp_path / "missing.mat")
    return rels


def test_label_space_is_per_model(fake_compcars):
    classes = compcars.load_classes("train")
    assert len(classes) == 3                       # 3 distinct model ids
    model_ids = {c.model_id for c in classes}
    assert model_ids == {"10", "11", "20"}
    # contiguous 0-indexed ids
    assert sorted(c.class_id for c in classes) == [0, 1, 2]


def test_samples_carry_right_class_and_path(fake_compcars):
    samples = compcars.load_train_samples("train")
    assert len(samples) == 7
    # model "10" has 3 images that all share one class id
    by_path = {s.image_path.name: s.class_id for s in samples}
    assert by_path["a.jpg"] == by_path["b.jpg"] == by_path["c.jpg"]
    assert by_path["a.jpg"] != by_path["d.jpg"]    # different model → different class
    assert str(samples[0].image_path).endswith(".jpg")
    assert "image" in str(samples[0].image_path)


def test_stratified_split(fake_compcars):
    samples = compcars.load_train_samples("train")
    train, val = compcars.split_train_val(samples, val_fraction=0.5, seed=0)
    assert len(train) + len(val) == len(samples)
    # every class with >1 sample contributes to val
    assert len(val) >= 2


def test_names_fall_back_without_mat(fake_compcars):
    classes = compcars.load_classes("train")
    # no .mat → synthetic make_/model_ names, still usable
    names = {c.raw_name for c in classes}
    assert any("make_" in n for n in names)
