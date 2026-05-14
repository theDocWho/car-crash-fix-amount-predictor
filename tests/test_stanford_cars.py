"""Tests for the Stanford Cars loader."""

from __future__ import annotations

import pytest

from ccdp.data import stanford_cars as sc


@pytest.mark.skipif(not sc.DEVKIT.exists(), reason="Stanford Cars devkit not on disk")
def test_load_classes():
    classes = sc.load_classes()
    assert len(classes) == 196
    assert classes[0].class_id == 0
    # parse_class_name should extract a year for most entries
    with_year = sum(1 for c in classes if c.year)
    assert with_year > 180   # >90% have a parseable year


@pytest.mark.skipif(not sc.DEVKIT.exists(), reason="Stanford Cars devkit not on disk")
def test_load_train_samples():
    samples = sc.load_train_samples()
    assert len(samples) == 8144
    s = samples[0]
    assert s.class_id >= 0
    assert s.bbox[2] > s.bbox[0] and s.bbox[3] > s.bbox[1]
    assert s.image_path.suffix == ".jpg"


@pytest.mark.skipif(not sc.DEVKIT.exists(), reason="Stanford Cars devkit not on disk")
def test_split_train_val_stratified():
    samples = sc.load_train_samples()
    train, val = sc.split_train_val(samples, val_fraction=0.1, seed=42)
    assert len(train) + len(val) == len(samples)
    # every class must appear in val
    val_classes = {s.class_id for s in val}
    assert val_classes == set(range(196))


def test_parse_class_name_known():
    p = sc.parse_class_name("AM General Hummer SUV 2000")
    assert p.year == 2000
    assert p.body_type == "suv"
    assert "hummer" in p.model
