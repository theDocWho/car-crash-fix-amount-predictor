"""Tests for the gate→identifier orchestration and the identify() ML stage."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ccdp.identification.auto_identify import auto_identify
from ccdp.identification.car_gate import GateResult
from ccdp.identification.car_identifier import identify
from ccdp.identification.ml_identifier import MLIdentification


class _FakeGate:
    def __init__(self, result: GateResult):
        self._result = result

    def detect(self, image):
        return self._result

    def crop_to_car(self, image, result, pad_frac: float = 0.05):
        return image


class _FakeIdentifier:
    def __init__(self, ml: MLIdentification):
        self._ml = ml

    def predict(self, image, topk: int = 3):
        return self._ml


def _ml(make="honda", model="civic", year=2018, conf=0.9) -> MLIdentification:
    return MLIdentification(
        make=make, model=model, year=year, body_type="sedan",
        raw_name=f"{make} {model}", class_id=1, confidence=conf,
        topk=[(f"{make} {model}", conf)],
    )


def test_no_car_short_circuits():
    gate = _FakeGate(GateResult(has_car=False, note="nothing here"))
    res = auto_identify(Image.new("RGB", (64, 64)), gate=gate,
                        identifier=_FakeIdentifier(_ml()))
    assert res.has_car is False
    assert res.identification is None
    assert "No car" in res.note


def test_car_present_fills_identification():
    gate = _FakeGate(GateResult(has_car=True, box=(0, 0, 64, 64), score=0.95, label="car"))
    res = auto_identify(Image.new("RGB", (64, 64)), gate=gate,
                        identifier=_FakeIdentifier(_ml()))
    assert res.has_car is True
    assert res.identification is not None
    assert res.identification.make == "honda"
    assert res.identification.model == "civic"
    assert res.identification.source == "ml"
    assert res.identification.segment == "mid"      # honda → mid segment


def test_low_confidence_falls_back_to_catalog():
    gate = _FakeGate(GateResult(has_car=True, box=(0, 0, 64, 64), score=0.9, label="car"))
    res = auto_identify(
        Image.new("RGB", (64, 64)), gate=gate,
        identifier=_FakeIdentifier(_ml(conf=0.2)), min_confidence=0.5,
    )
    assert res.has_car is True
    assert res.identification.make is None          # too unsure to trust
    assert res.identification.source == "none"


def test_run_gate_false_assumes_car():
    res = auto_identify(
        Image.new("RGB", (64, 64)), run_gate=False,
        identifier=_FakeIdentifier(_ml(make="bmw", model="3 series")),
    )
    assert res.has_car is True
    assert res.identification.make == "bmw"
    assert res.identification.segment == "luxury"


def test_identify_ml_stage_when_filename_blank(tmp_path: Path):
    # filename has no recognisable make → ML stage should fire and win.
    p = tmp_path / "IMG_0001.jpg"
    res = identify(p, use_ml=True, ml_identifier=_FakeIdentifier(_ml(make="audi", year=2019)))
    assert res.make == "audi"
    assert res.source == "ml"
    assert res.year == 2019
    assert "ml" in res.stages_tried


def test_identify_filename_beats_ml(tmp_path: Path):
    # filename already yields a make → ML stage must not override it.
    p = tmp_path / "toyota_camry_2020.jpg"
    res = identify(p, use_ml=True, ml_identifier=_FakeIdentifier(_ml(make="audi")))
    assert res.make == "toyota"
    assert res.source == "filename"
