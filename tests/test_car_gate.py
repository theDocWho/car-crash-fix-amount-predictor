"""Tests for the COCO Mask R-CNN car-presence gate.

The pure decision logic is tested directly; the model path is exercised with an
injected fake detector so no torchvision weights are downloaded.
"""

from __future__ import annotations

from PIL import Image

from ccdp.identification.car_gate import CarGate, GateResult, decide, pad_box


def test_decide_picks_highest_scoring_vehicle():
    res = decide(
        boxes=[[0, 0, 10, 10], [20, 20, 80, 90]],
        labels=[3, 8],          # car, truck
        scores=[0.6, 0.92],
        score_threshold=0.5,
    )
    assert res.has_car is True
    assert res.label == "truck"          # higher score wins
    assert res.score == 0.92
    assert res.n_vehicles == 2
    assert res.box == (20.0, 20.0, 80.0, 90.0)


def test_decide_ignores_non_vehicles():
    res = decide(boxes=[[0, 0, 5, 5]], labels=[1], scores=[0.99])  # person
    assert res.has_car is False
    assert res.box is None
    assert res.n_vehicles == 0


def test_decide_respects_threshold():
    res = decide(boxes=[[0, 0, 5, 5]], labels=[3], scores=[0.3], score_threshold=0.5)
    assert res.has_car is False


def test_decide_empty():
    res = decide(boxes=[], labels=[], scores=[])
    assert res.has_car is False


def test_pad_box_expands_and_clamps():
    # 100x100 box in a 200x200 image, padded 10% → grows by 10px each side.
    x1, y1, x2, y2 = pad_box((50, 50, 150, 150), width=200, height=200, pad_frac=0.1)
    assert (x1, y1, x2, y2) == (40, 40, 160, 160)
    # padding past the border clamps to image bounds
    x1, y1, x2, y2 = pad_box((0, 0, 200, 200), width=200, height=200, pad_frac=0.5)
    assert (x1, y1, x2, y2) == (0, 0, 200, 200)


class _FakeDetector:
    """Mimics a torchvision detection model: list[Tensor] -> list[dict]."""

    def __init__(self, out: dict):
        self._out = out

    def __call__(self, images):
        return [self._out]


def test_gate_detect_with_injected_model():
    gate = CarGate(
        model=_FakeDetector({"boxes": [[10, 10, 100, 120]], "labels": [3], "scores": [0.88]}),
        device="cpu",
    )
    res = gate.detect(Image.new("RGB", (200, 200), "white"))
    assert isinstance(res, GateResult)
    assert res.has_car is True
    assert res.label == "car"


def test_gate_crop_to_car_returns_box_region():
    gate = CarGate(model=_FakeDetector({"boxes": [], "labels": [], "scores": []}), device="cpu")
    img = Image.new("RGB", (200, 200), "white")
    res = GateResult(has_car=True, box=(50, 50, 150, 150))
    crop = gate.crop_to_car(img, res, pad_frac=0.0)
    assert crop.size == (100, 100)


def test_gate_crop_no_car_returns_full_image():
    gate = CarGate(model=_FakeDetector({"boxes": [], "labels": [], "scores": []}), device="cpu")
    img = Image.new("RGB", (200, 200), "white")
    crop = gate.crop_to_car(img, GateResult(has_car=False))
    assert crop.size == (200, 200)
