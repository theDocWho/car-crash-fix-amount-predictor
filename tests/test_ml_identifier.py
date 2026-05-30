"""Tests for the ML make/model identifier wrapper.

The model is injected (a tiny fake returning fixed logits) so no checkpoint is
loaded; the bundled class-name resource is exercised for real.
"""

from __future__ import annotations

import torch
from PIL import Image

from ccdp.identification.ml_identifier import (
    MLIdentifier,
    load_class_names,
)


def test_bundled_class_names_load():
    names = load_class_names()
    assert names is not None
    assert len(names) == 196
    assert names[0] == "AM General Hummer SUV 2000"


class _FakeIdModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, x):
        return self._logits

    def eval(self):
        return self

    def to(self, device):
        return self


def test_predict_maps_argmax_to_make_model_year():
    classes = [
        "Toyota Camry Sedan 2012",
        "Honda Civic Sedan 2012",
        "BMW 3 Series Sedan 2012",
    ]
    # logits favour index 1 (Honda Civic)
    model = _FakeIdModel(torch.tensor([[0.1, 6.0, 0.2]]))
    ident = MLIdentifier(model=model, class_names=classes, device="cpu")
    out = ident.predict(Image.new("RGB", (224, 224), "white"))

    assert out.make == "honda"
    assert out.model == "civic"
    assert out.year == 2012
    assert out.class_id == 1
    assert out.confidence > 0.9          # softmax dominated by index 1
    assert out.topk[0][0] == "Honda Civic Sedan 2012"
    assert len(out.topk) == 3


def test_predict_confidence_is_softmax_prob():
    classes = ["A X Sedan 2010", "B Y Sedan 2010"]
    model = _FakeIdModel(torch.tensor([[0.0, 0.0]]))  # equal logits → 0.5 each
    ident = MLIdentifier(model=model, class_names=classes, device="cpu")
    out = ident.predict(Image.new("RGB", (224, 224)))
    assert abs(out.confidence - 0.5) < 1e-5


def test_raw_name_out_of_range():
    assert MLIdentifier._raw_name(99, ["only", "two"]) == "class_99"
    assert MLIdentifier._raw_name(1, ["only", "two"]) == "two"
