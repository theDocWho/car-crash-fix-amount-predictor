"""Regression tests for the classifier threshold bug.

Bug: ``VariantAPipeline.predict(threshold=...)`` declared the parameter but
``_forward`` ignored it and hardcoded ``>= 0.5``. This file pins the
threshold-honored behavior so it can't silently regress.

We avoid loading real ResNet50 weights — instead we stub the pipeline's
classifier and transform with the smallest possible callables that produce
known sigmoid probabilities.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from PIL import Image

from ccdp.data.schema import DAMAGE_TYPES
from ccdp.infer import variant_a as variant_a_mod
from ccdp.infer.variant_a import VariantAPipeline


@pytest.fixture(autouse=True)
def _stub_extract_features(monkeypatch):
    """``_forward`` calls ``extract_features`` which expects a real ResNet50.
    Replace it with a no-op so we can swap in a tiny test classifier."""
    monkeypatch.setattr(variant_a_mod, "extract_features",
                        lambda model, x: torch.zeros(1, 2048))


class _ConstantLogits(nn.Module):
    """Returns the same logits regardless of input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, _x):
        return self._logits

    @property
    def fc(self):  # extract_features() expects a `.fc` attribute
        return nn.Identity()


def _stub_pipeline(pipeline: VariantAPipeline, logits_per_class: list[float]) -> None:
    """Replace the heavy classifier with a deterministic 6-class logit emitter."""
    logits = torch.tensor([logits_per_class], dtype=torch.float32)
    pipeline.classifier = _ConstantLogits(logits)
    pipeline.device = torch.device("cpu")
    # Tiny transform — feature extraction in `_forward` will run, but our
    # _ConstantLogits ignores its input so the actual values don't matter.
    pipeline.transform = lambda img: torch.zeros(3, 4, 4)


def _make_pipeline_skipping_init() -> VariantAPipeline:
    """Bypass __init__ — we only want to test the _forward thresholding."""
    return VariantAPipeline.__new__(VariantAPipeline)


def test_threshold_honored_high_filters_all():
    pipe = _make_pipeline_skipping_init()
    # Sigmoid(2.0) ≈ 0.88 — would clear the default 0.5 threshold.
    _stub_pipeline(pipe, [2.0] * len(DAMAGE_TYPES))
    img = Image.new("RGB", (8, 8))

    # With a strict threshold, nothing should be reported.
    types, probs, _ = pipe._forward(img, threshold=0.95)
    assert types == []
    # Probability dict must still report all classes regardless of threshold.
    assert set(probs.keys()) == set(DAMAGE_TYPES)


def test_threshold_default_keeps_above_half():
    pipe = _make_pipeline_skipping_init()
    # logits = [3, -3, 3, -3, 3, -3] → sigmoids alternate ~0.95 / ~0.05
    logits = [3.0 if i % 2 == 0 else -3.0 for i in range(len(DAMAGE_TYPES))]
    _stub_pipeline(pipe, logits)
    img = Image.new("RGB", (8, 8))

    types, probs, _ = pipe._forward(img, threshold=0.5)
    # Even-indexed classes pass; odd ones are filtered out.
    expected = [DAMAGE_TYPES[i] for i in range(len(DAMAGE_TYPES)) if i % 2 == 0]
    assert types == expected
    for i, name in enumerate(DAMAGE_TYPES):
        if i % 2 == 0:
            assert probs[name] > 0.5
        else:
            assert probs[name] < 0.5


def test_threshold_low_admits_all():
    pipe = _make_pipeline_skipping_init()
    # Logits of 0 → sigmoid 0.5 exactly. With threshold=0.1 all should pass.
    _stub_pipeline(pipe, [0.0] * len(DAMAGE_TYPES))
    img = Image.new("RGB", (8, 8))
    types, _, _ = pipe._forward(img, threshold=0.1)
    assert set(types) == set(DAMAGE_TYPES)
