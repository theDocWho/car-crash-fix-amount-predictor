"""Tests for the bounding-box overlay renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ccdp.viz.overlay import _color_for, _denormalize, annotate_detections, annotate_prediction


@dataclass
class _Fake:
    damage_type: str
    confidence: float
    xywh_norm: tuple[float, float, float, float]


def _blank(w=200, h=100, color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_denormalize_centered_full_image():
    assert _denormalize((0.5, 0.5, 1.0, 1.0), 200, 100) == (0, 0, 200, 100)


def test_denormalize_clamps_out_of_bounds():
    # Box that extends past the right/bottom edges should clamp.
    x1, y1, x2, y2 = _denormalize((0.9, 0.9, 0.4, 0.4), 100, 100)
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100


def test_color_for_known_damage_type_stable():
    # Same damage type always maps to the same color (visual consistency).
    assert _color_for("dent") == _color_for("dent")
    assert _color_for("dent") != _color_for("scratch")


def test_annotate_detections_does_not_mutate_input():
    img = _blank()
    arr_before = np.asarray(img).copy()
    annotate_detections(img, [_Fake("dent", 0.9, (0.5, 0.5, 0.2, 0.2))])
    arr_after = np.asarray(img)
    assert np.array_equal(arr_before, arr_after)


def test_annotate_detections_paints_pixels():
    img = _blank()
    out = annotate_detections(img, [_Fake("dent", 0.9, (0.5, 0.5, 0.4, 0.4))])
    # Output is a different image with non-white pixels (the box was drawn).
    out_arr = np.asarray(out)
    in_arr = np.asarray(img)
    assert out.size == img.size
    assert not np.array_equal(out_arr, in_arr)
    # At least some pixels should be the 'dent' tomato color.
    r, g, b = _color_for("dent")
    matches = (out_arr[..., 0] == r) & (out_arr[..., 1] == g) & (out_arr[..., 2] == b)
    assert matches.any()


def test_annotate_detections_empty_returns_copy():
    img = _blank()
    out = annotate_detections(img, [])
    assert out.size == img.size
    assert out is not img


def test_annotate_prediction_accepts_dict_form():
    img = _blank()
    pred_dict = {
        "detections": [
            {"damage_type": "scratch", "confidence": 0.7, "xywh_norm": (0.3, 0.4, 0.2, 0.2)},
        ],
    }
    out = annotate_prediction(img, pred_dict)
    assert out.size == img.size


def test_annotate_prediction_no_detections_safe():
    out = annotate_prediction(_blank(), {"detections": []})
    assert out.size == (200, 100)
