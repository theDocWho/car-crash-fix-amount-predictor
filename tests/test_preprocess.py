"""Tests for the Stage A pre-processing pipeline."""

from __future__ import annotations

import io

from PIL import Image

from ccdp.preprocess import normalize_for_inference, preprocess, quality_report


def _mk_image(w: int, h: int, color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


def test_quality_report_basic_fields():
    img = _mk_image(800, 600)
    qr = quality_report(img)
    assert qr["width"] == 800 and qr["height"] == 600
    assert qr["long_edge"] == 800
    assert qr["short_edge"] == 600
    assert qr["megapixels"] == 0.48
    assert "sharpness" in qr and qr["sharpness"] >= 0
    assert "brightness" in qr and 0 <= qr["brightness"] <= 255


def test_quality_report_flags_low_resolution():
    qr = quality_report(_mk_image(400, 300))
    assert qr["is_low_resolution"] is True


def test_normalize_no_op_for_small_image():
    img = _mk_image(800, 600)
    out = normalize_for_inference(img, max_long_edge=1600)
    assert out.size == (800, 600)


def test_normalize_downscales_large_image_preserving_aspect_ratio():
    img = _mk_image(3200, 2400)
    out = normalize_for_inference(img, max_long_edge=1600)
    assert max(out.size) == 1600
    # aspect-ratio preserved within rounding
    ratio_in = 3200 / 2400
    ratio_out = out.size[0] / out.size[1]
    assert abs(ratio_in - ratio_out) < 1e-3


def test_preprocess_accepts_bytes_pillow_and_path(tmp_path):
    img = _mk_image(2000, 1500)

    # Pillow path
    out, meta = preprocess(img)
    assert meta["downscaled"] is True
    assert meta["input_size"] == [2000, 1500]
    assert meta["resized_to"][0] <= 1600
    assert meta["super_resolved"] is False

    # bytes path
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out2, meta2 = preprocess(buf.getvalue())
    assert meta2["downscaled"] is True

    # path path
    p = tmp_path / "img.png"
    img.save(p)
    out3, meta3 = preprocess(p)
    assert meta3["downscaled"] is True


def test_preprocess_skips_downscale_for_already_small_image():
    img = _mk_image(800, 600)
    out, meta = preprocess(img)
    assert meta["downscaled"] is False
    assert out.size == (800, 600)
