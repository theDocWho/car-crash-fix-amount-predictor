"""Stage A pre-processing — pure, deterministic, no ML.

The two responsibilities of this module:

1. **Downscale large uploads** so we don't burn memory and disk on 12-megapixel
   phone photos before the model resizes them anyway. We use LANCZOS resampling
   because edges (dent / crack contours) are exactly what the models depend on.

2. **Score image quality** so the API response can carry a diagnostic for each
   prediction — useful for explaining why a low-confidence result might be due
   to a blurry / dark upload rather than a model failure.

The output is a ``PIL.Image`` ready for the existing :class:`VariantAPipeline`
and :class:`VariantBPipeline` to consume, plus a JSON-serialisable dict that
goes back to the caller in the ``preprocessing`` field of the response.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageStat

DEFAULT_MAX_LONG_EDGE = 1600     # bigger than YOLOv8's 640 input; preserves edge detail


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------


def _sharpness_score(img: Image.Image) -> float:
    """Variance of Laplacian — higher means sharper.

    Standard photographic blur metric. Values around 30 indicate a blurry
    image; values above 150 are well-focused. This is computed on a downscaled
    greyscale copy so the score doesn't blow up linearly with resolution.
    """
    small = img.convert("L").resize((256, 256))
    arr = np.asarray(small, dtype=np.float32)
    # 3x3 Laplacian kernel applied via numpy (avoids OpenCV dep)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = arr.shape
    out = np.zeros_like(arr)
    out[1:-1, 1:-1] = (
        arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
        - 4 * arr[1:-1, 1:-1]
    )
    return float(out.var())


def quality_report(img: Image.Image) -> dict[str, Any]:
    """Return a JSON-friendly dict describing the input image's properties."""
    w, h = img.size
    stat = ImageStat.Stat(img.convert("L"))
    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    sharpness = _sharpness_score(img)
    return {
        "width": w,
        "height": h,
        "megapixels": round(w * h / 1_000_000, 3),
        "long_edge": max(w, h),
        "short_edge": min(w, h),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "is_blurry": sharpness < 60,
        "is_low_light": brightness < 50,
        "is_low_resolution": max(w, h) < 512,
    }


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------


def normalize_for_inference(
    img: Image.Image,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> Image.Image:
    """Downscale ``img`` so its longer edge fits in ``max_long_edge``.

    No-op when the image is already small enough. We never upscale here —
    that's Stage B's job (deferred).
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img
    scale = max_long_edge / long_edge
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, resample=Image.LANCZOS)


# ---------------------------------------------------------------------------
# Top-level entry point used by the API and the CLI
# ---------------------------------------------------------------------------


def preprocess(
    image_or_bytes,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
) -> tuple[Image.Image, dict[str, Any]]:
    """Open + quality-score + downscale.

    Accepts either a path-like, raw bytes, a file-like object, or an already-
    opened ``PIL.Image`` — whichever is most convenient for the caller.

    Returns ``(image_ready_for_model, preprocessing_metadata)``. The metadata
    dict is what the API surface returns in its ``preprocessing`` field so the
    user can see what was done to their upload.
    """
    if isinstance(image_or_bytes, Image.Image):
        img = image_or_bytes
    elif isinstance(image_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_or_bytes))
    else:
        img = Image.open(image_or_bytes)

    qr_before = quality_report(img)
    resized = normalize_for_inference(img, max_long_edge=max_long_edge)
    downscaled = resized.size != img.size

    metadata: dict[str, Any] = {
        "input_size": [img.size[0], img.size[1]],
        "input_quality": qr_before,
        "downscaled": downscaled,
        "max_long_edge": max_long_edge,
        # Stage B placeholder — always false until super-resolution lands
        "super_resolved": False,
    }
    if downscaled:
        metadata["resized_to"] = [resized.size[0], resized.size[1]]
    return resized, metadata
