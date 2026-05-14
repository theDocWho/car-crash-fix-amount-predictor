"""Image pre-processing pipeline (Stage A: deterministic downscale + quality report).

Stage B (super-resolution via Real-ESRGAN) is deferred to a later checkpoint —
see `progress/phase_3_comparison_and_serving.md`.

Public API:
    quality_report(img)         -> dict of size + sharpness/brightness/contrast
    normalize_for_inference(img, max_long_edge=1600) -> resized PIL.Image
    preprocess(img_bytes, ...)  -> (PIL.Image, dict) ready for the variant pipelines
"""

from .pipeline import normalize_for_inference, preprocess, quality_report

__all__ = ["normalize_for_inference", "preprocess", "quality_report"]
