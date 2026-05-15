"""Draw YOLO detector boxes + labels onto a PIL image.

Pillow-only (no OpenCV, no matplotlib) so this stays cheap to import inside
the FastAPI / Gradio request path. Each damage class gets a fixed color so
the overlay is visually consistent across calls.

Boxes arrive as ``xywh_norm`` — center-x, center-y, width, height normalised
to ``[0, 1]`` — matching :class:`ccdp.infer.variant_b.DetectedBox`. We convert
back to absolute pixel corners here.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from ccdp.data.schema import DAMAGE_TYPES

# Fixed palette — one color per damage class so the same damage looks the
# same across images. Picked for contrast on typical car-paint colors.
_PALETTE: dict[str, tuple[int, int, int]] = {
    "dent":          (255,  99,  71),   # tomato
    "scratch":       (255, 215,   0),   # gold
    "crack":         (138,  43, 226),   # blueviolet
    "glass_shatter": ( 30, 144, 255),   # dodgerblue
    "lamp_broken":   (255, 140,   0),   # darkorange
    "tire_flat":     ( 50, 205,  50),   # limegreen
}
_DEFAULT_COLOR = (200, 200, 200)


def _color_for(damage_type: str) -> tuple[int, int, int]:
    return _PALETTE.get(damage_type, _DEFAULT_COLOR)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort font load — falls back to PIL's bitmap default if no TTF available."""
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _denormalize(xywh_norm: Sequence[float], w: int, h: int) -> tuple[int, int, int, int]:
    """Convert (xc, yc, bw, bh) in [0,1] → (x1, y1, x2, y2) pixel ints."""
    xc, yc, bw, bh = xywh_norm
    x1 = int(round((xc - bw / 2) * w))
    y1 = int(round((yc - bh / 2) * h))
    x2 = int(round((xc + bw / 2) * w))
    y2 = int(round((yc + bh / 2) * h))
    # Clamp to image bounds — degenerate boxes still render as a 1px line.
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return x1, y1, x2, y2


def annotate_detections(
    image: Image.Image,
    detections: Iterable,
    *,
    line_width: int = 4,
    font_size: int = 16,
    show_confidence: bool = True,
) -> Image.Image:
    """Return a copy of ``image`` with detection boxes + labels drawn on top.

    ``detections`` is any iterable of objects exposing ``damage_type``,
    ``confidence``, and ``xywh_norm`` — typically :class:`DetectedBox` from
    Variant B, but any duck-typed equivalent works (handy for tests).
    """
    out = image.convert("RGB").copy()
    w, h = out.size
    draw = ImageDraw.Draw(out, mode="RGBA")
    font = _load_font(font_size)

    for det in detections:
        damage_type = getattr(det, "damage_type", "unknown")
        confidence = float(getattr(det, "confidence", 0.0))
        xywh = getattr(det, "xywh_norm", (0.5, 0.5, 0.0, 0.0))
        color = _color_for(damage_type)

        x1, y1, x2, y2 = _denormalize(xywh, w, h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{damage_type} {confidence:.0%}" if show_confidence else damage_type
        # Measure text so the chip behind it sits flush with the top-left corner.
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:  # very old PIL
            tw, th = font.getsize(label)
        pad = 4
        chip_y2 = y1
        chip_y1 = max(0, y1 - th - 2 * pad)
        draw.rectangle([x1, chip_y1, x1 + tw + 2 * pad, chip_y2],
                       fill=(*color, 220))
        draw.text((x1 + pad, chip_y1 + pad), label, fill=(0, 0, 0), font=font)
    return out


def annotate_no_detections(
    image: Image.Image,
    message: str = "No damage detected by YOLOv8 detector",
) -> Image.Image:
    """Stamp a translucent banner across the image when the detector finds nothing.

    Without this, an empty-detection image is visually indistinguishable from
    a successful detection that drew no boxes — which is exactly the silent
    failure mode that bit us on real-world OOD photos.
    """
    # Draw on an RGBA overlay then alpha-composite back so the translucent
    # banner actually blends with the underlying image (drawing RGBA mode
    # straight onto an RGB canvas drops the alpha channel and renders nothing).
    base = image.convert("RGBA").copy()
    w, h = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(max(14, w // 40))
    try:
        tb = draw.textbbox((0, 0), message, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except AttributeError:
        tw, th = font.getsize(message)
    pad = 16
    bx1 = max(0, (w - tw) // 2 - pad)
    by1 = max(0, h // 2 - th // 2 - pad)
    bx2 = min(w, (w + tw) // 2 + pad)
    by2 = min(h, h // 2 + th // 2 + pad)
    draw.rectangle([bx1, by1, bx2, by2], fill=(0, 0, 0, 200))
    draw.text(((w - tw) // 2, (h - th) // 2 - pad // 2), message,
              fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, overlay).convert("RGB")


def annotate_prediction(image: Image.Image, prediction) -> Image.Image:
    """Convenience wrapper: pull ``.detections`` off a Variant B prediction.

    Works with either the :class:`PredictionB` dataclass or its ``.to_dict()``
    form — the Gradio + FastAPI layers can pass whichever they have.
    Returns the no-detections banner image when the prediction is empty so
    callers don't have to special-case it.
    """
    detections = getattr(prediction, "detections", None)
    if detections is None and isinstance(prediction, dict):
        detections = prediction.get("detections", [])
    if not detections:
        return annotate_no_detections(image)

    # If we got dicts (from to_dict()), wrap them so attribute access works.
    if detections and isinstance(detections[0], dict):
        class _D:
            def __init__(self, d):
                self.damage_type = d.get("damage_type", "unknown")
                self.confidence = d.get("confidence", 0.0)
                self.xywh_norm = d.get("xywh_norm", (0.5, 0.5, 0.0, 0.0))
        detections = [_D(d) for d in detections]
    return annotate_detections(image, detections)


__all__ = [
    "annotate_detections",
    "annotate_no_detections",
    "annotate_prediction",
    "DAMAGE_TYPES",
]
