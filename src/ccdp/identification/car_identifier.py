"""Non-ML stages of the car identification pipeline.

The pipeline runs stages in order; each stage may set `make / model / year`
and assigns a confidence. First stage producing make+model wins (highest
precision first). The ML stage (Stanford Cars fine-tune) lives in Phase 1.5
and is wired in at the bottom of `identify()` once available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# --- known-make vocabulary (extend as we encounter more in datasets) ----
KNOWN_MAKES: tuple[str, ...] = (
    "acura", "alfa romeo", "aston martin", "audi", "bentley", "bmw", "buick",
    "cadillac", "chevrolet", "chevy", "chrysler", "citroen", "dodge", "ferrari",
    "fiat", "ford", "genesis", "gmc", "honda", "hyundai", "infiniti", "isuzu",
    "jaguar", "jeep", "kia", "lamborghini", "land rover", "lexus", "lincoln",
    "maserati", "mazda", "mclaren", "mercedes", "mercedes-benz", "mini",
    "mitsubishi", "nissan", "opel", "peugeot", "porsche", "ram", "renault",
    "rolls royce", "rolls-royce", "saab", "saturn", "scion", "skoda", "smart",
    "subaru", "suzuki", "tata", "tesla", "toyota", "volkswagen", "vw", "volvo",
)

# Body-type keywords useful for filename/path heuristics
BODY_TYPE_KEYWORDS: dict[str, str] = {
    "sedan": "sedan", "saloon": "sedan",
    "suv": "suv", "crossover": "crossover",
    "hatchback": "hatchback", "hatch": "hatchback",
    "coupe": "coupe", "convertible": "convertible", "cabrio": "convertible",
    "pickup": "pickup", "truck": "pickup",
    "van": "van", "minivan": "minivan", "wagon": "wagon",
}

LUXURY_MAKES = {
    "audi", "bmw", "mercedes", "mercedes-benz", "porsche", "jaguar", "lexus",
    "infiniti", "acura", "land rover", "tesla", "cadillac", "lincoln", "genesis",
    "maserati", "bentley", "ferrari", "lamborghini", "rolls royce", "rolls-royce",
    "aston martin", "mclaren",
}
ECONOMY_MAKES = {
    "kia", "hyundai", "mitsubishi", "suzuki", "fiat", "nissan", "tata", "scion",
    "smart", "renault", "skoda",
}


@dataclass
class IdentificationResult:
    image_path: Path
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    body_type: str = "unknown"
    segment: str = "unknown"
    color: str = "unknown"
    confidence: float = 0.0
    source: str = "none"           # "filename" | "exif" | "ocr" | "ml" | "user"
    stages_tried: list[str] = field(default_factory=list)

    @property
    def is_exact(self) -> bool:
        return self.make is not None and self.model is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": str(self.image_path),
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "body_type": self.body_type,
            "segment": self.segment,
            "color": self.color,
            "confidence": self.confidence,
            "source": self.source,
            "stages_tried": list(self.stages_tried),
        }


# --- stages -------------------------------------------------------------


_YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def from_filename(path: Path) -> dict[str, Any]:
    """Extract make / model / year / body_type from the path string."""
    # combine parent dirs + filename so folder structure helps too
    haystack = _normalize(" ".join([*[p.name for p in path.parents][:3], path.stem]))
    out: dict[str, Any] = {}

    for make in KNOWN_MAKES:
        if make in haystack:
            out["make"] = make
            # tokens after the make are a model candidate (up to 2 tokens)
            after = haystack.split(make, 1)[1].strip().split()
            if after:
                out["model"] = " ".join(after[:2]).strip()
            break

    m = _YEAR_RE.search(haystack)
    if m:
        out["year"] = int(m.group(1))

    for kw, bt in BODY_TYPE_KEYWORDS.items():
        if kw in haystack:
            out["body_type"] = bt
            break

    return out


def from_exif(path: Path) -> dict[str, Any]:
    """Read EXIF metadata. Rarely useful for car ID but cheap. Returns {} on any failure."""
    try:
        from PIL import Image, ExifTags  # type: ignore
    except ImportError:
        return {}
    try:
        with Image.open(path) as img:
            raw = img.getexif()
        if not raw:
            return {}
        tags = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        out: dict[str, Any] = {}
        # very occasionally, Make/Model fields name the camera, but some
        # vehicle-mfg cameras embed plate/identity strings; treat as low-signal
        if isinstance(tags.get("ImageDescription"), str):
            out["exif_description"] = tags["ImageDescription"]
        if isinstance(tags.get("DateTime"), str):
            out["exif_datetime"] = tags["DateTime"]
        return out
    except Exception:  # noqa: BLE001
        return {}


def from_ocr(path: Path) -> dict[str, Any]:
    """OCR pass on the image to read badges/plates. Best-effort, lazy import.

    Returns make if any KNOWN_MAKES token appears in detected text. Year is
    only kept if the text also references a known make to avoid random plate
    digits being mistaken for years.
    """
    try:
        import easyocr  # type: ignore
    except ImportError:
        return {}
    try:
        reader = _ocr_reader()
        result = reader.readtext(str(path), detail=0, paragraph=True)
    except Exception:  # noqa: BLE001
        return {}
    text = _normalize(" ".join(result))
    if not text:
        return {}
    out: dict[str, Any] = {}
    for make in KNOWN_MAKES:
        if make in text:
            out["make"] = make
            break
    if "make" in out:
        m = _YEAR_RE.search(text)
        if m:
            out["year"] = int(m.group(1))
    return out


_OCR_READER_SINGLETON = None


def _ocr_reader():  # pragma: no cover - heavy import
    global _OCR_READER_SINGLETON
    if _OCR_READER_SINGLETON is None:
        import easyocr  # type: ignore
        _OCR_READER_SINGLETON = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR_READER_SINGLETON


# --- color (cheap heuristic) ---------------------------------------------


_COLOR_BUCKETS: list[tuple[str, tuple[int, int, int]]] = [
    ("black", (20, 20, 20)),
    ("white", (235, 235, 235)),
    ("silver", (190, 190, 190)),
    ("gray", (120, 120, 120)),
    ("red", (190, 40, 40)),
    ("blue", (40, 60, 190)),
    ("green", (40, 140, 60)),
    ("yellow", (220, 200, 40)),
    ("orange", (230, 140, 40)),
    ("brown", (110, 70, 40)),
]


def estimate_color(path: Path) -> str:
    """Coarse color bucket via mean RGB of resized image.

    Returns 'unknown' if PIL isn't available or the image can't be read.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return "unknown"
    try:
        with Image.open(path) as img:
            img = img.convert("RGB").resize((32, 32))
            pixels = list(img.getdata())
    except Exception:  # noqa: BLE001
        return "unknown"
    r = sum(p[0] for p in pixels) / len(pixels)
    g = sum(p[1] for p in pixels) / len(pixels)
    b = sum(p[2] for p in pixels) / len(pixels)
    best, best_dist = "unknown", float("inf")
    for name, (cr, cg, cb) in _COLOR_BUCKETS:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist, best = d, name
    return best


# --- segment classification ---------------------------------------------


def infer_segment(make: Optional[str]) -> str:
    if not make:
        return "unknown"
    m = make.lower()
    if m in LUXURY_MAKES:
        return "luxury"
    if m in ECONOMY_MAKES:
        return "economy"
    return "mid"


# --- top-level identify --------------------------------------------------


def identify(
    image_path: str | Path,
    use_ocr: bool = False,
) -> IdentificationResult:
    """Run all non-ML stages in order and return the merged result.

    OCR is opt-in because it's slow; turn on for a curated subset rather than
    every image.
    """
    p = Path(image_path)
    res = IdentificationResult(image_path=p, color=estimate_color(p))

    # Stage 1: filename / folder hints
    res.stages_tried.append("filename")
    fn = from_filename(p)
    if fn:
        if "make" in fn:
            res.make, res.source, res.confidence = fn["make"], "filename", 0.7
        if "model" in fn and fn["model"]:
            res.model = fn["model"]
        if "year" in fn:
            res.year = fn["year"]
        if "body_type" in fn:
            res.body_type = fn["body_type"]

    # Stage 2: EXIF (very rarely helps but free)
    res.stages_tried.append("exif")
    _ = from_exif(p)  # currently advisory only; could parse description further

    # Stage 3: OCR (opt-in)
    if use_ocr and (res.make is None or res.model is None):
        res.stages_tried.append("ocr")
        ocr = from_ocr(p)
        if "make" in ocr and res.make is None:
            res.make, res.source, res.confidence = ocr["make"], "ocr", 0.8
        if "year" in ocr and res.year is None:
            res.year = ocr["year"]

    # Segment fallback if we have make but no segment yet
    if res.segment == "unknown":
        res.segment = infer_segment(res.make)

    if not res.make:
        res.source = "none"
        res.confidence = 0.0

    return res
