"""Canonical record schema used by every dataset loader.

Each loader yields ``Record`` instances with as many fields filled as the source
dataset supports. Downstream code (reference table, training, identification)
consumes only this schema, so adding a new dataset later is a one-file change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal, Optional

# Canonical 15-part taxonomy used by the cost catalog as the pricing dimension.
# NOTE: No public training corpus labels parts at this granularity. Parts are
# *inferred* at inference time from (damage_type, damage_location, bbox center)
# via heuristic rules. The catalog stays parts-keyed because pricing is most
# naturally expressed per-part; mapping rules live in `infer_part_from_damage`.
CANONICAL_PARTS: tuple[str, ...] = (
    "front_bumper",
    "rear_bumper",
    "hood",
    "front_door",
    "rear_door",
    "front_fender",
    "rear_quarter_panel",
    "headlight",
    "taillight",
    "windshield",
    "side_mirror",
    "roof",
    "trunk",
    "wheel",
    "grille",
)

# Canonical 6 damage TYPES from CarDD (Wang et al. 2023). These ARE trainable
# labels — the YOLOv8 detector and ResNet50 multi-label classifier output these.
DAMAGE_TYPES: tuple[str, ...] = (
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat",
)

# Canonical 2-axis damage LOCATION from samwash94 dataset.
# (location ∈ {front, rear}) × (condition ∈ {normal, crushed, breakage}).
DamageLocation = Literal["front", "rear", "unknown"]
DamageCondition = Literal["normal", "crushed", "breakage", "unknown"]

Severity = Literal["minor", "moderate", "severe"]
Segment = Literal["economy", "mid", "luxury", "unknown"]
BodyType = Literal[
    "sedan", "suv", "hatchback", "coupe", "convertible", "pickup", "van",
    "wagon", "minivan", "crossover", "truck", "unknown",
]


@dataclass
class BBox:
    """YOLO-style normalized bbox (0..1) with damage-type label and optional severity.

    The `damage_type` field carries one of `DAMAGE_TYPES` (the trainable label
    from CarDD). `part` is *derived* downstream — leave it None at load time.
    """
    damage_type: str
    x_center: float
    y_center: float
    width: float
    height: float
    severity: Optional[Severity] = None
    confidence: Optional[float] = None
    part: Optional[str] = None         # inferred at inference, not loader time


@dataclass
class Record:
    """One image's worth of damage + car-identity + cost information."""

    image_path: Path
    dataset: str                                # source dataset slug

    # damage labels — any of these may be empty depending on source
    damage_types: list[str] = field(default_factory=list)        # multi-label, from CarDD
    bboxes: list[BBox] = field(default_factory=list)             # CarDD COCO / converted YOLO
    damage_location: str = "unknown"                             # 'front' | 'rear' | 'unknown' (samwash94)
    damage_condition: str = "unknown"                            # 'normal' | 'crushed' | 'breakage' | 'unknown'
    # inferred at scoring time from (damage_types, bboxes, damage_location)
    parts: list[str] = field(default_factory=list)
    parts_severity: dict[str, Severity] = field(default_factory=dict)

    # car identity — may be partially or fully missing
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    body_type: BodyType = "unknown"
    segment: Segment = "unknown"

    # cost target — recorded in source currency for traceability
    cost: Optional[float] = None
    cost_currency: Optional[str] = None         # "USD" or "INR"
    cost_usd: Optional[float] = None            # normalized via FX (snapshot recorded)
    cost_source: Optional[str] = None           # "ganeshsura", "iaai", "synthetic@<catalog_id>"
    fx_snapshot: dict[str, Any] = field(default_factory=dict)

    # identification provenance (filled by identifier pipeline, not loader)
    identification_tier: Optional[Literal["exact", "nearest_class", "none"]] = None
    identification_source: Optional[str] = None  # "filename" | "exif" | "ocr" | "ml" | "user"
    identification_confidence: Optional[float] = None

    # free-form extras (e.g. raw labels not yet mapped to canonical parts)
    extras: dict[str, Any] = field(default_factory=dict)

    # --- convenience ---------------------------------------------------------

    @property
    def image_id(self) -> str:
        """Stable id derived from the path. Used as PK across processed tables."""
        return f"{self.dataset}/{self.image_path.name}"

    @property
    def is_identified(self) -> bool:
        return self.make is not None and self.model is not None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        d["bboxes"] = [asdict(b) for b in self.bboxes]
        return d


def infer_part_from_damage(
    damage_type: str,
    bbox_center: Optional[tuple[float, float]] = None,
    damage_location: str = "unknown",
) -> Optional[str]:
    """Heuristic mapping from a (damage_type, bbox-center, location) tuple to a
    canonical part. Used at inference time to bridge trainable damage labels
    with the parts-keyed cost catalog.

    `bbox_center` is normalized (x, y) in [0, 1] from image top-left.
    `damage_location` is the auxiliary head's prediction ('front' | 'rear' | 'unknown').

    Rules are intentionally conservative — return None when ambiguous so the
    caller can fall through to a generic "front_bumper" / "rear_bumper" tier-3
    default. Mapping rationale documented in [CITATIONS.md] discussion.
    """
    dt = damage_type.lower().replace(" ", "_")

    # Type-only rules (independent of position)
    if dt == "tire_flat":
        return "wheel"
    if dt == "glass_shatter":
        return "windshield"
    if dt == "lamp_broken":
        if damage_location == "rear" or (bbox_center and bbox_center[1] > 0.55):
            return "taillight"
        return "headlight"

    # For dent/scratch/crack we need a position guess.
    if bbox_center is None and damage_location == "unknown":
        return None

    is_front = damage_location == "front" or (bbox_center and bbox_center[0] > 0.55)
    is_rear = damage_location == "rear" or (bbox_center and bbox_center[0] < 0.45)

    if dt in {"dent", "scratch", "crack"}:
        if bbox_center is None:
            return "front_bumper" if is_front else "rear_bumper" if is_rear else None
        x, y = bbox_center
        # very rough panel grid
        if y > 0.7:  # bottom band → bumper / wheel area
            return "front_bumper" if is_front else "rear_bumper"
        if y < 0.35:  # top band → hood / roof / trunk
            if is_front:
                return "hood"
            if is_rear:
                return "trunk"
            return "roof"
        # mid band → door / fender / quarter panel
        if is_front:
            return "front_door" if 0.3 < y < 0.65 else "front_fender"
        if is_rear:
            return "rear_door" if 0.3 < y < 0.65 else "rear_quarter_panel"
        return None
    return None


def map_to_canonical_part(raw_label: str) -> Optional[str]:
    """Best-effort mapping from free-text damage label to canonical part name.

    Returns None if no confident mapping exists; caller decides whether to keep
    in extras or drop. Kept intentionally small + auditable; expand in the EDA
    notebook as new label vocabularies are discovered.
    """
    s = raw_label.strip().lower().replace("-", " ").replace("_", " ")
    rules: list[tuple[tuple[str, ...], str]] = [
        (("front bumper", "front-bumper", "bumper front"), "front_bumper"),
        (("rear bumper", "back bumper", "bumper rear"), "rear_bumper"),
        (("hood", "bonnet"), "hood"),
        (("front door",), "front_door"),
        (("rear door", "back door"), "rear_door"),
        (("front fender", "front-fender", "fender front"), "front_fender"),
        (("rear quarter", "quarter panel", "quarter-panel"), "rear_quarter_panel"),
        (("headlight", "head light", "head lamp", "headlamp"), "headlight"),
        (("taillight", "tail light", "tail lamp", "taillamp", "rear light"), "taillight"),
        (("windshield", "windscreen", "front glass"), "windshield"),
        (("side mirror", "wing mirror", "rear view mirror"), "side_mirror"),
        (("roof",), "roof"),
        (("trunk", "boot", "tailgate", "rear gate"), "trunk"),
        (("wheel", "rim", "tire"), "wheel"),
        (("grille", "grill"), "grille"),
    ]
    for needles, canonical in rules:
        if any(n in s for n in needles):
            return canonical
    return None
