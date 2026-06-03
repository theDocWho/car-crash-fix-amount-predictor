"""Map the parts segmenter's 23 carparts-seg classes onto the cost catalog's
canonical parts, and assign each detected damage to the part it overlaps.

This replaces the heuristic :func:`ccdp.data.schema.infer_part_from_damage`
(bbox-centre + body-type guess) with a learned, mask-overlap assignment:
"this dent's mask sits mostly inside the front_bumper's mask → front_bumper".
"""

from __future__ import annotations

from typing import Optional

# carparts-seg (Ultralytics) class name -> canonical catalog part.
# back_glass / object have no catalog analogue and are intentionally dropped.
CARPARTS_TO_CANONICAL: dict[str, str] = {
    "front_bumper": "front_bumper",
    "back_bumper": "rear_bumper",
    "hood": "hood",
    "front_door": "front_door",
    "front_left_door": "front_door",
    "front_right_door": "front_door",
    "back_door": "rear_door",
    "back_left_door": "rear_door",
    "back_right_door": "rear_door",
    "front_light": "headlight",
    "front_left_light": "headlight",
    "front_right_light": "headlight",
    "back_light": "taillight",
    "back_left_light": "taillight",
    "back_right_light": "taillight",
    "front_glass": "windshield",
    "left_mirror": "side_mirror",
    "right_mirror": "side_mirror",
    "trunk": "trunk",
    "tailgate": "trunk",
    "wheel": "wheel",
}

_SEVERITY_ORDER = {"minor": 0, "moderate": 1, "severe": 2}


def to_canonical(carparts_name: str) -> Optional[str]:
    """carparts-seg class name -> canonical catalog part (or None if not mapped)."""
    return CARPARTS_TO_CANONICAL.get(carparts_name)


def severity_from_area(area_frac: float) -> str:
    """Map a damage mask's image-area fraction to a catalog severity bucket."""
    if area_frac >= 0.06:
        return "severe"
    if area_frac >= 0.015:
        return "moderate"
    return "minor"


def _worse(a: Optional[str], b: str) -> str:
    if a is None:
        return b
    return a if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b] else b


def _overlap_fraction(damage_mask, part_mask) -> float:
    """Fraction of the damage mask that falls inside the part mask."""
    import numpy as np
    dm = np.count_nonzero(damage_mask)
    if dm == 0:
        return 0.0
    return float(np.count_nonzero(np.logical_and(damage_mask, part_mask))) / float(dm)


def _heuristic_part(damage) -> Optional[str]:
    """Fallback: bbox-centre → canonical part (the old infer_part_from_damage)."""
    from ccdp.data.schema import infer_part_from_damage
    h, w = damage.mask.shape
    x1, y1, x2, y2 = damage.bbox
    center = ((x1 + x2) / 2 / max(w, 1), (y1 + y2) / 2 / max(h, 1))
    return infer_part_from_damage(damage.name, center)


def assign_damage_to_parts(
    damages,
    parts,
    min_overlap: float = 0.15,
    heuristic: bool = True,
) -> tuple[dict[str, str], list[dict]]:
    """Assign each damage instance to the canonical part it most overlaps.

    Mask-overlap with the parts model wins; when no part mask overlaps a damage
    (``heuristic=True``, the default) we fall back to the bbox-centre rule so a
    detected damage still contributes a cost instead of dropping to $0.

    Returns ``(parts_with_severity, assignments)`` — the ``{part: worst_severity}``
    map the catalog estimator wants, plus a per-damage audit list (with a
    ``source`` of ``mask`` / ``heuristic`` / ``none``).
    """
    parts_with_severity: dict[str, str] = {}
    assignments: list[dict] = []

    for d in damages:
        best_part: Optional[str] = None
        best_ov = 0.0
        for p in parts:
            canon = to_canonical(p.name)
            if canon is None:
                continue
            ov = _overlap_fraction(d.mask, p.mask)
            if ov > best_ov:
                best_ov, best_part = ov, canon

        sev = severity_from_area(d.area_frac)
        if best_part is not None and best_ov >= min_overlap:
            part, source = best_part, "mask"
        elif heuristic:
            part = _heuristic_part(d)
            source = "heuristic" if part else "none"
        else:
            part, source = None, "none"

        if part is not None:
            parts_with_severity[part] = _worse(parts_with_severity.get(part), sev)
        assignments.append({
            "damage_type": d.name, "part": part, "severity": sev, "source": source,
            "overlap": round(best_ov, 3), "area_frac": round(d.area_frac, 4),
            "confidence": round(d.score, 3),
        })
    return parts_with_severity, assignments


__all__ = [
    "CARPARTS_TO_CANONICAL",
    "to_canonical",
    "severity_from_area",
    "assign_damage_to_parts",
]
