"""ccdp.data — dataset schemas and loaders."""

from .loaders import iter_cardd, iter_comprehensive, iter_iaai
from .schema import (
    CANONICAL_PARTS,
    DAMAGE_TYPES,
    BBox,
    Record,
    infer_part_from_damage,
    map_to_canonical_part,
)

__all__ = [
    "BBox",
    "CANONICAL_PARTS",
    "DAMAGE_TYPES",
    "Record",
    "infer_part_from_damage",
    "iter_cardd",
    "iter_comprehensive",
    "iter_iaai",
    "map_to_canonical_part",
]
