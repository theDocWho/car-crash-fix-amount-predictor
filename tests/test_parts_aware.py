"""Tests for Variant D parts-aware costing: mapping, severity, mask assignment."""

from __future__ import annotations

import numpy as np

from ccdp.infer.parts_map import (
    assign_damage_to_parts,
    severity_from_area,
    to_canonical,
)
from ccdp.infer.seg_inference import SegInstance


def _inst(name, mask, score=0.9, bbox=(0, 0, 1, 1)):
    return SegInstance(name=name, score=score, mask=mask, bbox=bbox)


def _mask(h, w, region):
    m = np.zeros((h, w), bool)
    ys, xs = region
    m[ys[0]:ys[1], xs[0]:xs[1]] = True
    return m


def test_to_canonical_mapping():
    assert to_canonical("front_bumper") == "front_bumper"
    assert to_canonical("back_bumper") == "rear_bumper"
    assert to_canonical("front_left_light") == "headlight"
    assert to_canonical("back_right_light") == "taillight"
    assert to_canonical("tailgate") == "trunk"
    assert to_canonical("object") is None          # unmapped → dropped
    assert to_canonical("back_glass") is None


def test_severity_thresholds():
    assert severity_from_area(0.001) == "minor"
    assert severity_from_area(0.03) == "moderate"
    assert severity_from_area(0.20) == "severe"


def test_assign_damage_overlapping_part():
    part = _mask(100, 100, ((0, 100), (0, 50)))         # left half
    dmg = _mask(100, 100, ((40, 60), (10, 30)))         # inside left half
    pws, assigns = assign_damage_to_parts([_inst("dent", dmg)], [_inst("front_bumper", part)])
    assert pws.get("front_bumper") is not None
    assert assigns[0]["part"] == "front_bumper"
    assert assigns[0]["overlap"] == 1.0


def test_assign_damage_with_no_overlap_is_unassigned():
    # heuristic=False isolates the pure mask-overlap behaviour
    part = _mask(100, 100, ((0, 100), (0, 50)))         # left half
    dmg = _mask(100, 100, ((40, 60), (70, 90)))         # right half → no overlap
    pws, assigns = assign_damage_to_parts(
        [_inst("dent", dmg)], [_inst("front_bumper", part)], heuristic=False)
    assert pws == {}
    assert assigns[0]["part"] is None
    assert assigns[0]["source"] == "none"


def test_heuristic_fallback_when_no_part_overlaps():
    # damage with no overlapping part falls back to bbox-centre → a real part,
    # so it still contributes a cost instead of dropping to $0.
    dmg = _mask(100, 100, ((80, 95), (70, 90)))         # bottom-front area
    bbox = (70, 80, 90, 95)                              # front-bottom → front_bumper
    pws, assigns = assign_damage_to_parts([_inst("dent", dmg, bbox=bbox)], [])
    assert assigns[0]["source"] == "heuristic"
    assert assigns[0]["part"] is not None
    assert pws.get(assigns[0]["part"]) is not None


def test_assign_keeps_worst_severity_per_part():
    part = np.ones((200, 200), bool)                    # whole image = hood
    small = _mask(200, 200, ((0, 5), (0, 5)))           # tiny → minor
    big = _mask(200, 200, ((0, 120), (0, 120)))         # 36% → severe
    pws, _ = assign_damage_to_parts(
        [_inst("scratch", small), _inst("dent", big)], [_inst("hood", part)])
    assert pws["hood"] == "severe"


def test_unmapped_part_class_is_ignored():
    part = np.ones((50, 50), bool)
    dmg = np.ones((50, 50), bool)
    pws, assigns = assign_damage_to_parts(
        [_inst("dent", dmg)], [_inst("object", part)], heuristic=False)
    assert pws == {}                                    # "object" maps to None
    assert assigns[0]["part"] is None
