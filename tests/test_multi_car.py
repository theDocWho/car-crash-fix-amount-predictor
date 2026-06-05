"""Tests for multi-car damage grouping (pure logic + gate detect_all decision)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccdp.identification.car_gate import CarGate, VehicleInstance, nms_vehicles
from ccdp.infer.multi_car import _bbox_overlap, group_damages_by_vehicle


def test_nms_suppresses_subbox_and_superbox():
    big = VehicleInstance("car", 0.9, (0, 0, 100, 100))
    sub = VehicleInstance("car", 0.7, (10, 10, 60, 60))        # inside big -> drop
    scene = VehicleInstance("car", 0.6, (-5, -5, 200, 200))    # contains big -> drop
    sep = VehicleInstance("car", 0.8, (300, 0, 400, 100))      # separate -> keep
    kept = {v.box for v in nms_vehicles([big, sub, scene, sep])}
    assert (0, 0, 100, 100) in kept and (300, 0, 400, 100) in kept
    assert (10, 10, 60, 60) not in kept and (-5, -5, 200, 200) not in kept
    assert len(kept) == 2


@dataclass
class _Dmg:
    name: str
    bbox: tuple
    mask: object = None


def _mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_bbox_overlap_fraction():
    # damage box fully inside vehicle box -> 1.0
    assert _bbox_overlap((10, 10, 20, 20), (0, 0, 100, 100)) == 1.0
    # half inside
    assert abs(_bbox_overlap((0, 0, 20, 10), (10, 0, 100, 100)) - 0.5) < 1e-6
    # disjoint -> 0
    assert _bbox_overlap((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_group_by_bbox_two_cars():
    # car A on the left, car B on the right
    vehicles = [
        VehicleInstance(label="car", score=0.9, box=(0, 0, 100, 100)),
        VehicleInstance(label="car", score=0.8, box=(200, 0, 300, 100)),
    ]
    damages = [
        _Dmg("dent", (10, 10, 30, 30)),       # inside A
        _Dmg("scratch", (220, 20, 250, 40)),  # inside B
        _Dmg("crack", (210, 50, 230, 70)),    # inside B
        _Dmg("glass_shatter", (500, 500, 520, 520)),  # nowhere
    ]
    groups, unassigned = group_damages_by_vehicle(damages, vehicles, min_overlap=0.5)
    assert groups[0] == [0]          # car A -> dent
    assert groups[1] == [1, 2]       # car B -> scratch, crack
    assert unassigned == [3]         # debris


def test_group_by_mask_overlap_when_present():
    H, W = 100, 300
    vehicles = [
        VehicleInstance(label="car", score=0.9, box=(0, 0, 100, 100), mask=_mask(H, W, 0, 100, 0, 100)),
        VehicleInstance(label="truck", score=0.8, box=(150, 0, 300, 100), mask=_mask(H, W, 0, 100, 150, 300)),
    ]
    damages = [
        _Dmg("dent", (10, 10, 30, 30), mask=_mask(H, W, 10, 30, 10, 30)),     # in car 0
        _Dmg("scratch", (160, 10, 200, 40), mask=_mask(H, W, 10, 40, 160, 200)),  # in car 1
    ]
    groups, unassigned = group_damages_by_vehicle(damages, vehicles, min_overlap=0.3)
    assert groups[0] == [0]
    assert groups[1] == [1]
    assert unassigned == []


def test_no_vehicles_all_unassigned():
    damages = [_Dmg("dent", (10, 10, 30, 30))]
    groups, unassigned = group_damages_by_vehicle(damages, [], min_overlap=0.1)
    assert groups == []
    assert unassigned == [0]


class _FakeDetector:
    def __init__(self, out):
        self._out = out

    def __call__(self, images):
        return [self._out]


def test_detect_all_returns_every_vehicle():
    import torch
    from PIL import Image
    # two cars + one person; person must be dropped, both vehicles kept
    out = {
        "boxes": torch.tensor([[0, 0, 50, 50], [60, 0, 120, 50], [200, 0, 210, 10]], dtype=torch.float32),
        "labels": torch.tensor([3, 8, 1]),                  # car, truck, person
        "scores": torch.tensor([0.9, 0.8, 0.99]),
        "masks": torch.zeros((3, 1, 64, 128)),
    }
    gate = CarGate(model=_FakeDetector(out), device="cpu", score_threshold=0.5)
    vehicles = gate.detect_all(Image.new("RGB", (128, 64)))
    assert len(vehicles) == 2
    assert {v.label for v in vehicles} == {"car", "truck"}
