"""Visualization helpers — render model outputs onto images."""

from ccdp.viz.overlay import (
    annotate_car_box,
    annotate_detections,
    annotate_no_detections,
    annotate_prediction,
)

__all__ = [
    "annotate_car_box",
    "annotate_detections",
    "annotate_no_detections",
    "annotate_prediction",
]
