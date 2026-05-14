"""Catalog drift calibrator.

A trained XGBoost regressor learns dollar cost against a specific *training-time*
catalog. When the active catalog is updated later (parts prices change), we don't
want to retrain immediately. Instead we scale model output by:

    scale = active_catalog.median_cost() / training_catalog.median_cost()

This is a first-order correction. It assumes catalog updates shift price *levels*
relatively uniformly. For non-uniform shifts (e.g. only EV parts jumped), retrain.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog, load


@dataclass
class Calibrator:
    training_catalog_id: str
    training_median: float

    @classmethod
    def from_catalog(cls, training_catalog: Catalog) -> "Calibrator":
        return cls(
            training_catalog_id=training_catalog.catalog_id,
            training_median=training_catalog.median_cost(),
        )

    def scale(self, predicted_cost: float, active_catalog: Catalog) -> float:
        if self.training_median <= 0:
            return predicted_cost
        factor = active_catalog.median_cost() / self.training_median
        return predicted_cost * factor

    def scale_factor(self, active_catalog: Catalog) -> float:
        if self.training_median <= 0:
            return 1.0
        return active_catalog.median_cost() / self.training_median


def load_training_catalog(calibrator: Calibrator) -> Catalog:
    return load(calibrator.training_catalog_id)
