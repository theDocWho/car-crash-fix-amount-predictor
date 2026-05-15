"""CarDD multi-label dataset for Variant A (whole-image damage-type classifier).

Each Record carries a list of damage_types in `DAMAGE_TYPES`. We build:

- ``encode_labels(types)`` -> length-6 multi-hot vector aligned with DAMAGE_TYPES order.
- ``split_records()``       -> deterministic 80/10/10 train/val/test by image.
- ``build_torch_dataset()`` -> torch Dataset returning (image_tensor, label_tensor).
- ``pos_weight()``          -> per-class inverse-frequency weights for BCE.

CarDD has no make/model/year — those are sampled from iaai distributions at
XGBoost-training time (Phase 2A second half), not here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ccdp.data.schema import DAMAGE_TYPES, Record

LABEL_INDEX: dict[str, int] = {dt: i for i, dt in enumerate(DAMAGE_TYPES)}


def encode_labels(types: Iterable[str]) -> list[float]:
    """Multi-hot length-6 vector in canonical DAMAGE_TYPES order."""
    vec = [0.0] * len(DAMAGE_TYPES)
    for t in types:
        i = LABEL_INDEX.get(t)
        if i is not None:
            vec[i] = 1.0
    return vec


def split_records(
    records: list[Record],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[list[Record], list[Record], list[Record]]:
    """Deterministic per-image train/val/test split."""
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1.0"
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * fractions[0]))
    n_val = int(round(n * fractions[1]))
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def mix_negatives(
    positives: list[Record],
    negatives: list[Record],
    ratio: float,
    seed: int = 42,
) -> list[Record]:
    """Return ``positives`` + a deterministic subsample of ``negatives``.

    ``ratio`` is ``len(returned_negatives) / len(positives)``. So:
      * ``ratio=0`` returns positives unchanged (no-op; legacy CarDD-only flow).
      * ``ratio=1`` adds one negative for every positive (class-balanced wrt 'any damage').
      * ``ratio=2`` adds two negatives per positive.

    If fewer negatives are available than requested we use all of them rather
    than oversample with replacement — duplicating identical images would just
    teach the model to memorise a few photos.

    The output is shuffled deterministically so the train DataLoader doesn't
    see all positives followed by all negatives in batch order.
    """
    if ratio <= 0 or not positives or not negatives:
        return list(positives)
    target_n = int(round(len(positives) * ratio))
    rng = random.Random(seed)
    pool = list(negatives)
    rng.shuffle(pool)
    chosen = pool[:min(target_n, len(pool))]
    mixed = list(positives) + chosen
    rng.shuffle(mixed)
    return mixed


def pos_weight(records: Iterable[Record]) -> list[float]:
    """Inverse-frequency weights for BCEWithLogitsLoss. Length == len(DAMAGE_TYPES)."""
    counts = [0] * len(DAMAGE_TYPES)
    total = 0
    for r in records:
        total += 1
        for t in r.damage_types:
            i = LABEL_INDEX.get(t)
            if i is not None:
                counts[i] += 1
    weights = []
    for c in counts:
        if c == 0:
            weights.append(1.0)
        else:
            # treat negatives = (total - c); weight = negatives / positives
            weights.append(max(1.0, (total - c) / c))
    return weights


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object  # type: ignore


class CarDDMultiLabel(_TorchDataset):
    """Module-level so DataLoader workers can pickle it."""

    def __init__(self, records: list[Record], transform: Optional[Callable] = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image
        r = self.records[idx]
        img = Image.open(r.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(encode_labels(r.damage_types), dtype=torch.float32)
        return img, label


def build_torch_dataset(
    records: list[Record],
    transform: Optional[Callable] = None,
):
    """Return a torch.utils.data.Dataset of (image_tensor, multi-hot label tensor)."""
    return CarDDMultiLabel(records, transform)
