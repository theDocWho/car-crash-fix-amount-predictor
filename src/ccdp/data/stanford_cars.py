"""Stanford Cars 196 loader for the Phase 1.5 make/model/year identifier.

The Kaggle release at ``eduardo4jesus/stanford-cars-dataset`` contains:

- ``cars_train/cars_train/*.jpg``                   — 8,144 training images
- ``cars_test/cars_test/*.jpg``                     — 8,041 test images (labels withheld)
- ``car_devkit/devkit/cars_meta.mat``               — 196 class names ("Make Model Body Year")
- ``car_devkit/devkit/cars_train_annos.mat``        — per-image: bbox + class (1-indexed)
- ``car_devkit/devkit/cars_test_annos.mat``         — per-image: bbox only

Since the official test labels were never released, we split the 8,144 training
images 90/10 (deterministic) into train/val. The kaggle "test" set is unusable
for evaluation and is ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from scipy.io import loadmat

ROOT = Path("data/raw/stanford-cars-dataset")
TRAIN_IMG_DIR = ROOT / "cars_train" / "cars_train"
DEVKIT = ROOT / "car_devkit" / "devkit"

# class-name regex: "Make Model BodyType Year" with body type as a known token
_BODY_TOKENS = (
    "Sedan", "SUV", "Coupe", "Hatchback", "Convertible", "Wagon", "Minivan",
    "Van", "Cab", "Pickup", "Crew Cab", "Extended Cab", "Regular Cab",
    "Type-S", "IPL", "Hybrid",
)
_YEAR_RE = re.compile(r"(\d{4})$")


@dataclass(frozen=True)
class CarClass:
    class_id: int                 # 0-indexed (matches model output)
    raw_name: str                 # e.g. "AM General Hummer SUV 2000"
    make: str
    model: str
    body_type: str
    year: Optional[int]


@dataclass
class CarSample:
    image_path: Path
    class_id: int
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)


def parse_class_name(name: str) -> CarClass:
    """Best-effort parse of a Stanford Cars class string into structured fields."""
    tokens = name.strip().split()
    year = None
    m = _YEAR_RE.search(tokens[-1])
    if m:
        year = int(m.group(1))
        tokens = tokens[:-1]
    body_type = "unknown"
    for i in range(len(tokens) - 1, -1, -1):
        cand = tokens[i]
        for bt in _BODY_TOKENS:
            if cand == bt or " ".join(tokens[i:i + len(bt.split())]) == bt:
                body_type = bt.lower().replace(" ", "_").replace("-", "_")
                tokens = tokens[:i]
                break
        if body_type != "unknown":
            break
    make = tokens[0].lower() if tokens else "unknown"
    model = " ".join(tokens[1:]).lower() if len(tokens) > 1 else "unknown"
    return CarClass(class_id=-1, raw_name=name, make=make, model=model,
                    body_type=body_type, year=year)


def load_classes(devkit: Path = DEVKIT) -> list[CarClass]:
    meta = loadmat(devkit / "cars_meta.mat")
    out = []
    for i, c in enumerate(meta["class_names"][0]):
        parsed = parse_class_name(str(c[0]))
        out.append(CarClass(
            class_id=i, raw_name=parsed.raw_name, make=parsed.make,
            model=parsed.model, body_type=parsed.body_type, year=parsed.year,
        ))
    return out


def _scalar(arr) -> int:
    """Robustly extract a Python int from a possibly nested numpy field."""
    a = arr
    while hasattr(a, "shape") and a.shape:
        a = a[0]
    return int(a)


def load_train_samples(devkit: Path = DEVKIT, img_dir: Path = TRAIN_IMG_DIR) -> list[CarSample]:
    ann = loadmat(devkit / "cars_train_annos.mat")
    out = []
    for a in ann["annotations"][0]:
        fname = str(a["fname"].item()) if hasattr(a["fname"], "item") else str(a["fname"][0])
        cls = _scalar(a["class"]) - 1   # 1-indexed → 0-indexed
        bbox = (_scalar(a["bbox_x1"]), _scalar(a["bbox_y1"]),
                _scalar(a["bbox_x2"]), _scalar(a["bbox_y2"]))
        out.append(CarSample(image_path=img_dir / fname, class_id=cls, bbox=bbox))
    return out


def split_train_val(
    samples: list[CarSample],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[CarSample], list[CarSample]]:
    """Stratified per-class split."""
    import random
    rng = random.Random(seed)
    by_class: dict[int, list[CarSample]] = {}
    for s in samples:
        by_class.setdefault(s.class_id, []).append(s)
    train, val = [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n_val = max(1, int(round(len(cls_samples) * val_fraction)))
        val.extend(cls_samples[:n_val])
        train.extend(cls_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ----- torchvision Dataset ----------------------------------------------
# Module-level class so DataLoader can pickle it across worker processes.


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover — torch is required for this path
    _TorchDataset = object  # type: ignore


class StanfordCarsDataset(_TorchDataset):
    def __init__(self, items: list[CarSample], transform: Optional[Callable] = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image
        s = self.items[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = img.crop(s.bbox)
        if self.transform is not None:
            img = self.transform(img)
        return img, s.class_id


def build_torch_dataset(samples: list[CarSample], transform: Optional[Callable] = None):
    """Return a `torch.utils.data.Dataset` over the given samples."""
    return StanfordCarsDataset(samples, transform)
