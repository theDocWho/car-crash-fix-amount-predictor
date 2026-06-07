"""VMMRdb loader — large make/model/year identifier extension (Phase 6).

VMMRdb (Tafazzoli et al. 2017) is ~291k images across ~9,170 make·model·year
classes. We use the **CC0 / public-domain Kaggle mirror**
(``prabashwara/vmmrdb-dataset``), which the Kaggle CLI downloads like the other
datasets — no research agreement needed.

On-disk layout: one folder per class, named ``<make>_<model>_<year>`` (or with
spaces), with the class's images directly inside. The folder location under
``data/raw/vmmrdb-dataset`` is auto-detected, so the exact unzip nesting doesn't
matter. ``top_n`` caps to the N largest classes (by image count) so training
stays tractable — the full 9,170-class long tail is optional.

Public surface mirrors :mod:`ccdp.data.stanford_cars` so the continued-training
loop (:mod:`ccdp.train.continue_identifier`) is dataset-agnostic.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path("data/raw/vmmrdb-dataset")
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

# Module-level cap so the no-arg loader calls from continue_identifier honour it.
TOP_N: Optional[int] = None


def set_top_n(n: Optional[int]) -> None:
    global TOP_N
    TOP_N = n


@dataclass(frozen=True)
class VmmrClass:
    class_id: int
    raw_name: str          # "honda accord 2007" — parseable by stanford parse_class_name
    make: str
    model: str
    year: Optional[int]
    folder: str


@dataclass
class VmmrSample:
    image_path: Path
    class_id: int


def parse_folder(name: str) -> tuple[str, str, Optional[int]]:
    """'honda_accord_2007' / 'Honda Accord 2007' -> (make, model, year)."""
    clean = name.replace("_", " ").strip()
    m = _YEAR_RE.search(clean)
    year = int(m.group(0)) if m else None
    toks = [t for t in clean.split() if not _YEAR_RE.fullmatch(t)]
    make = toks[0].lower() if toks else "unknown"
    model = " ".join(toks[1:]).lower() if len(toks) > 1 else "unknown"
    return make, model, year


def _class_dir_counts(root: Path = ROOT) -> dict[str, int]:
    """Map each class folder (a dir holding images directly) -> image count."""
    counts: dict[str, int] = {}
    if not Path(root).exists():
        return counts
    for dirpath, _dirnames, filenames in os.walk(root):
        n = sum(1 for f in filenames if Path(f).suffix.lower() in _IMG_EXT)
        if n > 0:
            counts[dirpath] = n
    return counts


def _label_space(top_n: Optional[int], root: Path) -> dict[str, int]:
    """{class_folder_path: class_id}. Keeps the top_n folders by image count,
    then assigns ids by sorted folder name (stable/deterministic)."""
    counts = _class_dir_counts(root)
    folders = sorted(counts, key=lambda d: (-counts[d], d))
    if top_n:
        folders = folders[:top_n]
    return {d: i for i, d in enumerate(sorted(folders))}


def load_classes(top_n: Optional[int] = None, root: Optional[Path] = None) -> list[VmmrClass]:
    root = root or ROOT          # resolve at call time so a reassigned ROOT is honoured
    top_n = TOP_N if top_n is None else top_n
    out: list[VmmrClass] = []
    for folder, cid in _label_space(top_n, root).items():
        name = Path(folder).name
        make, model, year = parse_folder(name)
        out.append(VmmrClass(class_id=cid, raw_name=name.replace("_", " "),
                             make=make, model=model, year=year, folder=folder))
    return out


def load_train_samples(top_n: Optional[int] = None, root: Optional[Path] = None) -> list[VmmrSample]:
    root = root or ROOT          # resolve at call time so a reassigned ROOT is honoured
    top_n = TOP_N if top_n is None else top_n
    space = _label_space(top_n, root)
    out: list[VmmrSample] = []
    for folder, cid in space.items():
        for f in os.listdir(folder):
            if Path(f).suffix.lower() in _IMG_EXT:
                out.append(VmmrSample(image_path=Path(folder) / f, class_id=cid))
    return out


def split_train_val(
    samples: list[VmmrSample],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[VmmrSample], list[VmmrSample]]:
    """Stratified per-class split (mirrors stanford_cars / compcars)."""
    import random
    rng = random.Random(seed)
    by_class: dict[int, list[VmmrSample]] = {}
    for s in samples:
        by_class.setdefault(s.class_id, []).append(s)
    train, val = [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n_val = max(1, int(round(len(cls_samples) * val_fraction))) if len(cls_samples) > 1 else 0
        val.extend(cls_samples[:n_val])
        train.extend(cls_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object  # type: ignore


class VmmrDataset(_TorchDataset):
    def __init__(self, items: list[VmmrSample], transform: Optional[Callable] = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image
        s = self.items[idx]
        img = Image.open(s.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.class_id


def build_torch_dataset(samples: list[VmmrSample], transform: Optional[Callable] = None):
    return VmmrDataset(samples, transform)


__all__ = [
    "VmmrClass", "VmmrSample", "VmmrDataset", "set_top_n", "parse_folder",
    "load_classes", "load_train_samples", "split_train_val", "build_torch_dataset",
]
