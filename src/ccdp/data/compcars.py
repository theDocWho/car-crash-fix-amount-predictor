"""CompCars loader for Phase 6 make/model identifier extension.

CompCars (Yang et al. 2015, free for research) is much larger than Stanford Cars
— ~136k web images across 1,716 models — so it broadens make/model coverage when
we *continue-train* the existing identifier on it.

Expected on-disk layout (the standard release, under ``data/raw/compcars/data``)::

    image/<make_id>/<model_id>/<year>/<img>.jpg
    label/<make_id>/<model_id>/<year>/<img>.txt        # viewpoint + bbox (optional)
    misc/make_model_name.mat                            # make_names, model_names
    train_test_split/classification/{train,test}.txt    # rel image paths

Classification label = ``model_id`` (the middle directory). We map the set of
model_ids present in the split files to contiguous 0-indexed class ids. Make/model
human names come from ``make_model_name.mat`` when available; otherwise we fall
back to the numeric ids so training still works.

The public surface mirrors :mod:`ccdp.data.stanford_cars`
(``load_classes`` / ``load_train_samples`` / ``split_train_val`` /
``build_torch_dataset``) so the continued-training loop is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path("data/raw/compcars/data")
IMAGE_DIR = ROOT / "image"
SPLIT_DIR = ROOT / "train_test_split" / "classification"
MAKE_MODEL_MAT = ROOT / "misc" / "make_model_name.mat"


@dataclass(frozen=True)
class CompCarsClass:
    class_id: int            # 0-indexed (matches model output)
    raw_name: str            # "<make> <model>"
    make: str
    model: str
    body_type: str
    year: Optional[int]
    model_id: str            # original CompCars model directory id


@dataclass
class CompCarsSample:
    image_path: Path
    class_id: int


def _read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _model_id_of(rel_path: str) -> str:
    # rel path looks like "<make_id>/<model_id>/<year>/<img>.jpg"
    parts = Path(rel_path).parts
    return parts[1] if len(parts) >= 2 else parts[0]


def _load_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (make_id->make_name, model_id->model_name) from the .mat, if present."""
    if not MAKE_MODEL_MAT.exists():
        return {}, {}
    try:
        from scipy.io import loadmat
        m = loadmat(MAKE_MODEL_MAT)
        make_names = [str(x[0][0]) if len(x[0]) else "" for x in m["make_names"]]
        model_names = [str(x[0][0]) if len(x[0]) else "" for x in m["model_names"]]
    except Exception:  # noqa: BLE001 — names are best-effort
        return {}, {}
    makes = {str(i + 1): n for i, n in enumerate(make_names)}
    models = {str(i + 1): n for i, n in enumerate(model_names)}
    return makes, models


def _build_label_space(rel_paths: list[str]) -> dict[str, int]:
    model_ids = sorted({_model_id_of(p) for p in rel_paths})
    return {mid: i for i, mid in enumerate(model_ids)}


def load_classes(split: str = "train") -> list[CompCarsClass]:
    """One :class:`CompCarsClass` per model id present in the split file."""
    rel_paths = _read_split_file(SPLIT_DIR / f"{split}.txt")
    label_space = _build_label_space(rel_paths)
    make_names, model_names = _load_name_maps()

    # find the make_id that owns each model_id (first occurrence in paths)
    make_of_model: dict[str, str] = {}
    for p in rel_paths:
        parts = Path(p).parts
        if len(parts) >= 2 and parts[1] not in make_of_model:
            make_of_model[parts[1]] = parts[0]

    classes: list[CompCarsClass] = []
    for model_id, cid in label_space.items():
        make_id = make_of_model.get(model_id, "?")
        make = make_names.get(make_id, f"make_{make_id}").strip().lower() or f"make_{make_id}"
        model = model_names.get(model_id, f"model_{model_id}").strip().lower() or f"model_{model_id}"
        classes.append(CompCarsClass(
            class_id=cid, raw_name=f"{make} {model}".strip(),
            make=make, model=model, body_type="unknown", year=None, model_id=model_id,
        ))
    return classes


def load_train_samples(split: str = "train") -> list[CompCarsSample]:
    rel_paths = _read_split_file(SPLIT_DIR / f"{split}.txt")
    label_space = _build_label_space(rel_paths)
    out: list[CompCarsSample] = []
    for p in rel_paths:
        cid = label_space.get(_model_id_of(p))
        if cid is None:
            continue
        out.append(CompCarsSample(image_path=IMAGE_DIR / p, class_id=cid))
    return out


def split_train_val(
    samples: list[CompCarsSample],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[CompCarsSample], list[CompCarsSample]]:
    """Stratified per-class split (mirrors stanford_cars.split_train_val)."""
    import random
    rng = random.Random(seed)
    by_class: dict[int, list[CompCarsSample]] = {}
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


class CompCarsDataset(_TorchDataset):
    def __init__(self, items: list[CompCarsSample], transform: Optional[Callable] = None):
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


def build_torch_dataset(samples: list[CompCarsSample], transform: Optional[Callable] = None):
    return CompCarsDataset(samples, transform)


__all__ = [
    "CompCarsClass", "CompCarsSample", "CompCarsDataset",
    "load_classes", "load_train_samples", "split_train_val", "build_torch_dataset",
]
