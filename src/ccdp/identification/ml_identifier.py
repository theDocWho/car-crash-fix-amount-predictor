"""ML make/model/year identification — the stage that was trained but never wired.

Phase 1.5 trained a ResNet-50 on Stanford Cars (val acc 77%) and shipped the
weights, but :func:`ccdp.identification.car_identifier.identify` only ever ran
the filename/EXIF/OCR heuristics — the model was never called. This module is
the missing inference wrapper.

Class-index → (make, model, year) mapping uses a **bundled** class-name resource
(`data/stanford_cars_classes.json`) so inference works on a deployment that
never downloaded the Stanford Cars dataset. It falls back to reading the dataset
devkit if the bundle is somehow absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ccdp.data.stanford_cars import parse_class_name

_CLASS_RESOURCE = Path(__file__).parent / "data" / "stanford_cars_classes.json"

ImageLike = Union[str, Path, "object"]  # path-like or PIL.Image.Image


def load_class_names() -> Optional[list[str]]:
    """Load the 196 raw Stanford Cars class names.

    Order of preference:
      1. Bundled JSON resource (works anywhere, no dataset needed).
      2. The dataset devkit, if present (dev machines).
    Returns ``None`` if neither is available — the caller then emits generic
    ``class_<i>`` labels rather than crashing.
    """
    if _CLASS_RESOURCE.exists():
        try:
            return json.loads(_CLASS_RESOURCE.read_text())
        except (ValueError, OSError):
            pass
    try:
        from ccdp.data import stanford_cars as sc
        return [c.raw_name for c in sc.load_classes()]
    except Exception:  # noqa: BLE001 — dataset/scipy may be unavailable
        return None


@dataclass
class MLIdentification:
    """Structured output of :meth:`MLIdentifier.predict`."""

    make: Optional[str]
    model: Optional[str]
    year: Optional[int]
    body_type: str
    raw_name: str
    class_id: int
    confidence: float
    topk: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "body_type": self.body_type,
            "raw_name": self.raw_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "topk": [list(t) for t in self.topk],
        }


class MLIdentifier:
    """ResNet-50 make/model/year classifier inference wrapper.

    The model is lazy-loaded from the production identifier checkpoint on first
    :meth:`predict`. Inject ``model`` + ``class_names`` to bypass the checkpoint
    entirely in tests.
    """

    def __init__(
        self,
        model=None,
        class_names: Optional[list[str]] = None,
        ckpt_path: Optional[Path] = None,
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        self._model = model
        self._class_names = class_names
        self._ckpt_path = ckpt_path
        self._device = device
        self.image_size = image_size
        self._transform = None

    # -- lazy loading ------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from ccdp.models.identifier import build_resnet50_identifier
        from ccdp.registry import load_checkpoint, production_target
        from ccdp.utils import pick_device

        ckpt = self._ckpt_path or production_target("identifier")
        if ckpt is None or not Path(ckpt).exists():
            raise FileNotFoundError(
                "No identifier weights available. Train + promote one via "
                "`ccdp train identifier` then `ccdp registry promote <run_id> identifier`."
            )
        if self._device is None:
            self._device = str(pick_device())
        ck = load_checkpoint(Path(ckpt), map_location=self._device)
        # A continue-trained checkpoint (CompCars / VMMRdb) embeds its own class
        # names + count, so the model self-describes regardless of the bundled map.
        if self._class_names is None and ck.get("class_names"):
            self._class_names = list(ck["class_names"])
        num_classes = int(ck.get("num_classes") or len(self._names()))
        model = build_resnet50_identifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(ck["model"])
        model.eval().to(self._device)
        self._model = model
        return model

    def _names(self) -> list[str]:
        if self._class_names is None:
            self._class_names = load_class_names() or []
        return self._class_names

    def _eval_transform(self):
        if self._transform is None:
            from ccdp.utils import eval_transform
            self._transform = eval_transform(self.image_size)
        return self._transform

    # -- public API --------------------------------------------------------

    def predict(self, image: ImageLike, topk: int = 3) -> MLIdentification:
        """Classify a (ideally car-cropped) image into make/model/year."""
        import torch
        from PIL import Image

        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        model = self._ensure_model()
        device = self._device or "cpu"
        x = self._eval_transform()(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).squeeze(0).cpu()

        names = self._names()
        k = min(topk, probs.numel())
        top_p, top_i = torch.topk(probs, k)
        topk_pairs = [
            (self._raw_name(int(i), names), float(p))
            for p, i in zip(top_p.tolist(), top_i.tolist())
        ]

        class_id = int(top_i[0].item())
        confidence = float(top_p[0].item())
        raw = self._raw_name(class_id, names)
        parsed = parse_class_name(raw) if names else None
        return MLIdentification(
            make=(parsed.make if parsed else None),
            model=(parsed.model if parsed else None),
            year=(parsed.year if parsed else None),
            body_type=(parsed.body_type if parsed else "unknown"),
            raw_name=raw,
            class_id=class_id,
            confidence=confidence,
            topk=topk_pairs,
        )

    @staticmethod
    def _raw_name(idx: int, names: list[str]) -> str:
        if 0 <= idx < len(names):
            return names[idx]
        return f"class_{idx}"


__all__ = ["MLIdentifier", "MLIdentification", "load_class_names"]
