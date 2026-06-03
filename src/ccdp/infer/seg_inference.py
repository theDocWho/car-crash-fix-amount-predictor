"""Thin Ultralytics YOLOv8-seg inference wrapper shared by Variant D.

Loads a segmentation ``.pt`` and returns per-instance masks **resized to the
original image size**, so the damage masks and the part masks (from two
different models run on the same image) live on the same pixel grid and can be
overlapped directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ImageLike = Union[str, Path, "object"]


@dataclass
class SegInstance:
    """One segmented instance: class name, score, boolean mask, bbox."""

    name: str
    score: float
    mask: "object"                       # np.ndarray[bool], shape (H, W) of the original image
    bbox: tuple[float, float, float, float]  # xyxy, original px

    @property
    def area_frac(self) -> float:
        import numpy as np
        m = self.mask
        return float(np.count_nonzero(m)) / float(m.size or 1)


def _resize_mask(mask, height: int, width: int):
    import numpy as np
    from PIL import Image
    im = Image.fromarray((mask.astype("uint8") * 255)).resize((width, height), Image.NEAREST)
    return np.asarray(im) > 127


class SegModel:
    """Ultralytics YOLOv8-seg model → list of :class:`SegInstance`.

    Inject ``model`` (anything exposing ``.predict``) to bypass loading in tests.
    """

    def __init__(self, ckpt: Optional[Path] = None, conf: float = 0.25, model=None):
        self.conf = conf
        self._model = model
        self._ckpt = ckpt

    def _ensure(self):
        if self._model is None:
            if self._ckpt is None or not Path(self._ckpt).exists():
                raise FileNotFoundError(f"Segmentation weights not found: {self._ckpt}")
            from ultralytics import YOLO
            self._model = YOLO(str(self._ckpt))
        return self._model

    def predict(self, image: ImageLike, conf: Optional[float] = None) -> list[SegInstance]:
        model = self._ensure()
        source = str(image) if isinstance(image, (str, Path)) else image
        res = model.predict(source, conf=(self.conf if conf is None else conf), verbose=False)[0]
        out: list[SegInstance] = []
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return out
        h, w = res.orig_shape
        names = res.names
        masks = res.masks.data.cpu().numpy()           # (N, mh, mw) at inference res
        cls = res.boxes.cls.cpu().tolist()
        conf_ = res.boxes.conf.cpu().tolist()
        xyxy = res.boxes.xyxy.cpu().tolist()
        for i in range(len(cls)):
            m = masks[i].astype(bool)
            if m.shape != (h, w):
                m = _resize_mask(m, h, w)
            out.append(SegInstance(
                name=names[int(cls[i])], score=float(conf_[i]),
                mask=m, bbox=tuple(xyxy[i]),
            ))
        return out


__all__ = ["SegModel", "SegInstance"]
