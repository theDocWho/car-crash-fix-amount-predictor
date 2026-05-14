"""Small, single-purpose helpers used across `ccdp`.

Modules:
    device      — pick the best torch device (MPS / CUDA / CPU), seed RNGs.
    transforms  — torchvision transforms shared by trainers and inference.

These exist to keep the top-level training / inference code short and let any
single piece of behaviour have *one* canonical implementation.
"""

from .device import pick_device, seed_everything
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_transform,
    train_transform,
)

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "eval_transform",
    "pick_device",
    "seed_everything",
    "train_transform",
]
