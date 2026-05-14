"""Device selection and reproducibility helpers.

Centralised so the same logic can't drift between trainers and inference.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device() -> torch.device:
    """Return the most capable torch device available.

    Order of preference: Apple Silicon MPS, NVIDIA CUDA, CPU. Chosen here
    because the project's primary target is M-series Macs; CUDA support is
    a free bonus when running on a workstation.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Set Python, NumPy and PyTorch RNG seeds for reproducible runs.

    Note: full bitwise reproducibility on MPS isn't guaranteed by PyTorch;
    this gets us *seed-driven* identical splits and broadly identical training
    trajectories, which is what we want for science.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
