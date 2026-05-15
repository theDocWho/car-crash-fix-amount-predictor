"""Checkpoint + model registry.

Layout::

    checkpoints/
    ├── registry.json
    ├── production/                     # symlinks to currently deployed models
    │   ├── identifier.pt -> ../identifier/run_2026-05-12_a/best.pt
    │   ├── classifier.pt
    │   └── ...
    ├── identifier/run_<ts>_<tag>/
    │   ├── last.pt
    │   ├── best.pt
    │   ├── config.yaml
    │   └── metrics.json
    └── ...

Each run dir owns its own ``last.pt`` and ``best.pt`` symlinks. ``registry.json``
is the canonical index: one entry per run, with metadata that lets us reconstruct
how it was trained (dataset hash, config hash, git commit, training_catalog_id,
val metrics).
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

CHECKPOINTS_ROOT = Path("checkpoints")
REGISTRY_PATH = CHECKPOINTS_ROOT / "registry.json"
PRODUCTION_DIR = CHECKPOINTS_ROOT / "production"


def _now_id(tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{ts}_{tag}"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@dataclass
class RegistryEntry:
    run_id: str
    variant: str                     # "identifier" | "classifier" | "detector" | "xgb_a" | ...
    run_dir: str
    created_at: str
    config_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    git_commit: Optional[str] = None
    training_catalog_id: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----- registry I/O ------------------------------------------------------


def _load_index() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"version": 1, "entries": [], "production": {}}
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def _save_index(idx: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w") as f:
        json.dump(idx, f, indent=2)


def create_run(
    variant: str,
    tag: str = "run",
    training_catalog_id: Optional[str] = None,
    notes: str = "",
) -> Path:
    """Allocate a new run directory and return its path."""
    run_id = _now_id(tag)
    run_dir = CHECKPOINTS_ROOT / variant / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    entry = RegistryEntry(
        run_id=run_id,
        variant=variant,
        run_dir=str(run_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        training_catalog_id=training_catalog_id,
        notes=notes,
    )
    idx = _load_index()
    idx["entries"].append(entry.to_dict())
    _save_index(idx)
    return run_dir


def update_metrics(run_id: str, metrics: dict[str, Any]) -> None:
    idx = _load_index()
    for e in idx["entries"]:
        if e["run_id"] == run_id:
            e["metrics"].update(metrics)
            break
    _save_index(idx)
    # also write into the run dir for self-containedness
    for e in idx["entries"]:
        if e["run_id"] == run_id:
            with (Path(e["run_dir"]) / "metrics.json").open("w") as f:
                json.dump(e["metrics"], f, indent=2)
            break


def list_entries(variant: Optional[str] = None) -> list[dict[str, Any]]:
    idx = _load_index()
    entries = idx["entries"]
    if variant is not None:
        entries = [e for e in entries if e["variant"] == variant]
    return entries


def production_target(variant: str) -> Optional[Path]:
    """Resolve the production weights path for ``variant``.

    Two lookup paths so this works both for full local development (where
    ``registry.json`` has a populated ``production`` dict) and for stripped-
    down deployments like the HuggingFace Space (where ``registry.json`` is
    not shipped — it's in ``.gitignore`` — and weights are dropped directly
    into ``checkpoints/production/{variant}.pt`` by the boot script):

      1. Registry index — authoritative when present.
      2. ``checkpoints/production/{variant}.pt`` on disk — fallback so the
         models actually load on HF. Without this, VariantBPipeline raised
         FileNotFoundError and VariantAPipeline silently fell back to an
         *untrained* ImageNet ResNet50 with a random 6-output head — every
         prediction was noise. (Bug latent since v0.1.0; this is the fix.)
    """
    idx = _load_index()
    target = idx.get("production", {}).get(variant)
    if target:
        return Path(target)
    # File-on-disk fallback for deployments without a registry.
    candidate = CHECKPOINTS_ROOT / "production" / f"{variant}.pt"
    if candidate.exists() or candidate.is_symlink():
        return candidate
    return None


def promote(run_id: str, variant: str, weights_filename: str = "best.pt") -> Path:
    """Repoint the production symlink for `variant` to this run's weights."""
    idx = _load_index()
    matches = [e for e in idx["entries"] if e["run_id"] == run_id and e["variant"] == variant]
    if not matches:
        raise KeyError(f"no run {run_id} for variant {variant}")
    target = Path(matches[0]["run_dir"]) / weights_filename
    if not target.exists():
        raise FileNotFoundError(f"weights not found: {target}")

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    link = PRODUCTION_DIR / f"{variant}.pt"
    if link.is_symlink() or link.exists():
        link.unlink()
    # store relative path so the repo is portable
    rel = os.path.relpath(target, link.parent)
    link.symlink_to(rel)

    idx.setdefault("production", {})[variant] = str(link)
    _save_index(idx)
    return link


# ----- checkpoint helpers ----------------------------------------------


def save_checkpoint(
    run_dir: Path,
    state: dict[str, Any],
    epoch: int,
    is_best: bool,
    filename: Optional[str] = None,
) -> Path:
    """Write a per-epoch checkpoint and refresh last.pt / best.pt symlinks."""
    import torch  # local — avoid import cost in non-training paths
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / (filename or f"epoch_{epoch:03d}.pt")
    torch.save(state, target)

    _update_symlink(run_dir / "last.pt", target.name)
    if is_best:
        _update_symlink(run_dir / "best.pt", target.name)
    return target


def _update_symlink(link: Path, target_name: str) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target_name)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    import torch
    return torch.load(path, map_location=map_location, weights_only=False)
