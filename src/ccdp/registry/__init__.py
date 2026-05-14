"""ccdp.registry — checkpoint + model registry."""

from .registry import (
    CHECKPOINTS_ROOT,
    PRODUCTION_DIR,
    REGISTRY_PATH,
    RegistryEntry,
    create_run,
    list_entries,
    load_checkpoint,
    production_target,
    promote,
    save_checkpoint,
    update_metrics,
)

__all__ = [
    "CHECKPOINTS_ROOT",
    "PRODUCTION_DIR",
    "REGISTRY_PATH",
    "RegistryEntry",
    "create_run",
    "list_entries",
    "load_checkpoint",
    "production_target",
    "promote",
    "save_checkpoint",
    "update_metrics",
]
