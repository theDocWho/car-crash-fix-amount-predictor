"""Tests for the checkpoint + model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.registry import registry as reg


@pytest.fixture
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "CHECKPOINTS_ROOT", tmp_path)
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(reg, "PRODUCTION_DIR", tmp_path / "production")
    return tmp_path


def test_create_run_writes_index(tmp_registry):
    run_dir = reg.create_run(variant="identifier", tag="x", training_catalog_id="cat_a")
    assert run_dir.exists()
    entries = reg.list_entries()
    assert len(entries) == 1
    assert entries[0]["variant"] == "identifier"
    assert entries[0]["training_catalog_id"] == "cat_a"


def test_update_metrics_persists(tmp_registry):
    run_dir = reg.create_run(variant="identifier", tag="x")
    run_id = run_dir.name.replace("run_", "")
    reg.update_metrics(run_id, {"best_val_acc": 0.82, "epoch_1": {"val_acc": 0.5}})
    entries = reg.list_entries()
    assert entries[0]["metrics"]["best_val_acc"] == 0.82
    assert (run_dir / "metrics.json").exists()


def test_save_checkpoint_creates_symlinks(tmp_registry):
    run_dir = reg.create_run(variant="identifier", tag="x")
    target = reg.save_checkpoint(run_dir, state={"epoch": 1}, epoch=1, is_best=True)
    assert target.exists()
    assert (run_dir / "last.pt").is_symlink()
    assert (run_dir / "best.pt").is_symlink()


def test_promote_updates_production_symlink(tmp_registry):
    run_dir = reg.create_run(variant="identifier", tag="x")
    reg.save_checkpoint(run_dir, state={"epoch": 1}, epoch=1, is_best=True)
    run_id = run_dir.name.replace("run_", "")
    link = reg.promote(run_id, variant="identifier")
    assert link.is_symlink()
    assert reg.production_target("identifier") == link


def test_promote_missing_weights_errors(tmp_registry):
    run_dir = reg.create_run(variant="identifier", tag="x")
    run_id = run_dir.name.replace("run_", "")
    with pytest.raises(FileNotFoundError):
        reg.promote(run_id, variant="identifier", weights_filename="missing.pt")
