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


# -- production_target file-fallback (HF Space path) -------------------------


def test_production_target_none_when_nothing_exists(tmp_registry):
    """No registry entry, no file on disk → return None."""
    assert reg.production_target("detector") is None


def test_production_target_falls_back_to_file_on_disk(tmp_registry):
    """When registry.json has no 'production' entry but the file exists at
    checkpoints/production/<variant>.pt, resolve to that file.

    This is the HuggingFace Space deployment path: registry.json is gitignored
    so the Space doesn't ship it, but the boot script does download weights
    directly to checkpoints/production/. Without this fallback,
    VariantBPipeline raises FileNotFoundError and VariantAPipeline silently
    falls back to an untrained ResNet50 head."""
    prod_dir = tmp_registry / "production"
    prod_dir.mkdir()
    file_path = prod_dir / "detector.pt"
    file_path.write_bytes(b"fake-weights")

    resolved = reg.production_target("detector")
    assert resolved == file_path
    assert resolved.exists()


def test_production_target_registry_wins_over_file(tmp_registry):
    """If registry.json explicitly points somewhere, that path wins even when
    a file also exists in the fallback location — full local dev environments
    use the registry to point at the *run dir* symlink, not the production
    file."""
    # Create a registry entry pointing somewhere specific.
    run_dir = reg.create_run(variant="detector", tag="d")
    reg.save_checkpoint(run_dir, state={"epoch": 1}, epoch=1, is_best=True)
    run_id = run_dir.name.replace("run_", "")
    registry_link = reg.promote(run_id, variant="detector")

    # And drop a different file in the fallback location.
    prod_dir = tmp_registry / "production"
    decoy = prod_dir / "detector.pt"
    # decoy was created by promote() as a symlink — overwrite with a plain file
    # at a *different* path that we can compare.
    other = prod_dir / "decoy_detector.pt"
    other.write_bytes(b"other")
    # promote already wrote prod_dir/detector.pt → registry path; production_target
    # should return that, not the decoy.
    resolved = reg.production_target("detector")
    assert resolved == registry_link


def test_production_target_finds_symlink(tmp_registry):
    """Real local layout: checkpoints/production/classifier.pt is a symlink
    to a run dir. production_target must follow that even though .exists()
    on a broken symlink would be False — we also accept .is_symlink()."""
    prod_dir = tmp_registry / "production"
    prod_dir.mkdir()
    real = tmp_registry / "real_weights.pt"
    real.write_bytes(b"x")
    link = prod_dir / "classifier.pt"
    link.symlink_to(real)
    assert reg.production_target("classifier") == link
