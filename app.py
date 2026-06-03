"""HuggingFace Space entrypoint.

The HF Space Gradio SDK looks for an ``app.py`` at the repo root that defines
a Gradio app named ``demo``. We delegate to :func:`ccdp.api.demo.build_demo`
so the demo's behaviour lives in the package and stays testable.

On first boot the Space downloads model weights from this repo's GitHub
Release (v0.1.0) into ``checkpoints/production/``. Subsequent boots reuse the
cached weights, so cold-start is only slow once (~30 s on free CPU).

Layout produced on disk so the registry path-resolution logic finds the
XGBoost bundles correctly:

    checkpoints/production/
    ├── classifier.pt
    ├── detector.pt
    ├── identifier.pt
    ├── xgb_a/
    │   ├── best.pt          (symlink to best.ubj)
    │   ├── best.ubj
    │   └── bundle.json
    ├── xgb_a.pt             (symlink to xgb_a/best.pt) — used by `production_target("xgb_a")`
    ├── xgb_b/  (same shape)
    └── xgb_b.pt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# HuggingFace Spaces run `app.py` from the repo root without installing the
# local `ccdp` package. Our source lives under `src/ccdp/`, so we add `src/`
# to sys.path here — before any `ccdp` import — so the bootstrap works
# whether or not `pip install -e .` was run.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

WEIGHTS_DIR = Path("checkpoints/production")
CATALOG_DIR = Path("data/parts_cost_catalog")
RELEASE_TAG = "v0.2.0"
GITHUB_REPO = "theDocWho/car-crash-fix-amount-predictor"

# Top-level assets that go directly under WEIGHTS_DIR
TOP_LEVEL_ASSETS = [
    "classifier.pt",
    "detector.pt",
    "identifier.pt",
    "training_catalog.yaml",
]

# Variant-scoped assets: (filename_in_release, destination_subdir, local_name)
XGB_ASSETS = [
    ("xgb_a.ubj",     "xgb_a", "best.ubj"),
    ("bundle_a.json", "xgb_a", "bundle.json"),
    ("xgb_b.ubj",     "xgb_b", "best.ubj"),
    ("bundle_b.json", "xgb_b", "bundle.json"),
]

# Variant D (parts-aware YOLOv8-seg) weights are OPTIONAL: fetched best-effort so
# the Space keeps booting on releases that predate them. Upload these to the
# release tag above to light Variant D up — no code change or tag bump needed.
OPTIONAL_TOP_LEVEL = ["yoloseg.pt", "parts.pt"]


def _curl(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dst), url], check=True)


def _curl_optional(url: str, dst: Path) -> bool:
    """Best-effort fetch — returns False (and leaves nothing) if the asset is absent."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["curl", "-fL", "--retry", "2", "-o", str(dst), url]).returncode != 0:
        dst.unlink(missing_ok=True)
        return False
    return True


def _fetch_release_assets() -> None:
    """Download missing release assets into ``checkpoints/production/``.

    Top-level model weights go directly under the production dir. XGBoost
    assets are placed inside a per-variant subdirectory so the registry-style
    symlinks (`xgb_a.pt` -> `xgb_a/best.pt`) point at a dir that also contains
    the matching `bundle.json` — which is what `BaseVariantPipeline` needs.

    If a ``.release_tag`` sentinel exists with a different value than the
    current ``RELEASE_TAG``, wipe the cached weights so we don't keep
    serving stale models after a release bump on persistent-storage Spaces.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = WEIGHTS_DIR / ".release_tag"
    if sentinel.exists() and sentinel.read_text().strip() != RELEASE_TAG:
        print(f"[boot] release tag changed "
              f"({sentinel.read_text().strip()} -> {RELEASE_TAG}); refreshing cached weights")
        for asset in TOP_LEVEL_ASSETS:
            (WEIGHTS_DIR / asset).unlink(missing_ok=True)
        for asset, subdir, local_name in XGB_ASSETS:
            (WEIGHTS_DIR / subdir / local_name).unlink(missing_ok=True)
            (WEIGHTS_DIR / subdir / "best.pt").unlink(missing_ok=True)
        for variant in ("xgb_a", "xgb_b"):
            link = WEIGHTS_DIR / f"{variant}.pt"
            if link.is_symlink() or link.exists():
                link.unlink()
    base = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"

    # Top-level files
    for asset in TOP_LEVEL_ASSETS:
        dst = WEIGHTS_DIR / asset
        if dst.exists():
            continue
        print(f"[boot] fetch {asset}")
        _curl(f"{base}/{asset}", dst)

    # Per-variant XGBoost files into their own subdirs
    for asset, subdir, local_name in XGB_ASSETS:
        dst = WEIGHTS_DIR / subdir / local_name
        if dst.exists():
            continue
        print(f"[boot] fetch {asset} -> {subdir}/{local_name}")
        _curl(f"{base}/{asset}", dst)

    # Mirror best.ubj -> best.pt so the BaseVariantPipeline's `*.pt` symlink
    # chain resolves correctly.
    for _, subdir, _ in XGB_ASSETS:
        run_dir = WEIGHTS_DIR / subdir
        best_ubj = run_dir / "best.ubj"
        best_pt = run_dir / "best.pt"
        if best_ubj.exists() and not best_pt.exists():
            best_pt.symlink_to("best.ubj")

    # And the top-level xgb_a.pt / xgb_b.pt symlinks the registry expects.
    for variant in ("xgb_a", "xgb_b"):
        link = WEIGHTS_DIR / f"{variant}.pt"
        target_rel = f"{variant}/best.pt"
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(target_rel)

    # Best-effort: Variant D parts-aware seg weights (absent on older releases).
    for asset in OPTIONAL_TOP_LEVEL:
        dst = WEIGHTS_DIR / asset
        if not dst.exists() and _curl_optional(f"{base}/{asset}", dst):
            print(f"[boot] fetched optional {asset}")

    # Stamp the release tag so next boot can detect a release bump.
    sentinel.write_text(RELEASE_TAG)


def _bootstrap_catalog() -> None:
    """Place the bundled catalog YAML where ``ccdp.costing`` expects it."""
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    src = WEIGHTS_DIR / "training_catalog.yaml"
    if not src.exists():
        return
    target = CATALOG_DIR / "catalog_2026-05-12T05-45-11_initial.yaml"
    if not target.exists():
        target.write_bytes(src.read_bytes())
    active = CATALOG_DIR / "active.yaml"
    if not active.exists() and not active.is_symlink():
        active.symlink_to(target.name)


_fetch_release_assets()
_bootstrap_catalog()

# Import deferred until after the bootstrap so the loaders find catalog + weights.
from ccdp.api.demo import build_demo  # noqa: E402

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
