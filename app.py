"""HuggingFace Space entrypoint.

The HF Space Gradio SDK looks for an ``app.py`` at the repo root that defines
a Gradio app named ``demo``. We delegate to :func:`ccdp.api.demo.build_demo`
so the demo's behaviour lives in the package and stays testable.

On first boot the Space downloads model weights from this repo's GitHub
Release (v0.1.0) into ``checkpoints/production/``. Subsequent boots reuse the
cached weights, so cold-start is only slow once (~30 s on free CPU).

Caveat (documented for v0.1): the XGBoost bundle JSON sidecars don't ship in
the v0.1.0 release yet, so the Space's predictions fall through to the
catalog-only Tier 3 fallback. Cost numbers will be approximations rather than
the trained XGBoost output. Damage detection (Variant A classifier + Variant B
detector) is unaffected. A future release will bundle the JSONs and this file
will pick them up automatically.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

WEIGHTS_DIR = Path("checkpoints/production")
CATALOG_DIR = Path("data/parts_cost_catalog")
RELEASE_TAG = "v0.1.0"
GITHUB_REPO = "theDocWho/car-crash-fix-amount-predictor"
ASSETS = [
    "classifier.pt",
    "detector.pt",
    "identifier.pt",
    "xgb_a.ubj",
    "xgb_b.ubj",
    "training_catalog.yaml",
]


def _fetch_release_assets() -> None:
    """Download missing release assets into ``checkpoints/production/``."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    missing = [a for a in ASSETS if not (WEIGHTS_DIR / a).exists()]
    if not missing:
        print("[boot] all release assets already present; skipping download")
        return
    print(f"[boot] downloading {len(missing)} asset(s) from {GITHUB_REPO}@{RELEASE_TAG}")
    base = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"
    for asset in missing:
        dst = WEIGHTS_DIR / asset
        subprocess.run(
            ["curl", "-fL", "--retry", "3", "-o", str(dst), f"{base}/{asset}"],
            check=True,
        )
        size_kb = dst.stat().st_size // 1024
        print(f"[boot] fetched {asset} ({size_kb} KB)")


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
