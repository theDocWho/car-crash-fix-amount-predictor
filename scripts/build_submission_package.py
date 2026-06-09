"""Build the standalone submission package.

Produces `submission_package/` (and optionally a zip) containing everything
needed to run `ccdp_submission.ipynb` without internet, git, or the GitHub
release — bundles the package source, trained weights, sample images, and
a self-contained README.

Usage:
    python scripts/build_submission_package.py            # build folder only
    python scripts/build_submission_package.py --zip      # also build .zip
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Two variants ship side by side:
#  - vmmrdb (default, mainline)  : 1163-class VMMRdb continue-train identifier
#  - stanford (safety-net fallback): 196-class Stanford-only identifier
VARIANT_CONFIG = {
    "vmmrdb": {
        "pkg_name": "ccdp_submission_v0.2.0",
        "out_dir": ROOT / "submission_package",
        "notebook": ROOT / "notebooks" / "ccdp_submission.ipynb",
        "identifier_src": ROOT / "checkpoints/identifier/identifier.pt",
        "identifier_label": "ResNet-50 make/model identifier (VMMRdb 1163-class, val 0.3304)",
    },
    "stanford": {
        "pkg_name": "ccdp_submission_v0.2.0_stanford",
        "out_dir": ROOT / "submission_package_stanford",
        "notebook": ROOT / "notebooks" / "ccdp_submission_stanford.ipynb",
        "identifier_src": ROOT / "checkpoints/production/identifier_stanford.pt",
        "identifier_label": "ResNet-50 make/model identifier (Stanford-Cars 196-class, val 0.7703)",
    },
}

# Default kept for back-compat with callers that import these directly.
OUT = VARIANT_CONFIG["vmmrdb"]["out_dir"]
PKG_NAME = VARIANT_CONFIG["vmmrdb"]["pkg_name"]


def weight_sources_for(variant: str) -> dict[str, Path]:
    """Map of bundled-weight name -> source path for the given variant.

    `identifier.pt` differs across variants (VMMRdb 1163-class vs Stanford
    196-class); damage_seg / parts_seg / damage_det are identical because
    everything downstream of the identifier is the same pipeline.
    """
    cfg = VARIANT_CONFIG[variant]
    return {
        "identifier.pt": cfg["identifier_src"],
        "damage_seg.pt": ROOT / "checkpoints/production/yoloseg.pt",
        "parts_seg.pt":  ROOT / "checkpoints/production/parts.pt",
        "damage_det.pt": ROOT / "checkpoints/production/detector.pt",
        # damage_cls.pt (Variant A) is 283 MB — skipped in both variants.
    }


# Retained for back-compat with anything importing this module.
WEIGHT_SOURCES = weight_sources_for("vmmrdb")

# CarDD val images we ship as the demo inputs.
SAMPLE_IMAGE_DIR = ROOT / "data/raw/car-damage-detection/CarDD_release/CarDD_COCO/val2017"
SAMPLE_IMAGE_COUNT = 10


REQUIREMENTS_TXT = """\
# Install: pip install -r requirements.txt
# Then: pip install -e . (from this folder) so `import ccdp` works.

torch>=2.2
torchvision>=0.17
ultralytics>=8.1
xgboost>=2.0
scikit-learn>=1.4
pandas>=2.2
numpy>=1.26
pillow>=10.2
opencv-python-headless>=4.9
matplotlib>=3.8
pyyaml>=6.0
requests>=2.31
typer>=0.12
rich>=13.7
pydantic>=2.6
"""


def _build_readme(variant: str) -> str:
    cfg = VARIANT_CONFIG[variant]
    pkg_name = cfg["pkg_name"]
    id_label = cfg["identifier_label"]
    flavor_note = (
        "**This is the Stanford-only fallback variant.** The mainline submission "
        "(`ccdp_submission_v0.2.0.zip`) uses the VMMRdb-trained 1163-class "
        "identifier; this fallback uses the proven 196-class Stanford-only "
        "identifier (val 0.7703) as a safety net. Downstream (damage seg, parts "
        "seg, Variant D) is identical between the two.\n\n"
        if variant == "stanford" else ""
    )
    return f"""\
# CCDP — Car Crash Fix-Amount Predictor (capstone submission package)

{flavor_note}Standalone reproducible bundle of the project. **No internet, git, or
GitHub-release download required** — code, trained weights, and sample
images are all in this folder.

## What's in here

```
{pkg_name}/
├── README.md                 # this file
├── ccdp_submission.ipynb     # the single-notebook submission
├── requirements.txt          # pip deps
├── pyproject.toml            # package metadata so `pip install -e .` works
├── CITATIONS.md              # dataset citations
├── src/ccdp/                 # the Python package (vendored)
├── models/                   # 4 trained model weights (~120 MB)
│   ├── identifier.pt         # {id_label}
│   ├── damage_seg.pt         # YOLOv8-seg damage masks (CarDD nc=6)
│   ├── parts_seg.pt          # YOLOv8-seg car-parts masks (nc=15)
│   └── damage_det.pt         # YOLOv8 damage box detector (Variant B)
└── sample_images/            # {SAMPLE_IMAGE_COUNT} CarDD val images for the demo
```

## How to run

### Option A — Locally (recommended for review)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install jupyter
jupyter lab        # or: jupyter notebook
```

Open `ccdp_submission.ipynb` and run all cells top-to-bottom. **No training
is required to see results** — every training cell is guarded by
`RUN_TRAINING = False`, and the demo uses the bundled weights in `models/`.

### Option B — Google Colab

1. Upload the zip to Google Drive.
2. In a new Colab notebook:

   ```python
   from google.colab import drive; drive.mount('/content/drive')
   !unzip -q /content/drive/MyDrive/{pkg_name}.zip -d /content/
   %cd /content/{pkg_name}
   !pip -q install -r requirements.txt && pip -q install -e .
   ```

3. Open `ccdp_submission.ipynb` from the file browser and run.

## What the notebook does

1. **§1** Environment + bundled-weight wiring.
2. **§1.4** Datasets used + citations.
3. **§1.5** Sample-image preview.
4. **§2** Identifier — training narrative, curves, per-class F1, confusion matrices.
5. **§3** Damage segmentation training (CarDD nc=6).
6. **§3b** (optional) Path A extension with HITL.
7. **§4** Parts segmentation training (carparts nc=15).
8. **§5–§8** Variant A → B → C → D walkthrough.
9. **§9** Multi-car extension.
10. **§10** Live inference demo.
11. **§11** Reproducibility checklist + final metrics + Variant D smoke holdout MAE.

## Notes for the reviewer

- `models/identifier.pt` is **self-describing** — `class_names`, `num_classes`,
  `best_val`, and the training config are embedded in the .pt itself. §2.2
  loads and prints them as the live model card.
- The `damage_cls.pt` (Variant A multilabel head) is NOT included — 283 MB
  and Variant A is shown only schematically. Variant D is what ships.
- Datasets cited in `CITATIONS.md` are NOT bundled — see citations for the
  Kaggle / HF source links.

## Links

- Code repo: <https://github.com/theDocWho/car-crash-fix-amount-predictor>
- Weights release: v0.2.0 on the same repo
"""




# -----------------------------------------------------------------------------
# Notebook patching: take the canonical notebook and rewrite the setup cells
# so they work standalone (no git clone, no release download).
# -----------------------------------------------------------------------------

NB_SETUP_INSTALL_CELL = """\
# === Submission-package setup ===
# This notebook is shipped inside `{pkg_name}/`. The bundled package source is
# in `src/ccdp/`, weights in `models/`, sample images in `sample_images/`.
# Run `pip install -r requirements.txt && pip install -e .` from the package
# root BEFORE opening this notebook (see README.md).

import os, sys, pathlib
PKG_ROOT = pathlib.Path('.').resolve()
# Detect Colab so paths still resolve if the user opened the notebook from /content/
if 'google.colab' in sys.modules and not (PKG_ROOT / 'src' / 'ccdp').exists():
    # Try the conventional unzipped location
    cands = sorted(pathlib.Path('/content').glob('{pkg_name}*'))
    if cands:
        PKG_ROOT = cands[-1].resolve()
        os.chdir(PKG_ROOT)
        print(f'Switched to {{PKG_ROOT}}')

assert (PKG_ROOT / 'src' / 'ccdp').exists(), (
    f"Can't find src/ccdp at {{PKG_ROOT}}. Open this notebook from the package root.")

try:
    import ccdp
    print(f'ccdp imported OK from {{PKG_ROOT}}')
except ImportError:
    print('ccdp not installed — running: pip install -e .')
    os.system('pip -q install -e .')
    import ccdp
    print('ccdp installed and imported')
""".replace("{pkg_name}", PKG_NAME)

NB_WEIGHTS_CELL = """\
# === Wire bundled weights into the path the inference cells read from ===
# Copies models/*.pt -> checkpoints/production/<name>.pt (and yoloseg.pt /
# parts.pt aliases for the existing inference modules).
import pathlib, shutil

PKG_ROOT = pathlib.Path('.').resolve()
PROD = PKG_ROOT / 'checkpoints' / 'production'
PROD.mkdir(parents=True, exist_ok=True)

# Submission-package name -> destination filenames in checkpoints/production/.
# The inference modules read 'identifier.pt', 'yoloseg.pt' (damage seg),
# 'parts.pt', 'detector.pt'.
MAPPING = {
    'identifier.pt': ['identifier.pt'],
    'damage_seg.pt': ['yoloseg.pt', 'damage_seg.pt'],
    'parts_seg.pt':  ['parts.pt', 'parts_seg.pt'],
    'damage_det.pt': ['detector.pt', 'damage_det.pt'],
}
for src_name, dst_names in MAPPING.items():
    src = PKG_ROOT / 'models' / src_name
    if not src.exists():
        print(f'  {src_name:18s} NOT in models/ — inference cells may fall back to schematics')
        continue
    for dst_name in dst_names:
        dst = PROD / dst_name
        if not dst.exists():
            shutil.copy(src, dst)
    print(f'  {src_name:18s} -> {", ".join(str((PROD / d).relative_to(PKG_ROOT)) for d in dst_names)}')

# Initialise the parts-cost catalog
os.system('ccdp costing init || true')
"""


def patch_notebook(src_nb: Path, dst_nb: Path) -> None:
    """Read the canonical notebook, swap out §1.1 install + §1.3 weight-fetch
    cells for standalone equivalents, and write to dst."""
    nb = json.loads(src_nb.read_text())

    def join_src(c):
        s = c.get("source", "")
        return s if isinstance(s, str) else "".join(s)

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = join_src(cell)
        # §1.1 — replace the git-clone install cell
        if "git clone" in src and "car-crash-fix-amount-predictor" in src:
            cell["source"] = NB_SETUP_INSTALL_CELL
        # §1.3 — replace the urllib download-from-release cell
        elif "urllib.request.urlretrieve" in src and "releases/download" in src:
            cell["source"] = NB_WEIGHTS_CELL
        # §1.5 / §10 sample-image fallback — also offer the bundled sample_images dir
        elif "data/raw/car-damage-detection" in src and "cardd_val" in src:
            cell["source"] = src.replace(
                "cardd_val = Path('data/raw/car-damage-detection/CarDD_release/CarDD_COCO/val2017')",
                "cardd_val = (Path('sample_images') if Path('sample_images').exists()\n"
                "             else Path('data/raw/car-damage-detection/CarDD_release/CarDD_COCO/val2017'))",
            ).replace(
                "cardd_val = pathlib.Path('data/raw/car-damage-detection/CarDD_release/CarDD_COCO/val2017')",
                "cardd_val = (pathlib.Path('sample_images') if pathlib.Path('sample_images').exists()\n"
                "             else pathlib.Path('data/raw/car-damage-detection/CarDD_release/CarDD_COCO/val2017'))",
            )
    dst_nb.parent.mkdir(parents=True, exist_ok=True)
    dst_nb.write_text(json.dumps(nb, indent=1))


# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------

def _ensure_stanford_identifier_checkpoint() -> Path:
    """Generate ``checkpoints/production/identifier_stanford.pt`` if missing.

    Slimmed copy of the Stanford-Cars 196-class best.pt (drops optimizer/scheduler
    state, ~280 MB → 99 MB) plus ``class_names`` injected from the dataset — so
    §2.2 of the notebook can show a self-describing model card without needing
    the source repo's data loaders at notebook-bake time.
    """
    dst = ROOT / "checkpoints/production/identifier_stanford.pt"
    if dst.exists():
        return dst
    src = ROOT / "checkpoints/identifier/run_2026-05-13T14-30-04_identifier_v2/best.pt"
    if not src.exists():
        sys.exit(
            f"FATAL: Stanford run dir missing — {src}\n"
            f"Cannot synthesize identifier_stanford.pt without the original run."
        )
    import torch
    from ccdp.data import stanford_cars as sc

    print(f"  synthesizing identifier_stanford.pt from {src.name}…")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    classes = sc.load_classes()
    slim = {
        "model": ck["model"],
        "epoch": ck["epoch"],
        "stage": ck["stage"],
        "best_val": ck["best_val"],
        "num_classes": ck["num_classes"],
        "class_names": [c.raw_name for c in classes],
        "config": ck.get("config", {}),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)
    print(f"  wrote {dst}  ({dst.stat().st_size/1e6:.1f} MB, {slim['num_classes']} classes)")
    return dst


def build(out: Path, with_zip: bool, variant: str = "vmmrdb") -> None:
    cfg = VARIANT_CONFIG[variant]
    if variant == "stanford":
        _ensure_stanford_identifier_checkpoint()
    weight_sources = weight_sources_for(variant)
    pkg_name = cfg["pkg_name"]
    notebook_src = cfg["notebook"]
    print(f"=== Building variant '{variant}' -> {out} ===")
    print(f"    identifier: {cfg['identifier_label']}")
    if not cfg["identifier_src"].exists():
        sys.exit(f"FATAL: identifier source missing — {cfg['identifier_src']}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # 1. Vendor the package source
    src_pkg = out / "src" / "ccdp"
    shutil.copytree(ROOT / "src" / "ccdp", src_pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"vendored src/ccdp -> {src_pkg}")

    # 2. Minimal pyproject.toml — pip install -e . needs this
    pyproject = (ROOT / "pyproject.toml").read_text()
    # Strip the dev/serve extras — submission only needs core + ml
    (out / "pyproject.toml").write_text(pyproject)
    print("wrote pyproject.toml")

    # 3. requirements.txt
    (out / "requirements.txt").write_text(REQUIREMENTS_TXT)
    print("wrote requirements.txt")

    # 4. README.md
    (out / "README.md").write_text(_build_readme(variant))
    print("wrote README.md")

    # 5. CITATIONS.md
    shutil.copy(ROOT / "CITATIONS.md", out / "CITATIONS.md")
    print("copied CITATIONS.md")

    # 6. Bundled weights
    models_dir = out / "models"
    models_dir.mkdir()
    for name, src in weight_sources.items():
        if not src.exists():
            print(f"  WARN: {src} missing — skipping {name}")
            continue
        shutil.copy(src, models_dir / name)
        size_mb = (models_dir / name).stat().st_size / 1e6
        print(f"  models/{name}  ({size_mb:.1f} MB)")

    # 7. Sample images
    samples_dir = out / "sample_images"
    samples_dir.mkdir()
    if SAMPLE_IMAGE_DIR.exists():
        for i, p in enumerate(sorted(SAMPLE_IMAGE_DIR.glob("*.jpg"))[:SAMPLE_IMAGE_COUNT]):
            shutil.copy(p, samples_dir / p.name)
        print(f"copied {len(list(samples_dir.iterdir()))} sample images")
    else:
        print(f"  WARN: {SAMPLE_IMAGE_DIR} missing — sample_images/ is empty")

    # 8. Standalone notebook (variant-specific)
    patch_notebook(notebook_src, out / "ccdp_submission.ipynb")
    print(f"patched + wrote ccdp_submission.ipynb (from {notebook_src.name})")

    # 8b. Pre-rendered diagram PNGs (Mermaid → PNG via mermaid.ink). The
    # notebook's markdown references `assets/diagrams/*.png`; these need to
    # ship alongside the .ipynb for offline rendering.
    diagrams_src = ROOT / "notebooks" / "assets" / "diagrams"
    if diagrams_src.exists():
        diagrams_dst = out / "assets" / "diagrams"
        diagrams_dst.mkdir(parents=True, exist_ok=True)
        for p in diagrams_src.glob("*.png"):
            shutil.copy(p, diagrams_dst / p.name)
        n = sum(1 for _ in diagrams_dst.glob("*.png"))
        print(f"copied {n} pre-rendered diagram PNGs -> assets/diagrams/")
    else:
        print("  WARN: notebooks/assets/diagrams/ missing — Mermaid images won't render offline")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nPackage: {out}  ({total/1e6:.1f} MB across {sum(1 for _ in out.rglob('*') if _.is_file())} files)")

    if with_zip:
        zip_path = out.parent / f"{pkg_name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in out.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=Path(pkg_name) / f.relative_to(out))
        print(f"zipped -> {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_CONFIG), default="vmmrdb",
                    help="Which submission flavour to build (default: vmmrdb mainline; "
                         "stanford for the 196-class safety-net fallback).")
    ap.add_argument("--zip", action="store_true", help="Also produce the .zip alongside the folder.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output dir (defaults to the per-variant out_dir).")
    args = ap.parse_args()
    out = args.out or VARIANT_CONFIG[args.variant]["out_dir"]
    build(out, with_zip=args.zip, variant=args.variant)


if __name__ == "__main__":
    main()
