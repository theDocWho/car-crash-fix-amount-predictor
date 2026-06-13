"""Build the CCDP submission as size-capped zip bundles (<= 50 MB each).

Produces two zips:

  1. ccdp_submission.zip          — everything needed to read/run the notebook:
                                    src/, scripts/, the notebook, the report,
                                    diagrams, docs, sample images, eval CSVs.
                                    (~10 MB — all content in one file.)

  2. ccdp_weights_essential.zip   — the model weights that fit under 50 MB:
                                    detector + parts + damage-seg + the two
                                    XGBoost regressors + catalog. (~18 MB.)
                                    These unzip into submission/weights/ so the
                                    notebook's §1.3 picks them up automatically.

Why two zips and not one: a combined content+weights archive would still be
fine size-wise for the essentials, but the two largest models cannot be made
to fit a 50 MB cap by compression — `.pt` checkpoints are raw float tensors:

    identifier.pt   103 MB -> 91.5 MB zipped  (only 11% smaller)
    classifier.pt   270 MB -> 248  MB zipped

So identifier.pt and classifier.pt are NOT bundled; the notebook fetches them
from the v1.0.0 GitHub release (§1.3, no credentials needed). The essential
zip already covers the full damage/parts/cost pipeline (Variants B/C/D +
multi-car); identifier.pt only adds make-aware price tiers and classifier.pt
is used by Variant A.

Usage:
    python scripts/build_bundles.py                # -> dist/
    python scripts/build_bundles.py --out dist/     # custom output dir
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per zip

# ---- Bundle 1: all content (paths relative to ROOT, archived as-is) ----
CONTENT = [
    "src",
    "scripts",
    "pyproject.toml",
    "requirements.txt",
    "submission/ccdp_submission.ipynb",
    "submission/CCDP_Project_Report.docx",
    "submission/README.md",
    "submission/RUNNING.md",
    "submission/requirements.txt",
    "submission/assets",
    "submission/test_images",
    "data/eval",
]

# ---- Bundle 2: essential weights (source dir -> archived under submission/weights/) ----
WEIGHTS_SRC = ROOT / "submission/submission/weights"
WEIGHTS_ESSENTIAL = [
    "detector.pt", "parts.pt", "yoloseg.pt",
    "xgb_a.ubj", "xgb_b.ubj",
    "bundle_a.json", "bundle_b.json", "training_catalog.yaml",
]
# Deliberately excluded (too big for the 50 MB cap; fetched from the release):
WEIGHTS_RELEASE_ONLY = ["identifier.pt (~92 MB zipped)", "classifier.pt (~248 MB zipped)"]

EXCLUDE = ("__pycache__", ".pyc", ".DS_Store", ".ipynb_checkpoints", ".egg-info")


def _included(p: Path) -> bool:
    return not any(tok in str(p) for tok in EXCLUDE)


def _iter_files(rel: str):
    src = ROOT / rel
    if src.is_file():
        if _included(src):
            yield src
    elif src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file() and _included(f):
                yield f
    else:
        print(f"  ! missing (skipped): {rel}")


def _check(zip_path: Path, n: int) -> tuple[Path, int]:
    size = zip_path.stat().st_size
    status = "OK" if size <= MAX_BYTES else "OVER 50 MB!"
    print(f"  {zip_path.name:30s} {n:3d} files  {size/1e6:6.1f} MB  [{status}]")
    return zip_path, size


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building bundles into {out_dir}/  (cap {MAX_BYTES // (1024*1024)} MB each)\n")
    results = []

    # ---- Bundle 1: content ----
    z1 = out_dir / "ccdp_submission.zip"
    n = 0
    with zipfile.ZipFile(z1, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in CONTENT:
            for f in _iter_files(rel):
                zf.write(f, f.relative_to(ROOT)); n += 1
    results.append(_check(z1, n))

    # ---- Bundle 2: essential weights -> submission/weights/<name> ----
    z2 = out_dir / "ccdp_weights_essential.zip"
    n = 0
    if WEIGHTS_SRC.is_dir():
        with zipfile.ZipFile(z2, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for name in WEIGHTS_ESSENTIAL:
                src = WEIGHTS_SRC / name
                if src.exists():
                    zf.write(src, f"submission/weights/{name}"); n += 1
                else:
                    print(f"  ! missing weight (skipped): {name}")
        results.append(_check(z2, n))
    else:
        print(f"  ! weights source not found: {WEIGHTS_SRC} (skipping weights zip)")

    print()
    over = [z for z, s in results if s > MAX_BYTES]
    if over:
        raise SystemExit("ERROR: over the 50 MB cap: " + ", ".join(z.name for z in over))
    print(f"All {len(results)} bundles within the 50 MB cap. "
          f"Total {sum(s for _, s in results)/1e6:.1f} MB.")
    print("Not bundled (compression cannot reach 50 MB; fetched by notebook §1.3 "
          "from the v1.0.0 release):")
    for w in WEIGHTS_RELEASE_ONLY:
        print(f"  - {w}")
    print("\nUnzip ccdp_submission.zip, then (optionally) ccdp_weights_essential.zip "
          "into the SAME folder; follow submission/RUNNING.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    build(Path(ap.parse_args().out))
