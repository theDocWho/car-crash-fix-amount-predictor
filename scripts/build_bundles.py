"""Build the CCDP submission as size-capped zip bundles.

The deliverable is split into three zips so each stays well under a 50 MB
upload cap. Model weights (>100 MB each) and the multi-GB datasets are
deliberately NOT bundled — the notebook fetches them on demand (§1.3 weights
from the GitHub release, §1.5 datasets). All three zips unpack into the SAME
folder and reconstruct the project layout described in submission/RUNNING.md.

    Bundle 1  ccdp_bundle_1_code.zip      — src/, scripts/, packaging metadata
    Bundle 2  ccdp_bundle_2_notebook.zip  — notebook, report, diagrams, docs
    Bundle 3  ccdp_bundle_3_samples.zip   — test images, small eval CSV/JSON, fetch helper

Usage:
    python scripts/build_bundles.py                 # -> dist/
    python scripts/build_bundles.py --out dist/      # custom output dir
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per zip

# Each bundle: list of (source path relative to ROOT, recurse?) entries.
BUNDLES: dict[str, list[str]] = {
    "ccdp_bundle_1_code": [
        "src",
        "scripts",
        "pyproject.toml",
        "requirements.txt",
        "submission/requirements.txt",
        "submission/RUNNING.md",
    ],
    "ccdp_bundle_2_notebook": [
        "submission/ccdp_submission.ipynb",
        "submission/CCDP_Project_Report.docx",
        "submission/README.md",
        "submission/assets",
    ],
    "ccdp_bundle_3_samples": [
        "submission/test_images",
        "data/eval",
    ],
}

# Globs to exclude from any bundle (caches, large/raw artifacts, weights).
EXCLUDE = (
    "__pycache__", ".pyc", ".DS_Store", ".ipynb_checkpoints",
    ".pt", ".ubj", ".onnx",            # never ship weights
)


def _included(p: Path) -> bool:
    s = str(p)
    return not any(tok in s for tok in EXCLUDE)


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


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building bundles into {out_dir}/  (cap {MAX_BYTES // (1024*1024)} MB each)\n")
    summary = []
    for name, entries in BUNDLES.items():
        zip_path = out_dir / f"{name}.zip"
        n = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for rel in entries:
                for f in _iter_files(rel):
                    zf.write(f, f.relative_to(ROOT))
                    n += 1
        size = zip_path.stat().st_size
        status = "OK" if size <= MAX_BYTES else "OVER 50 MB!"
        print(f"  {zip_path.name:30s} {n:4d} files  {size/1e6:6.1f} MB  [{status}]")
        summary.append((zip_path, size, n))

    print()
    over = [z for z, s, _ in summary if s > MAX_BYTES]
    if over:
        raise SystemExit(f"ERROR: {len(over)} bundle(s) exceed the 50 MB cap: "
                         + ", ".join(z.name for z in over))
    print(f"All {len(summary)} bundles within the 50 MB cap. "
          f"Total {sum(s for _, s, _ in summary)/1e6:.1f} MB.")
    print("Unzip all three into the same folder; then follow submission/RUNNING.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"), help="output directory")
    build(Path(ap.parse_args().out))
