"""Build the CCDP submission as a single size-capped zip (<= 50 MB).

Produces one file:

  ccdp_submission.zip   — everything needed to read/run the notebook:
                          src/, scripts/, the notebook, the report, diagrams,
                          docs, sample images, eval CSVs. (~10 MB.)

Model weights are NOT bundled: the notebook's §1.3 cell downloads them on
demand from the v1.0.0 GitHub release (no credentials needed), and `.pt`
checkpoints are raw float tensors that barely compress anyway
(identifier.pt 103 MB -> 91.5 MB zipped; classifier.pt 270 MB -> 248 MB), so
they would never fit a 50 MB zip. Datasets are likewise fetched on demand by
the §1.5 cell. The notebook also reads fine from its baked outputs with no
weights or datasets at all.

Usage:
    python scripts/build_bundles.py                # -> dist/ccdp_submission.zip
    python scripts/build_bundles.py --out dist/     # custom output dir
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 50 * 1024 * 1024  # 50 MB hard cap

# All content, archived with its path relative to ROOT.
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

# Never ship weights/caches; the notebook fetches weights via §1.3.
EXCLUDE = (
    "__pycache__", ".pyc", ".DS_Store", ".ipynb_checkpoints", ".egg-info",
    ".pt", ".ubj", ".onnx",
)


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


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "ccdp_submission.zip"
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in CONTENT:
            for f in _iter_files(rel):
                zf.write(f, f.relative_to(ROOT))
                n += 1

    size = zip_path.stat().st_size
    status = "OK" if size <= MAX_BYTES else "OVER 50 MB!"
    print(f"  {zip_path.name:24s} {n:3d} files  {size/1e6:6.1f} MB  [{status}]")
    if size > MAX_BYTES:
        raise SystemExit(f"ERROR: {zip_path.name} exceeds the 50 MB cap.")
    print("\nSingle bundle within the 50 MB cap.")
    print("Weights are downloaded by the notebook's §1.3 cell (v1.0.0 release);")
    print("datasets (optional) by the §1.5 cell. Unzip, then follow submission/RUNNING.md.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    build(Path(ap.parse_args().out))
