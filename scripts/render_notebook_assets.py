"""Pre-render the submission notebook so it shows everything on GitHub.

GitHub renders .ipynb statically — Mermaid code blocks inside markdown cells
show as plain code (no diagram), and code cells with no saved outputs look
blank. This script fixes both:

1. **Mermaid → PNG.** Each ```mermaid block is replaced with an ![](image)
   reference. PNGs fetched from the public mermaid.ink renderer and saved to
   notebooks/assets/diagrams/.
2. **Dataset previews + inference previews → executed outputs.** Re-executes
   the cells that produce matplotlib grids / inference visualizations so the
   .ipynb on disk ships with embedded PNG outputs that render on GitHub.

Idempotent. Re-run any time the notebook changes.

Usage:
    python scripts/render_notebook_assets.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import ssl
import urllib.request
from pathlib import Path

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "ccdp_submission.ipynb"
DIAGRAMS_DIR = ROOT / "notebooks" / "assets" / "diagrams"
MERMAID_INK = "https://mermaid.ink/img/{b64}?type=png&bgColor=ffffff"

# Which code cells to execute and bake outputs for (matched by substring).
# Heavy / network-dependent cells are skipped — we only execute the cheap
# previewers that produce useful static images.
EXECUTE_IF_CONTAINS = (
    "show_grid(samples[:8], 'CarDD val samples",
    "show_grid(samples[:8], 'Stanford-Cars random",
    "label_files = list(yolo_dir.glob",            # YOLO label preview (text only, but still)
    "model = YOLO(str(weights))",                  # damage_seg overlay preview (§3.3)
    "print(f'Parts detected in",                   # parts_seg detection list (§4.1)
)

# Cells we never auto-execute (require uploaded image, slow, or interactive)
NEVER_EXECUTE_IF_CONTAINS = (
    "from google.colab import files",
    "estimate_multi(IMG_PATH)",
    "files.upload()",
)


# ---------------------------------------------------------------------------
# Mermaid → PNG
# ---------------------------------------------------------------------------

def _mermaid_to_png(source: str, dest: Path) -> bool:
    """POST the Mermaid source to mermaid.ink and save the PNG. Returns True on success."""
    # mermaid.ink uses url-safe base64 (no padding stripped in their decoder).
    b64 = base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii")
    url = MERMAID_INK.format(b64=b64)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ccdp-build/0.1"})
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
            data = r.read()
        if not data or len(data) < 200:
            print(f"  ! mermaid.ink returned {len(data)} bytes for {dest.name} — skipping")
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  ! mermaid.ink failed for {dest.name}: {e}")
        return False


_MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)


def _stable_id(source: str, ix: int) -> str:
    """Deterministic name per diagram so re-runs don't churn the disk."""
    h = hashlib.sha1(source.encode()).hexdigest()[:10]
    return f"diagram_{ix:02d}_{h}"


def replace_mermaid_blocks(nb: dict) -> int:
    """Mutate notebook in place: replace each Mermaid fence with an image
    reference (relative to the notebook). Return the count replaced."""
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}      # source -> rel path (dedupe identical diagrams)
    ix = 0
    n_replaced = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        if "```mermaid" not in src:
            continue

        def _sub(match):
            nonlocal ix, n_replaced
            body = match.group(1).strip()
            if body in seen:
                return f"![diagram](assets/diagrams/{seen[body]})"
            name = _stable_id(body, ix) + ".png"
            ix += 1
            dest = DIAGRAMS_DIR / name
            if not dest.exists():
                ok = _mermaid_to_png(body, dest)
                if not ok:
                    return match.group(0)   # keep raw fence on failure
                print(f"  rendered {name}  ({dest.stat().st_size/1024:.0f} KB)")
                time.sleep(0.4)             # be polite to mermaid.ink
            else:
                print(f"  reuse    {name}")
            seen[body] = name
            n_replaced += 1
            return f"![diagram](assets/diagrams/{name})"

        new_src = _MERMAID_BLOCK.sub(_sub, src)
        cell["source"] = new_src
    return n_replaced


# ---------------------------------------------------------------------------
# Executing selected cells to bake outputs
# ---------------------------------------------------------------------------

def _cell_should_execute(src: str) -> bool:
    if any(s in src for s in NEVER_EXECUTE_IF_CONTAINS):
        return False
    return any(s in src for s in EXECUTE_IF_CONTAINS)


def execute_preview_cells(nb_path: Path) -> int:
    """Execute the notebook full-through (allow_errors=True), then save with
    embedded outputs. Cells that we know will fail (Colab uploads, the
    git-clone install) are temporarily blanked to no-ops so they don't poison
    state for the cells we DO want outputs from."""
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("  ! nbclient/nbformat not installed — skipping cell execution")
        print("    pip install nbclient nbformat ipykernel")
        return 0

    nb = nbformat.read(str(nb_path), as_version=4)

    # Stub out cells that don't make sense in a non-Colab batch context.
    n_stubbed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        # The §1.1 install cell tries `pip install -e .` which we already did
        # in the dev venv; let it run (it's a noop).
        # Stub the upload widget + the download-from-release cell (slow on a
        # cold machine; release weights already fetched if you ran §1.3 once).
        if any(s in src for s in NEVER_EXECUTE_IF_CONTAINS):
            cell.metadata.setdefault("ccdp", {})["original_source"] = cell.source
            cell.source = "# (skipped during pre-render — runtime-only cell)"
            n_stubbed += 1

    print(f"  stubbed {n_stubbed} runtime-only cells; executing the rest…")
    client = NotebookClient(
        nb, timeout=300, kernel_name="ccdp-dev",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=True,
    )
    try:
        client.execute()
    except Exception as e:
        print(f"  ! execution error: {e}")

    # Restore stubbed sources
    for cell in nb.cells:
        if cell.cell_type == "code" and "ccdp" in cell.metadata \
                and "original_source" in cell.metadata["ccdp"]:
            cell.source = cell.metadata["ccdp"].pop("original_source")
            if not cell.metadata["ccdp"]:
                cell.metadata.pop("ccdp")
            cell.outputs = []   # don't show stub output

    nbformat.write(nb, str(nb_path))
    n_with_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"  cells with saved outputs: {n_with_out}")
    return n_with_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", type=Path, default=NB_PATH,
                    help=f"Notebook to render (default: {NB_PATH.name}).")
    ap.add_argument("--skip-diagrams", action="store_true", help="Skip Mermaid rendering")
    ap.add_argument("--skip-execute", action="store_true", help="Skip cell execution")
    args = ap.parse_args()

    nb_path = args.notebook
    print(f"Reading {nb_path}")
    nb = json.loads(nb_path.read_text())

    if not args.skip_diagrams:
        print("\n[1/2] Rendering Mermaid diagrams to PNG…")
        n = replace_mermaid_blocks(nb)
        print(f"  replaced {n} mermaid blocks with image references")
        nb_path.write_text(json.dumps(nb, indent=1))

    if not args.skip_execute:
        print("\n[2/2] Executing preview cells to bake static outputs…")
        execute_preview_cells(nb_path)

    print(f"\nDone. {nb_path}")
    print(f"      {DIAGRAMS_DIR} ({sum(1 for _ in DIAGRAMS_DIR.glob('*.png'))} PNGs)")


if __name__ == "__main__":
    main()
