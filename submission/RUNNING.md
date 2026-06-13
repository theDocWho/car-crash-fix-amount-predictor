# Running the CCDP submission

This guide gets `ccdp_submission.ipynb` running end-to-end **locally** and on
**Colab / Kaggle**. The notebook ships with every output already baked, so you
can also just read it on GitHub with no setup at all.

> **TL;DR for reviewers:** open `ccdp_submission.ipynb` — all training curves,
> metrics, architecture summaries, and the 5 canned test cases are already
> visible. To re-run, follow one of the two paths below.

---

## What you need (and what's bundled vs fetched)

| Artifact | Size | Shipped in the zips? | How it's obtained |
|---|---|---|---|
| Code (`src/ccdp`, `scripts`) | ~1.5 MB | ✅ Bundle 1 | unzipped |
| Notebook + diagrams + report | ~12 MB | ✅ Bundle 2 | unzipped |
| Sample/test images + eval CSVs | ~2 MB | ✅ Bundle 3 | unzipped |
| **Model weights** (`identifier.pt` 103 MB, `classifier.pt` 270 MB, …) | ~390 MB | ❌ (exceed the 50 MB cap) | §1.3 downloads from the v1.0.0 GitHub release |
| **Datasets** (Stanford 2 GB, CarDD 5.7 GB, carparts-seg 133 MB) | ~8 GB | ❌ | §1.5 fetch cell (opt-in) |

Weights and datasets are **not** in the zips on purpose — a single weight file
is larger than the 50 MB per-zip limit. The notebook fetches them on demand and
also runs read-only from the baked outputs if you fetch nothing.

The three bundles unzip into the **same folder** and reconstruct this layout:

```
car-crash-fix-amount-predictor/
├── pyproject.toml  requirements.txt          (Bundle 1)
├── src/ccdp/  scripts/                        (Bundle 1)
├── submission/
│   ├── ccdp_submission.ipynb                  (Bundle 2)
│   ├── CCDP_Project_Report.docx               (Bundle 2)
│   ├── assets/diagrams/                        (Bundle 2)
│   ├── requirements.txt  README.md  RUNNING.md (Bundle 2)
│   ├── test_images/                            (Bundle 3)
│   └── weights/                                (populated by §1.3)
└── data/
    ├── eval/                                   (Bundle 3 — small CSV/JSON)
    └── raw/                                    (populated by §1.5)
```

---

## Path A — Local (Jupyter)

Python **3.11+** recommended. From the reconstructed project root:

```bash
# 1. create an environment
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 2. install dependencies (notebook + ML stack)
pip install -r submission/requirements.txt
pip install -e .                                       # installs the `ccdp` package

# 3. (macOS only) xgboost needs OpenMP:
#    brew install libomp

# 4. launch
jupyter lab submission/ccdp_submission.ipynb
```

Inside the notebook, run top to bottom:

1. **§1.1 Setup** — auto-detects the repo root and `chdir`s there, so relative
   paths resolve whether you launched Jupyter from the repo root *or* from
   `submission/`. Installs `ccdp` if it isn't already importable.
2. **§1.3 Weights** — `RUN_FETCH = True` (default) downloads only the missing
   weights from the v1.0.0 release into `submission/weights/`. Already have
   them? Drop the `.pt` / `.ubj` files into `submission/weights/` (or
   `checkpoints/production/`) and the cell skips the download.
3. **§1.5 Datasets (optional)** — needed only for the dataset-preview grids and
   the `RUN_TRAINING` cells; **inference (§5–§12) needs only the weights**. The
   fetch cell reports what's mounted. To download what's missing set
   `RUN_FETCH_DATA = True` (Stanford + CarDD via Kaggle — see credentials
   below; carparts-seg via a direct 133 MB zip). To use copies you already
   have, set `DATA_ROOT` to the folder that contains them.
4. Everything else runs on CPU. `RUN_TRAINING` cells are guarded `False` and
   skip by default — flip them on only on a GPU.

### Kaggle credentials (only if you fetch Stanford/CarDD)
Create an API token at kaggle.com → Account → "Create New API Token", then
either place `kaggle.json` at `~/.kaggle/kaggle.json` (`chmod 600`), or export
`KAGGLE_USERNAME` and `KAGGLE_KEY` before launching Jupyter.

### Loading datasets from a folder you already have
Set `DATA_ROOT` in the §1.5 fetch cell to any directory laid out as:
```
<DATA_ROOT>/stanford-cars-dataset/cars_train/cars_train/*.jpg
<DATA_ROOT>/car-damage-detection/CarDD_release/CarDD_COCO/{train2017,val2017,test2017}
<DATA_ROOT>/carparts-seg/images/{train,val,test}
```

---

## Path B — Colab / Kaggle

You only need the notebook file; the code is cloned from GitHub by the first
cell.

1. **Colab:** File → Upload notebook → pick `ccdp_submission.ipynb`
   (or open it directly from the GitHub URL).
   **Kaggle:** Create → New Notebook → File → Import → upload the `.ipynb`.
2. Run **§1.1** — it detects Colab/Kaggle, clones the repo, runs
   `pip install -e .[ml]`, and appends `src/` to `sys.path`.
3. Run **§1.3** to fetch weights from the release (works without any
   credentials).
4. **§1.5** datasets: on Kaggle, add the datasets via "Add Input" and point
   `DATA_ROOT` at `/kaggle/input/...`, or set `RUN_FETCH_DATA = True`. On Colab,
   set `RUN_FETCH_DATA = True` (upload your `kaggle.json` first for the Kaggle
   sources).
5. A **GPU runtime** is only useful for the `RUN_TRAINING` cells; inference and
   all five test cases run fine on CPU.

---

## Reproducing the trained models (optional, GPU)

Each section's training cell is guarded by `RUN_TRAINING = False` with smoke
defaults. To reproduce the released weights, set it `True` on a GPU runtime:

| Model | Command (or notebook cell) | ~time on a T4 |
|---|---|---|
| Identifier (ResNet-50, 196-class) | `ccdp train identifier --dataset stanford --epochs-stage1 3 --epochs-stage2 22 --batch-size 32` | 3–4 h |
| Damage seg (YOLOv8n-seg) | §3.2 cell (`epochs=80, batch=16, imgsz=640`) | ~2 h |
| Parts seg (YOLOv8n-seg) | §4 cell (`epochs=80, batch=16, imgsz=640`) | ~2 h |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: ccdp` right after install (Colab) | re-run §1.1; it appends `src/` to `sys.path`. |
| `xgboost ... libomp.dylib not loaded` (macOS) | `brew install libomp`. |
| `SSL: CERTIFICATE_VERIFY_FAILED` in §1.3 | `pip install certifi`; the cell already falls back to `curl`. |
| "not mounted locally" in a preview cell | dataset isn't downloaded — set `RUN_FETCH_DATA = True` or point `DATA_ROOT` at your copy (inference doesn't need this). |
| Weights cell does nothing | `RUN_FETCH` is `False`, or the files already exist in `submission/weights/`. |
