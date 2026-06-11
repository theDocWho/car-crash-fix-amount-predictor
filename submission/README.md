# CCDP — Capstone submission

This folder is the self-contained submission deliverable.

## Contents

```
submission/
├── ccdp_submission.ipynb       # the canonical notebook (baked outputs ready)
├── requirements.txt            # python deps to run the notebook locally
├── test_images/                # 5 canned test cases (used by §12)
│   ├── 01_no_car.jpg
│   ├── 02_car_no_damage.jpg
│   ├── 03_single_car_damaged.jpg
│   ├── 04_multi_car_all_damaged.jpg
│   └── 05_multi_car_one_damaged.jpg
├── assets/diagrams/            # pre-rendered architecture diagrams
├── weights/                    # gitignored; populated on demand by §1.3
└── README.md                   # this file
```

## Just reviewing? No setup needed

Open `ccdp_submission.ipynb` on GitHub (or any Jupyter viewer) — every cell
already has baked outputs: training curves, per-class F1, dataset previews,
and the 5 canned test cases at §12. No environment, no weights, no datasets
required to review.

## Running it locally

```bash
git clone https://github.com/theDocWho/car-crash-fix-amount-predictor.git
cd car-crash-fix-amount-predictor
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r submission/requirements.txt
pip install -e .
jupyter lab submission/ccdp_submission.ipynb
```

Then, inside the notebook:

1. **§1.1** — run the setup cell. It finds the repo root automatically, so it
   works whether you launched Jupyter from the repo root or from `submission/`.
2. **§1.3 Production weights** — `RUN_FETCH = True` (default) downloads any
   missing weights from the v1.0.0 GitHub release into `submission/weights/`
   (~140 MB total). Already have them? Drop the `.pt`/`.ubj` files into
   `submission/weights/` or `checkpoints/production/` and the cell will pick
   them up without downloading.
3. **§1.5 Datasets (optional)** — only needed for the dataset-preview cells
   and the `RUN_TRAINING` cells; inference (§5–§12) needs only the weights.
   The fetch cell reports what's mounted. Flip `RUN_FETCH_DATA = True` to
   download what's missing (Stanford-Cars ~2 GB and CarDD ~5.7 GB via Kaggle —
   needs `~/.kaggle/kaggle.json`; carparts-seg ~133 MB direct). Or point
   `DATA_ROOT` at the folder where your copies already live.
4. Run the rest — all training cells are guarded by `RUN_TRAINING = False`
   and skip by default. §10 lets you upload your own image; §12 runs the five
   canned test cases.

## Running on Colab / Kaggle

Upload (or open) `ccdp_submission.ipynb` and run §1.1 — it detects the
platform, clones the repo, and installs everything. Then follow the same
§1.3 / §1.5 steps as above. A GPU runtime is only useful if you flip on the
`RUN_TRAINING` cells; inference runs fine on CPU.

## Final metrics (v1.0.0)

| Model | Metric | Value |
|---|---|---|
| Identifier (Stanford-Cars, 196 classes) | top-1 val | **0.7703** |
| Identifier — make-level (49 makes) | accuracy | **0.8541** |
| Damage seg (CarDD, nc=6) | Box mAP50 / Mask mAP50 | 0.712 / 0.711 |
| Parts seg (carparts-seg, 23 raw → 15 canonical) | Box mAP50 / Mask mAP50 | 0.704 / 0.714 |
| Variant D end-to-end | n=20 smoke holdout MAE | see §11.2 |

See §2.4 for the honest comparison vs. the VMMRdb 1163-class extension we
explored but did not ship (preserved on draft PRs #31 / #32 and in
`notebooks/legacy/ccdp_submission_vmmrdb.ipynb`).
