# CCDP — Capstone submission

This folder is the self-contained submission deliverable.

## Contents

```
submission/
├── ccdp_submission.ipynb       # the canonical notebook (baked outputs ready)
├── test_images/                # 5 canned test cases
│   ├── 01_no_car.jpg
│   ├── 02_car_no_damage.jpg
│   ├── 03_single_car_damaged.jpg
│   ├── 04_multi_car_all_damaged.jpg
│   └── 05_multi_car_one_damaged.jpg
├── weights/                    # gitignored; populated on demand by §1.3
└── README.md                   # this file
```

## How to read this notebook

Open `ccdp_submission.ipynb` on GitHub — every cell already has baked
outputs (training curves, F1 plots, per-class metrics, and the 5 canned
test cases at §12). No environment, no weights, no datasets required to
review.

## If you want to run it

1. Open in Colab or local Jupyter (`ccdp-dev` kernel).
2. In **§1.3 Fetch released production weights**, flip `RUN_FETCH = True`
   and run. Weights land in `submission/weights/` (~110 MB identifier +
   ~30 MB everything else; ~140 MB total once `classifier.pt` is also
   pulled).
3. Run the rest of the cells — all training cells are guarded by
   `RUN_TRAINING = False` and skip by default.

## Final metrics (v1.0.0)

| Model | Metric | Value |
|---|---|---|
| Identifier (Stanford-Cars, 196 classes) | top-1 val | **0.7703** |
| Identifier — make-level (49 makes) | accuracy | **0.8541** |
| Damage seg (CarDD, nc=6) | Box mAP50 / Mask mAP50 | 0.712 / 0.711 |
| Parts seg (carparts, nc=15) | Box mAP50 / Mask mAP50 | 0.704 / 0.714 |
| Variant D end-to-end | n=20 smoke holdout MAE | see §11.2 |

See §2.4 for the honest comparison vs. the VMMRdb 1163-class extension we
explored but did not ship (preserved on draft PRs #31 / #32 and in
`notebooks/legacy/ccdp_submission_vmmrdb.ipynb`).
