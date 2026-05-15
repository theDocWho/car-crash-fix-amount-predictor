# Beginner notebooks

A guided, math-from-first-principles tour through every model in the project. Each notebook is self-contained, runnable on **Google Colab** (free CPU or T4) or your laptop, and exercises the *real* `ccdp` package — the cells aren't toy reimplementations.

## Run order

| # | Notebook | Open in Colab |
|---|---|---|
| 00 | [Overview](00_overview.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/00_overview.ipynb) |
| 01 | [Data & preprocessing](01_data_and_preprocessing.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/01_data_and_preprocessing.ipynb) |
| 02 | [ResNet50 classifier](02_classifier_resnet50.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/02_classifier_resnet50.ipynb) |
| 03 | [YOLOv8 detector](03_detector_yolov8.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/03_detector_yolov8.ipynb) |
| 04 | [XGBoost cost regressor](04_cost_regressor_xgboost.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/04_cost_regressor_xgboost.ipynb) |
| 05 | [Metrics deep dive](05_metrics_deep_dive.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/05_metrics_deep_dive.ipynb) |
| 06 | [End-to-end inference](06_end_to_end_inference.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theDocWho/car-crash-fix-amount-predictor/blob/main/notebooks/06_end_to_end_inference.ipynb) |

## What each notebook teaches

- **00 — Overview**: the pipeline as a whole, what each model contributes, project layout.
- **01 — Data & preprocessing**: dataset sources, why downscale (memory math), LANCZOS resampling, variance-of-Laplacian blur scoring, MixUp / CutMix visualised.
- **02 — Classifier**: convolution from scratch, the residual trick, two-stage transfer learning, BCE-with-logits, runnable training cell.
- **03 — Detector**: bounding box conventions, IoU, NMS, anchor-free YOLOv8 grid output, mAP step-by-step, runnable smoke test.
- **04 — Cost regressor**: gradient boosting math, why trees beat NNs on tabular data, the catalog **calibration** trick.
- **05 — Metrics**: precision, recall, F1, RMSE, MAE, MAPE, R² — derived and verified against `ccdp.eval.metrics`.
- **06 — End-to-end**: load production weights, run the full pipeline, draw boxes on a real image.

## Demo-scale vs full training

Every training cell defaults to a **demo-scale** run (a few epochs, tiny batch, often synthetic data) so the whole notebook completes in under 10 minutes on free Colab. A clearly-marked "FULL TRAINING" cell shows what to uncomment to launch the multi-hour production run on the real datasets.

## Re-generating the notebooks

The notebooks are produced by `_build_notebooks.py` so the source-of-truth is plain Python rather than messy ipynb JSON. To tweak content:

```bash
# edit notebooks/_build_notebooks.py
python notebooks/_build_notebooks.py
git add notebooks/*.ipynb
```

## Kaggle credentials on Colab

Notebooks 02 / 03 / 04 can do **full** training on the real CarDD + Stanford Cars datasets. To download those, you need a Kaggle API token:

1. Go to <https://www.kaggle.com/settings>, scroll to "API", click "Create New Token". A `kaggle.json` will download.
2. In Colab, run:

```python
from google.colab import files
files.upload()  # pick kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

3. The notebook's data-fetch cell will then succeed.

Demo-scale cells use synthetic tensors and need **no** Kaggle account.
