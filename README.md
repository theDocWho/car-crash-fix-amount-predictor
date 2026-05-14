# Car Crash Fix Amount Predictor (`ccdp`)

End-to-end capstone project that identifies damaged parts from car photos and estimates repair cost. Two damage-recognition variants are trained, compared, and routed through a calibrated cost pipeline:

- **Variant A** — ResNet50 multi-label classifier over the 6 CarDD damage types (no localization).
- **Variant B** — YOLOv8 detector + bounding-box-aware XGBoost regressor (with part localization).

Cost output is calibrated against a **versioned, swappable parts-cost catalog** so prices stay current without retraining the model.

See [PLAN.md](PLAN.md) for the full design and [progress/STATUS.md](progress/STATUS.md) for what's built.

---

## Current production metrics

| Model | Variant | Best metric | Run dir |
|---|---|---|---|
| Stanford Cars identifier (ResNet50 + RandAugment + MixUp + CutMix) | — | **val acc 77.0%** | `run_2026-05-13T14-30-04_identifier_v2` |
| CarDD damage classifier (ResNet50, 6 damage types) | A | **val macro-F1 0.834** | `run_2026-05-12T20-59-41_classifier_v1` |
| CarDD damage detector (YOLOv8n) | B | **mAP50 0.687 / mAP50-95 0.540** | `run_2026-05-13T05-18-40_yolov8n_v2` |
| XGBoost(A) — image features + tabular → cost | A | **val R² 0.630, test R² 0.642, MAPE 32.9%** | `run_2026-05-13T04-34-33_xgb_a_v1` |
| XGBoost(B) — A's features + bbox stats → cost | B | **val R² 0.716, test R² 0.736, MAPE 24.4%** | `run_2026-05-13T11-16-25_xgb_b_v2` |

Variant B beats Variant A by **+9 pts of R²** and **−8 pts of MAPE** — quantifying exactly how much the detection-derived bbox features improve cost regression.

Trained weights for all five models are attached to the [v0.1.0 release](https://github.com/theDocWho/car-crash-fix-amount-predictor/releases/tag/v0.1.0).

---

## Quickstart

```bash
# 1. Set up env (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,serve,dev]"

# 2. (macOS only) certifi + libomp shims
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export DYLD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")

# 3. Initialize the cost catalog
ccdp costing init

# 4. (Optional) Fetch live USD→INR rate
ccdp fx refresh

# 5. Either download datasets + train from scratch (~4 hrs), OR
#    grab the trained weights from the v0.1.0 release.
```

### Train from scratch

```bash
# Datasets (requires ~/.kaggle/access_token; ~10 GB total)
bash scripts/download_datasets.sh

# Train (sequential on a single MPS device; expect ~4 hrs total)
ccdp train identifier   --epochs-stage1 3 --epochs-stage2 12 --batch-size 32 --num-workers 4
ccdp train classifier   --epochs-stage1 3 --epochs-stage2 12 --batch-size 32 --num-workers 4
ccdp train detector     --epochs 50 --batch 16 --imgsz 640 --workers 4 --tag yolov8n_v1

# Promote (or use whichever run beats current production)
ccdp registry promote <id_run> identifier
ccdp registry promote <cls_run> classifier
ccdp registry promote <det_run> detector

# Build downstream features + train both XGBoost variants
ccdp train extract-features
ccdp train synth-targets
ccdp train extract-bbox-features
ccdp train xgb --variant a --n-estimators 600 --max-depth 7 --tag xgb_a_v1
ccdp train xgb --variant b --n-estimators 600 --max-depth 7 --tag xgb_b_v1
ccdp registry promote <xgb_a_run> xgb_a
ccdp registry promote <xgb_b_run> xgb_b
```

### Use pretrained weights (from the GitHub release)

Download the release assets into `checkpoints/production/`:

```bash
mkdir -p checkpoints/production
gh release download v0.1.0 -R theDocWho/car-crash-fix-amount-predictor -D checkpoints/production
```

### Inference

```bash
# Variant A (classifier-only)
ccdp infer path/to/car.jpg --model resnet \
    --make toyota --model-name camry --year 2019 --body-type sedan --currency USD

# Variant B (detector + bbox features)
ccdp infer path/to/car.jpg --model yolov8 \
    --make toyota --model-name camry --year 2019 --body-type sedan --currency INR

# Both side-by-side
ccdp infer path/to/car.jpg --model both --currency USD
```

Every prediction returns `damage_types`, `parts`, `cost_usd`, `tier`, `provenance`, `catalog_id`, and `fx_snapshot` — fully audited.

---

## What's in this repo

```
src/ccdp/
├── data/            # standardized Record schema + dataset loaders (CarDD, comprehensive, iaai, Stanford Cars)
├── identification/  # car-id pipeline (filename/EXIF/OCR/ML) + reference table + unidentified bucket
├── costing/         # versioned parts-cost catalog + FX module + calibrator
├── models/          # ResNet50 backbones (identifier + classifier) and XGBoost bundle
├── train/           # trainers, feature extractors, synthetic cost-target generator, mixup/cutmix
├── registry/        # checkpoint registry (run dirs, last/best symlinks, production/ symlinks)
├── infer/           # Variant A and Variant B end-to-end pipelines
└── cli.py           # `ccdp …` Typer CLI
```

See [PLAN.md §10](PLAN.md) for the full layout, [progress/](progress/) for per-phase status docs, and [CITATIONS.md](CITATIONS.md) for dataset attributions.

---

## Honest disclosures

Documented in detail in [PLAN.md §3](PLAN.md) and the phase docs under [progress/](progress/). Headline items:

1. **No real per-image repair cost data exists publicly.** The IAAI free sample paywalls its cost columns; the ganeshsura Kaggle dataset's cost column is combinatorially synthetic with a broken CSV-to-image join. We chose to make this explicit rather than hide it.
2. **Cost target is synthetic** — derived from `Catalog.estimate(parts × severity × segment)` plus age and ±10% Gaussian noise per row. The trained XGBoost learns this rule, and the **calibrator** scales predictions to whatever catalog is active at inference time so you can update prices without retraining.
3. **Parts-level damage labels are not in any public dataset.** Trainable labels are damage *type* (CarDD: dent/scratch/crack/glass_shatter/lamp_broken/tire_flat) and damage *location* (comprehensive: front/rear × normal/crushed/breakage). Parts are inferred at inference time from `(damage_type, bbox_center, body_type)` via `infer_part_from_damage()`.
4. **Cost predictions are not insurable quotes** — they're calibrated estimates with full provenance, intended for triage / first-line estimation.

When real cost data becomes available (e.g., a research-access slice from Rebrowser's IAAI dataset, or an authoritative body-shop pricing table), swap the catalog via `ccdp costing import` and existing models continue to work via the calibrator.

---

## License

MIT — see [LICENSE](LICENSE).

## Citations

All datasets and key dependencies are cited in [CITATIONS.md](CITATIONS.md). If you use this project academically, please cite it as:

> Roy, A. (2026). *Car Crash Fix Amount Predictor* [Software]. GitHub. https://github.com/theDocWho/car-crash-fix-amount-predictor
