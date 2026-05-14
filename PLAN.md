# Car Crash Fix Amount Predictor — Capstone Project Plan

**Last updated:** 2026-05-12
**Owner:** Abhishek Roy
**Status:** Plan finalized, ready to scaffold

---

## 1. Project goal

Build a system that, given a photo of a damaged car (and optionally year/make/model), returns:
1. The damaged parts identified in the photo.
2. An estimated repair cost in USD or INR.
3. Provenance metadata describing how confident the system is and which data backed the estimate.

Two damage-recognition models will be trained and compared:
- **Variant A:** ResNet50 transfer-learning multi-label classifier (parts-as-labels).
- **Variant B:** YOLOv8 (Ultralytics) object detector for damaged-region bounding boxes.

Both feed a downstream XGBoost regressor that combines image features with tabular car metadata to predict cost. A pluggable, versioned **parts-cost catalog** lets the dollar estimate stay current without retraining the model.

---

## 2. Locked-in decisions

| # | Decision | Notes |
|---|---|---|
| 1 | Damage task: **multi-label whole-image classification (A)** + **YOLOv8 detection (B)** | Two variants, compared side-by-side |
| 2 | Cost target: **real datasets where available, synthetic-rule fallback** | Primary: ganeshsura. Secondary: iaai. Fallback: catalog-driven synthetic |
| 3 | Car metadata at training time: **synthesized** for damage-only datasets | Joined to real (year/make/model) where the cost-bearing datasets provide it |
| 4 | Hardware: **Apple Silicon M1/M2/M3, 16GB unified RAM** | PyTorch with MPS backend; Colab optional, not required |
| 5 | Currency normalization: **USD canonical**, with live INR conversion via FX module | Catalog records its native currency; FX rate fetched on demand |
| 6 | Detector framework: **Ultralytics YOLOv8** | AGPL-3.0 license accepted for capstone scope |
| 7 | Initial catalog seed: **data-driven from iaai + ganeshsura medians**, replaceable later | Versioned + auditable; updates create new timestamped catalogs |
| 8 | Make/model identification: **fine-tune a separate ResNet50 head on Stanford Cars 196** | Used to recover car identity from damage-only images |
| 9 | Deliverables: notebook + report PDF, modular Python package, FastAPI + Gradio demo, model registry | All four |
| 10 | FX provider: **exchangerate.host / open.er-api.com** (free, no API key) | Cached daily; manual override supported |

---

## 3. Datasets

> **2026-05-12 revision:** original assumptions about parts-level labels and per-image repair costs did not survive contact with the actual data. See [progress/phase_1_data_and_identification.md](progress/phase_1_data_and_identification.md) for the full diff. Below is the post-revision plan; full citations in [CITATIONS.md](CITATIONS.md).

### Damage-recognition training corpora

| Dataset | Labels | Role |
|---|---|---|
| `nasimetemadi/car-damage-detection` (CarDD) | 6 **damage TYPES**: dent, scratch, crack, glass shatter, lamp broken, tire flat — COCO segmentation | **Primary** corpus for both Variant A (multi-label) and Variant B (YOLOv8 detection) |
| `samwash94/comprehensive-car-damage-detection` | 6 location×condition folders: F/R × {Crushed, Normal, Breakage} | **Auxiliary head**: front/rear location + condition (severity proxy) |

### Metadata-bearing

| Dataset | Role |
|---|---|
| `rebrowser/iaai-dataset` (free sample, 12,353 rows) | **Car-metadata distribution source** (year/make/model/bodyStyle/vehicleClass/damage-location vocabulary). **Cost fields are paywalled** in the free sample — used for distributions only, not for cost supervision. |

### Auxiliary (car identification)

| Dataset | Role |
|---|---|
| `eduardo4jesus/stanford-cars-dataset` (Stanford Cars 196, Krause et al. 2013) | Phase 1.5 make/model/year identifier fine-tune |

### Datasets considered and **not used**

| Dataset | Why excluded |
|---|---|
| `ganeshsura/car-damage-detection-and-cost-estimation` | CSV-to-image join is broken (hashed Roboflow filenames vs sequential CSV keys); `est_cost` column is combinatorially synthetic, not real invoices. Catalog-driven synthetic cost is more transparent. |
| TartesiaDS (`github.com/tartesia/TartesiaDS`) | 108 images, 3 coarse categories, gated by Google Form, no parts/cost labels. Strictly inferior to CarDD for training. |
| `jessicali9530/stanford-cars-dataset` | Kaggle mirror requires rules-acceptance and returned 403; switched to `eduardo4jesus/stanford-cars-dataset`. |

### Cost-supervision honesty statement

No public dataset gives us (damage image → real repair invoice). Cost predictions in this project come from:

1. The **versioned parts-cost catalog** (transparent rule-based pricing, swappable via `ccdp costing import`).
2. The **calibrator** that scales model output by `active_median / training_median` when the catalog is updated.
3. The **three-tier degradation chain** (exact / nearest-class / catalog-only) with explicit provenance on every prediction.

The final report makes this constraint explicit. Once authoritative pricing tables or real invoice data become available, the catalog can be replaced and the existing trained model continues to work via the calibrator without retraining.

Notebook `01_eda.ipynb` prints actual schemas, sample sizes, and label distributions on first run.

---

## 4. Architecture overview

```
                          ┌─────────────── damage image ──────────────┐
                          │                                            │
              ┌───────────▼────────────┐              ┌────────────────▼──────────────┐
              │  Variant A:            │              │  Variant B:                   │
              │  ResNet50 multi-label  │              │  YOLOv8 detector              │
              │  classifier            │              │  (boxes + classes + areas)    │
              └───────────┬────────────┘              └────────────────┬──────────────┘
                          │ parts list                                 │ parts + boxes + areas
                          │                                            │
              ┌───────────▼────────────┐              ┌────────────────▼──────────────┐
              │  Image features        │              │  Image features + box stats   │
              │  (2048-d from ResNet)  │              │                               │
              └───────────┬────────────┘              └────────────────┬──────────────┘
                          │                                            │
                          │   ⊕ tabular metadata (year, make, model)   │
                          │                                            │
              ┌───────────▼────────────┐              ┌────────────────▼──────────────┐
              │  XGBoost(A)            │              │  XGBoost(B)                   │
              └───────────┬────────────┘              └────────────────┬──────────────┘
                          │ cost_A                                     │ cost_B
                          │                                            │
                          └────────────────┬───────────────────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Cost calibrator    │  (scales by active catalog
                                │                     │   vs training-time catalog)
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  FX module          │  (USD ↔ INR if requested)
                                └──────────┬──────────┘
                                           │
                                ┌──────────▼──────────┐
                                │  Response           │  parts, cost, provenance,
                                │                     │  catalog_id, fx_snapshot
                                └─────────────────────┘
```

Parallel to the damage models, a **car-identifier** model (ResNet50 on Stanford Cars) attempts to recover `(make, model, year)` from the image when not user-supplied, feeding the three-tier cost-estimation logic in §6.

---

## 5. Car-identification pipeline

Module: `src/ccdp/identification/`

Steps, run in order with confidence thresholds:

1. **Filename / folder hints** — many Kaggle datasets encode make/model in path.
2. **EXIF metadata** check.
3. **OCR on badge / plate regions** (EasyOCR) — high-precision, low-recall.
4. **Visual make/model classifier** — fine-tuned ResNet50 on Stanford Cars 196.
5. **Body-type / segment classifier** — coarse fallback (sedan / SUV / hatchback / truck / luxury / economy / mid).

Output per image: `{identified: bool, make, model, year, body_type, segment, confidence, source}`.

### Unidentified bucket

SQLite DB `data/unidentified_cars.db`:

| col | description |
|---|---|
| `image_id` | PK |
| `assigned_name` | auto-generated placeholder, e.g. `unknown_red_sedan_001` |
| `predicted_body_type` | from coarse classifier |
| `predicted_segment` | economy / mid / luxury heuristic |
| `user_supplied_make` | nullable, fillable later |
| `user_supplied_model` | nullable |
| `user_supplied_year` | nullable |
| `last_updated` | timestamp |

CLIs:
```
ccdp unidentified list
ccdp unidentified label --image-id X --make Honda --model Civic --year 2018
```

Gradio demo includes a **"Label this car"** tab. Newly labeled rows are queued for the next continued-training run.

### Reference table

Built from cost-bearing datasets:

```
make | model | year | body_type | segment | avg_cost | n_samples
```

Used by `nearest()` fallback (§6).

---

## 6. Three-tier cost estimation

```python
def estimate_cost(image, user_metadata=None, currency="USD", catalog_id=None):
    parts, severity_map = damage_model.predict(image)
    car_id = identifier.predict(image, user_metadata)
    catalog = costing.load(catalog_id or "active")

    if car_id.confidence >= HIGH:                            # Tier 1: exact match
        cost = xgb_exact.predict(features(image) + onehot(car_id))
        provenance = f"exact match: {car_id.make} {car_id.model} {car_id.year}"

    elif car_id.body_type or car_id.segment:                 # Tier 2: nearest class
        nearest = reference_table.nearest(car_id.body_type, car_id.segment)
        cost = xgb_class.predict(features(image) + onehot(nearest))
        provenance = f"approximated from {car_id.body_type}/{car_id.segment} (example: {nearest.example_model})"

    else:                                                    # Tier 3: catalog only
        cost = catalog.estimate(parts, severity_map)
        provenance = "category-only estimate from active catalog"

    cost = calibrator.scale(cost, catalog)                   # adjust for catalog drift
    if currency != "USD":
        cost = fx.convert(cost, "USD", currency)

    return {parts, cost, currency, provenance, catalog_id, fx_snapshot, tier, confidence_band}
```

Every prediction returns its tier; reports show the distribution across the test set and per-tier RMSE so reviewers see graceful degradation.

---

## 7. Pluggable, versioned parts-cost catalog

Module: `src/ccdp/costing/`

### Storage

```
data/parts_cost_catalog/
├── catalog_2026-05-12T14-30-00_initial.yaml
├── catalog_2026-06-01T09-00-00_q2-update.yaml
├── catalog_2026-08-15T11-22-00_user-import.yaml
└── active.yaml -> catalog_2026-08-15T11-22-00_user-import.yaml
```

### Catalog schema

```yaml
catalog_id: 2026-05-12T14-30-00_initial
created_at: 2026-05-12T14:30:00Z
created_by: abhishek
source: "data-driven medians from iaai + ganeshsura"
currency: USD
fx_snapshot:                          # if any non-USD source was converted
  USD_INR: 83.2
  fetched_at: 2026-05-12T14:25:00Z
parts:
  front_bumper:
    base_cost: { economy: 280, mid: 520, luxury: 1450 }
    severity_multiplier: { minor: 0.4, moderate: 1.0, severe: 1.8 }
    labor_hours: { minor: 1.5, moderate: 4, severe: 8 }
  hood:
    base_cost: { economy: 340, mid: 610, luxury: 1820 }
    ...
labor_rate_per_hour: { economy: 65, mid: 95, luxury: 165 }
notes: "Initial seed derived from dataset medians; replace with authoritative body-shop tables as available."
```

### How the catalog interacts with the trained model

- **Tier 3 fallback** reads the active catalog at inference — no retrain needed when costs change.
- **XGBoost** does *not* hard-code costs; it learns from training data and records the `training_catalog_id`. At inference, the **calibrator** scales predictions by `active.median / training.median` so a fresh catalog corrects stale model output without retraining.
- Every prediction response includes the `catalog_id` actually used.

### CLI

```
ccdp costing list                       # show versions + which is active
ccdp costing show <catalog_id>          # print catalog
ccdp costing activate <catalog_id>      # repoint active symlink
ccdp costing import --file new.csv      # build new catalog from CSV
ccdp costing import --from-dataset iaai # re-derive from dataset snapshot
ccdp costing diff <id_a> <id_b>         # per-part % change
```

Naming convention: `catalog_<ISO8601-utc>_<tag>.yaml`. Tag is free-form (e.g. `initial`, `q2-update`, `user-import`, `vendor-export-2026q3`).

### Gradio demo

Dropdown: `Parts cost catalog: [2026-08-15 user-import ▼]` — switching it re-prices the same image instantly.

---

## 8. FX (USD ↔ INR) module

Module: `src/ccdp/costing/fx.py`

```python
from ccdp.costing.fx import get_rate, refresh_rate

get_rate("USD", "INR")        # cached; warns if >24h old
refresh_rate("USD", "INR")    # forces live fetch
```

- **Primary source:** `exchangerate.host`
- **Fallback:** `open.er-api.com`, then `frankfurter.app`
- **Cache:** `data/fx_cache.json` → `{pair, rate, fetched_at, source}`
- **Offline:** `FX_OFFLINE=1` env var or `--fx-rate 83.2` CLI override; recorded as `fx_source: manual_override`
- **CLI:** `ccdp fx refresh`, `ccdp fx show`
- **API:** `GET /estimate?image=...&currency=INR&refresh_fx=true`

Every prediction response includes the FX rate + source + timestamp used. Reports show the FX snapshot under each converted figure.

---

## 9. Model variants A vs B — training & comparison

### Variant A — ResNet50 multi-label classifier

- Backbone: ImageNet-pretrained ResNet50.
- Head: Dense(512) → Dense(N_parts) with sigmoid.
- Stage 1: frozen backbone, ~5 epochs, batch 32, LR 1e-3.
- Stage 2: unfreeze layer3+layer4, ~15 epochs, LR 1e-4 with `ReduceLROnPlateau`.
- Augmentations: flip, rotate ±15°, color jitter, random crop.
- Loss: `BCEWithLogitsLoss` with positive-class weighting if imbalanced.
- Mixed precision via MPS.

### Variant B — YOLOv8 detector

- Backbone: `yolov8n` (start) → `yolov8s` if accuracy gap warrants the extra compute.
- Annotation source: bbox-annotated subsets of samwash94; pseudo-label remaining with high-confidence threshold from an initial trained model.
- 100 epochs default, ultralytics defaults for the rest.
- Extra features piped to XGBoost(B): `n_damage_regions`, `total_damaged_area_pct`, `largest_region_area_pct`, `area_pct_per_part`.

### Comparison report

`notebooks/05_model_comparison.ipynb` + a dedicated report chapter:

| Dimension | Variant A | Variant B |
|---|---|---|
| Parts ID macro-F1 / micro-F1 / per-class | | |
| Localization mAP@0.5 | n/a | |
| Cost regression: RMSE / MAE / MAPE / R² | | |
| Inference latency (M-series MPS, ms/image) | | |
| Model size (MB) | | |
| Training wall-clock | | |
| Tier 1 / 2 / 3 distribution on test set | | |
| Error slices: by severity, by segment | | |
| 10 qualitative side-by-side examples (Grad-CAM vs boxes) | | |

Report ends with a **recommendation** + **when-to-use** decision table.

API endpoint exposes `?model=resnet|yolov8|both` to run either or both.

---

## 10. Repository layout

```
car-damage-capstone/
├── PLAN.md                                ← this file
├── README.md
├── pyproject.toml
├── configs/                               # YAML configs per experiment
├── data/
│   ├── raw/                               # downloaded datasets (gitignored)
│   ├── processed/
│   ├── parts_cost_catalog/                # versioned catalogs (§7)
│   ├── unidentified_cars.db
│   └── fx_cache.json
├── src/ccdp/
│   ├── data/                              # loaders, augmentations, synthetic generators
│   ├── identification/                    # car-id pipeline (§5)
│   │   ├── reference_table.py
│   │   ├── car_identifier.py
│   │   ├── matcher.py
│   │   ├── unidentified.py
│   │   └── fallback_estimator.py
│   ├── costing/                           # versioned cost module (§7) + FX (§8)
│   │   ├── catalog.py
│   │   ├── fx.py
│   │   └── calibrator.py
│   ├── models/
│   │   ├── resnet_classifier.py           # Variant A
│   │   ├── yolov8_detector.py             # Variant B
│   │   ├── xgb_regressor.py
│   │   └── ensemble.py
│   ├── train/
│   │   ├── train_resnet.py
│   │   ├── train_yolov8.py
│   │   ├── train_xgb.py
│   │   ├── train_car_identifier.py
│   │   └── continue_training.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── comparison_report.py
│   │   └── report_generator.py            # writes PDF
│   ├── registry/                          # checkpoint + model versioning (§11)
│   ├── infer/                             # end-to-end pipeline
│   └── api/                               # FastAPI + Gradio
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 01b_dataset_join_and_identification.ipynb
│   ├── 02_train_classifier_resnet.ipynb
│   ├── 03_train_detector_yolov8.ipynb
│   ├── 04_train_regressor.ipynb
│   ├── 05_model_comparison.ipynb
│   └── 06_continued_training_demo.ipynb
├── scripts/
│   ├── download_datasets.sh
│   └── promote_model.py
├── checkpoints/                           # see §11
└── reports/                               # generated reports
```

---

## 11. Checkpointing, registry, and promotion

### Per-run checkpoint format (PyTorch)

`checkpoints/<variant>/run_<ISO-ts>_<tag>/`:
- `epoch_{N}.pt` — `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `metrics`, `rng_state`, `config_hash`, `training_catalog_id`, `git_commit`
- `last.pt` symlink → most recent
- `best.pt` symlink → best val metric
- `config.yaml` — frozen hyperparams
- `metrics.json` — per-epoch metrics

Resume:
```
python -m ccdp.train.train_resnet --resume checkpoints/resnet/run_X/last.pt
```
Re-hydrates optimizer/scheduler/RNG so the loss curve is continuous.

### YOLOv8 checkpoints

Ultralytics writes its own `runs/detect/train*/weights/{best,last}.pt`. Wrapped by the registry: symlinked into `checkpoints/yolov8/run_<ts>_<tag>/` with our metadata sidecar.

### XGBoost checkpoints

- `.ubj` format via `booster.save_model()`.
- Incremental: `xgb.train(..., xgb_model=prior_booster)` for new data.

### Continued training on new data

```
ccdp train continue \
  --base-checkpoint checkpoints/resnet/run_X/best.pt \
  --new-data data/raw/new_batch/ \
  --lr 1e-5 \
  --freeze-until layer3
```

- Lower LR + optional layer freezing to avoid catastrophic forgetting.
- New run dir; never overwrites prior best.
- Newly labeled rows from the unidentified bucket auto-included.

### Registry

```
checkpoints/
├── registry.json                          # canonical index
├── production/                            # symlinks to deployed models
│   ├── classifier.pt
│   ├── detector.pt
│   ├── identifier.pt
│   ├── xgb_a.ubj
│   └── xgb_b.ubj
├── resnet/run_<ts>_<tag>/
├── yolov8/run_<ts>_<tag>/
├── identifier/run_<ts>_<tag>/
└── xgb/run_<ts>_<tag>/
```

`registry.json` records per run: dataset hash, config hash, git commit, training catalog id, val metrics, train wall-clock, notes.

### Promotion

```
ccdp registry promote <run_id>
```
- Runs candidate vs current production on a frozen held-out test set.
- Writes a comparison report (metrics diff + per-class regression check).
- Flips the symlink only if candidate wins on the primary metric (macro-F1 for classifier, mAP@0.5 for detector, RMSE for regressor) AND does not regress >2% on any guard metric.
- Old production model stays in its run dir indefinitely (backup requirement).

Rollback: `ccdp registry rollback` — repoints production symlink to the previously active run.

---

## 12. Compute estimate (M-series 16GB)

| Stage | Wall-clock |
|---|---|
| ResNet50 damage classifier fine-tune, 20 epochs | ~2–4 hrs |
| YOLOv8n detector, 100 epochs on ~5k images | ~1–2 hrs |
| YOLOv8s detector, 100 epochs | ~3–5 hrs (only if needed) |
| Make/model identifier (Stanford Cars) | ~2–3 hrs |
| XGBoost(A) + XGBoost(B) | ~5 min total |
| **End-to-end first training** | **~8–14 hrs** spread over a few sessions |

All trainable locally on M-series with MPS. Colab is a useful backup for the YOLOv8s variant if you want a larger backbone later but **is not required**.

---

## 13. Evaluation & report

Auto-generated at `reports/report_<run_id>.pdf` and `reports/report_<run_id>.html`.

Sections:
1. **Dataset summary** — counts, class balance, synthetic vs real disclosure, schema reconciliation.
2. **Identification coverage** — % identified by tier, body-type distribution of unidentified, gap matrix (make × year × body_type).
3. **Training** — curves, hyperparameters, hardware/wall-clock, training catalog id.
4. **Variant A results** — per-class P/R/F1, confusion matrix, ROC/PR curves, Grad-CAM examples, regression metrics.
5. **Variant B results** — mAP@0.5, per-class AP, qualitative boxes, regression metrics.
6. **Comparison** — table from §9, recommendation, when-to-use decision table.
7. **Provenance breakdown** — tier distribution on test set, per-tier RMSE.
8. **Slice analyses** — luxury vs economy, mild vs severe damage.
9. **Failure cases** — annotated examples.
10. **Catalog & FX snapshot** — which catalog and FX rate the report was generated against.
11. **Unidentified bucket appendix** — count, examples, labeling workflow.
12. **Limitations** — synthetic-cost caveats prominently disclosed; estimates indicative, not insurable.

---

## 14. Implementation phases

| Phase | Scope | Duration |
|---|---|---|
| 0 | Scaffold repo, costing module, FX module, initial data-driven catalog seed | 1.5 days |
| 1 | Dataset join, identification module, reference table, unidentified bucket | 2 days |
| 1.5 | Make/model identifier fine-tune (Stanford Cars) | 1–2 days |
| 2A | ResNet50 classifier + XGBoost(A) + registry integration | 2 days |
| 2B | YOLOv8 detector + XGBoost(B) | 2 days |
| 3 | Comparison notebook + report generator + FastAPI + Gradio demo with catalog/FX/model switchers | 2 days |
| 4 | Continued-training + promotion workflow validation; final report | 1 day |

**Total:** ~12–14 working days.

---

## 15. CLI surface (consolidated)

```
# Data
ccdp data download
ccdp data eda

# Identification
ccdp unidentified list
ccdp unidentified label --image-id X --make M --model Mo --year Y

# Costing
ccdp costing list
ccdp costing show <catalog_id>
ccdp costing activate <catalog_id>
ccdp costing import --file new.csv
ccdp costing import --from-dataset iaai
ccdp costing diff <id_a> <id_b>

# FX
ccdp fx show
ccdp fx refresh

# Training
ccdp train resnet --config configs/resnet.yaml
ccdp train yolov8 --config configs/yolov8.yaml
ccdp train identifier --config configs/identifier.yaml
ccdp train xgb --variant {a,b}
ccdp train continue --base-checkpoint <path> --new-data <path>

# Registry
ccdp registry list
ccdp registry promote <run_id>
ccdp registry rollback
ccdp registry diff <id_a> <id_b>

# Inference
ccdp infer --image path.jpg --model {resnet,yolov8,both} --currency {USD,INR}

# Serving
ccdp serve api      # FastAPI on :8000
ccdp serve demo     # Gradio on :7860

# Reporting
ccdp report generate --run-id <id>
```

---

## 16. Open / deferred items

- **YOLOv8 license (AGPL-3.0):** accepted for capstone. If commercial use ever required, swap to RT-DETR or MMDetection Faster R-CNN.
- **Catalog authority:** initial seed is data-driven; mechanism exists to import authoritative body-shop tables later via `ccdp costing import`.
- **Identifier accuracy ceiling:** Stanford Cars covers 196 models through ~2012. If contemporary models matter, evaluate VMMRdb or a HuggingFace pretrained car classifier as an upgrade path.
- **Real cost dataset gaps:** if iaai + ganeshsura coverage proves thin in certain segments, Tier 3 catalog fallback covers it; gap will be visible in the coverage matrix in the report.

---

## 17. Definition of done

- [ ] All four notebooks runnable end-to-end on a fresh checkout.
- [ ] Both Variant A and Variant B models trained, registered, and promotable.
- [ ] Comparison report generated and committed.
- [ ] Gradio demo runs locally with catalog selector, FX refresh button, model selector.
- [ ] FastAPI endpoint serves predictions with full provenance fields.
- [ ] Continued-training demo notebook successfully extends a prior checkpoint on a held-out subset.
- [ ] Promotion workflow validated: a candidate run successfully promotes (or is rejected on a guard metric) on a deliberate test.
- [ ] Final PDF report includes all sections in §13.
- [ ] README documents quickstart, training, inference, catalog updates, and FX refresh.
