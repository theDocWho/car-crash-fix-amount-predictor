# Phase 2A — ResNet50 multi-label damage-type classifier + XGBoost(A)

**Status:** ✅ **Done** (scaffold + verified end-to-end on MPS; full training is a sequence of CLI commands)
**Completed:** 2026-05-13

## Goal

Build the full **Variant A** stack: a ResNet50 multi-label classifier over the 6 CarDD damage types, a 2048-d feature extractor that streams every CarDD image into a parquet cache, a synthetic cost-target generator that joins iaai metadata distributions to a catalog-derived per-image cost, and an XGBoost(A) regressor on top. Wrap it in `VariantAPipeline` so a single image + optional metadata produces a complete prediction (damage types, parts, cost, tier, provenance).

## Deliverables

- [x] [src/ccdp/data/damage_dataset.py](../src/ccdp/data/damage_dataset.py) — multi-hot label encoding, deterministic 80/10/10 splits, inverse-frequency `pos_weight`.
- [x] [src/ccdp/models/damage_classifier.py](../src/ccdp/models/damage_classifier.py) — ResNet50 + dropout/512/dropout/6 head, `set_finetune_stage`, `extract_features`.
- [x] [src/ccdp/train/train_damage_classifier.py](../src/ccdp/train/train_damage_classifier.py) — two-stage trainer, BCE+pos_weight, per-class P/R/F1, macro/micro F1, full resume.
- [x] [src/ccdp/train/extract_features.py](../src/ccdp/train/extract_features.py) — runs trained backbone over all CarDD images, writes parquet with `image_id, split, damage_types, f_0..f_2047`.
- [x] [src/ccdp/train/synthesize_cost.py](../src/ccdp/train/synthesize_cost.py) — samples metadata from iaai pool, maps damage types → parts, computes catalog-driven cost with age + noise factors. Outputs `cost_source = synthetic@<catalog_id>` per row for traceability.
- [x] [src/ccdp/models/xgb_regressor.py](../src/ccdp/models/xgb_regressor.py) — `XGBRegressorBundle` (feature schema + training catalog id) + `make_feature_matrix` for reproducible one-hot encoding at inference.
- [x] [src/ccdp/train/train_xgb.py](../src/ccdp/train/train_xgb.py) — joins features+targets on `image_id`, trains XGBoost(A), reports RMSE/MAE/MAPE/R² on val and test, saves `best.ubj` + `bundle.json`.
- [x] [src/ccdp/infer/variant_a.py](../src/ccdp/infer/variant_a.py) — `VariantAPipeline` (image → classifier → optional XGBoost → cost). Falls back gracefully to the three-tier catalog estimator when XGBoost or metadata is unavailable. Currency-converts via the FX module.
- [x] CLI: `ccdp train classifier`, `ccdp train extract-features`, `ccdp train synth-targets`, `ccdp train xgb`, `ccdp infer`.
- [x] Tests: [tests/test_damage_classifier.py](../tests/test_damage_classifier.py) — **54 / 54 passing**.
- [x] End-to-end smoke verified on MPS.

## Smoke-run results (MPS, M-series 16GB)

Classifier (1 + 1 epochs of 25 batches × 16):

| metric | value |
|---|---|
| stage 1, ~9 s train + 6 s val on 400 samples | val macro-F1 = 0.61 |
| stage 2, ~10 s train + 6 s val on 400 samples | **val macro-F1 = 0.68, micro-F1 = 0.68** |
| pos_weight (auto) | dent 1.3, scratch 1.0, crack 5.6, glass_shatter 5.1, lamp_broken 5.0, **tire_flat 12.2** |
| trainable params | stage 1: 1,052,166  •  stage 2: 23,115,270 |

Feature extraction on 8 batches × 32 (768 images total): **~14 s**. Full corpus (4000 images): ~75 s extrapolated.

XGBoost(A) on those 768 features:

| split | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| val | $392.99 | $283.92 | 51.6% | 0.50 |
| test | $472.70 | $323.25 | 55.6% | 0.53 |

End-to-end `ccdp infer` on a real CarDD val image, user-supplied metadata `--make honda --model-name civic --year 2018 --body-type sedan`:
```
damage_types: ["dent", "scratch"]
cost_usd:     $642.31
tier:         "exact"
provenance:   "xgb_a(exact); training_catalog=2026-05-12T05-45-11_initial;
              calibrated to 2026-05-12T05-45-11_initial"
probabilities: {dent: 0.55, scratch: 0.72, crack: 0.41,
                glass_shatter: 0.08, lamp_broken: 0.31, tire_flat: 0.02}
```

## How to verify

```bash
source .venv/bin/activate
pytest -q                                                # 54 / 54 passing

# One-time macOS env (torchvision weights + xgboost OMP)
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export DYLD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")

# Smoke (~30 s total)
ccdp train classifier --epochs-stage1 1 --epochs-stage2 1 --batch-size 16 \
    --num-workers 0 --smoke-batches 25 --tag smoke
CLS=$(ls checkpoints/classifier/run_*smoke*/best.pt | head -1)
ccdp train extract-features --checkpoint "$CLS" --batch-size 32 --num-workers 0 --smoke-batches 8
ccdp train synth-targets
ccdp train xgb --n-estimators 80 --max-depth 5 --tag xgb_a_smoke

# Inference
IMG=$(find data/raw/car-damage-detection -name '*.jpg' | head -1)
ccdp infer "$IMG" --make honda --model-name civic --year 2018 --body-type sedan

# Full training (~25 min wall-clock target):
ccdp train classifier --epochs-stage1 3 --epochs-stage2 12 --batch-size 32 --num-workers 4
ccdp registry promote <classifier_run_id> classifier
ccdp train extract-features
ccdp train synth-targets
ccdp train xgb --n-estimators 600 --max-depth 7 --learning-rate 0.05 --tag xgb_a_v1
ccdp registry promote <xgb_run_id> xgb_a
```

## Honest disclosure (relevant for the report)

- The cost target is **synthetic**, derived from `Catalog.estimate(parts_with_severity, segment)` × age factor × ±10% Gaussian noise per row. There is no real per-image repair invoice data available publicly (see [progress/phase_1_data_and_identification.md](phase_1_data_and_identification.md) and [PLAN.md §3](../PLAN.md)).
- Tabular features (make/model/year/body_type) are **sampled** from the iaai metadata distribution, not derived from each CarDD image. They give XGBoost realistic feature correlations but they do not describe the specific car in the image.
- Each prediction records `training_catalog_id` and `bundle_run_id` in its provenance string. When the catalog is updated later, the `Calibrator` scales output by `active.median / training.median` so the trained XGBoost remains useful **without retraining**.
- Variant A produces **no part localization** — bbox-derived features arrive in Variant B (Phase 2B). The `parts: []` field in inference responses is honest about this limitation; XGBoost still produces a calibrated cost.

## Pending

None blocking. Two follow-ups deferred:

- **Promotion auto-eval** is still the simple symlink flip from Phase 1.5 — the A/B comparison harness lives with Phase 4.
- **The training-time catalog is the only one ever seen by XGBoost.** Until Phase 4, calibration coverage is small (one catalog). When a real second catalog lands, validate the scaling factor against a held-out test set before relying on it.

## Notes for future phases

- **Phase 2B** (YOLOv8 detector) shares `XGBRegressorBundle` — just appends bbox-stat features (`n_damage_regions`, `total_damaged_area_pct`, per-part area dict) to the same image-feature row and trains a separate XGBoost(B). The Variant A → B comparison in Phase 3 reuses the metrics already produced by `train_xgb.py`.
- **Phase 3** (comparison + serving) will read both registered runs (classifier + xgb_a) and run side-by-side eval on the held-out test split that `damage_dataset.split_records(..., seed=42)` produces — already deterministic so both variants compare on the same images.
- **Phase 4** (promotion + continued training) will gate `ccdp registry promote` behind an A/B test against the current production model. The hook is already in place: `production_target('classifier')` and `production_target('xgb_a')` return the active symlink target.
