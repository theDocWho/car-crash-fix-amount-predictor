# Phase 2B — YOLOv8 damage detector + XGBoost(B)

**Status:** ✅ **Done** (scaffold + verified end-to-end on MPS; full training is a CLI sequence)
**Completed:** 2026-05-13

## Goal

Build **Variant B**: a YOLOv8 detector that localizes each damaged region with a bounding box + damage-type label, plus an XGBoost(B) cost regressor that consumes the same 2048-d image features as Variant A *plus* bbox-derived features (region count, per-class area, largest region). Wrap end-to-end in `VariantBPipeline` and extend `ccdp infer` to support `--model yolov8 | both`.

## Deliverables

- [x] [src/ccdp/data/cardd_yolo.py](../src/ccdp/data/cardd_yolo.py) — CarDD → Ultralytics YOLO layout (deterministic 80/10/10 split shared with Variant A); `data.yaml` writer.
- [x] [src/ccdp/train/train_yolov8.py](../src/ccdp/train/train_yolov8.py) — thin Ultralytics wrapper, registry integration (`variant="detector"`), absolute-path fix so artifacts land under `checkpoints/detector/run_*` not `runs/detect/`.
- [x] [src/ccdp/train/extract_bbox_features.py](../src/ccdp/train/extract_bbox_features.py) — two paths: `extract_from_ground_truth` (no detector, upper-bound eval) and `extract_with_detector` (runs YOLOv8 inference over CarDD and aggregates per-image stats).
- [x] [src/ccdp/train/train_xgb.py](../src/ccdp/train/train_xgb.py) — extended to merge bbox features when `variant=b`; registry variant naming generalized to `xgb_<variant>`.
- [x] [src/ccdp/infer/variant_b.py](../src/ccdp/infer/variant_b.py) — `VariantBPipeline`: image → YOLOv8 → bboxes → `infer_part_from_damage(damage_type, bbox_center, location)` → XGBoost(B) → calibrated cost + FX.
- [x] CLI: `ccdp train detector`, `ccdp train build-yolo-dataset`, `ccdp train extract-bbox-features [--gt]`, `ccdp train xgb --variant b`, `ccdp infer --model yolov8|both`.
- [x] Tests: [tests/test_yolo_and_bbox.py](../tests/test_yolo_and_bbox.py) — **59 / 59 passing** overall.

## Smoke-run results (MPS, M-series 16GB)

### YOLOv8n smoke (1 epoch, batch 8, imgsz 320)

- 3200 train + 400 val images, ~3 minutes wall-clock
- Validation: P=0.49, R=0.30, **mAP50=0.30, mAP50-95=0.20**
- Best.pt symlink correctly created under `checkpoints/detector/run_<ts>_smoke/`

### XGBoost(B) on ground-truth CarDD bboxes (80 trees on 768 train/val/test)

| variant | val RMSE | val R² | val MAPE | test RMSE | test R² |
|---|---|---|---|---|---|
| A (image features only) | $393 | 0.50 | 51.6% | $473 | 0.53 |
| **B (image + bbox stats)** | **$305** | **0.70** | **29.6%** | **$381** | **0.69** |

Bbox features deliver a meaningful improvement: **−22% RMSE, +0.20 R², halved MAPE** on val.

### End-to-end `ccdp infer --model both` on a real CarDD val image

```
variant_a:
  damage_types:  [dent, scratch]
  cost_usd:      $642.31
  tier:          exact
  parts:         [] (no localization in A)

variant_b:
  damage_types:  [dent, scratch]
  n_detections:  2
  parts:         [rear_bumper, trunk]    ← localized from YOLO bbox positions
  cost_usd:      $770.71
  tier:          exact
  provenance:    xgb_b(exact); training_catalog=...; calibrated to ...
```

## How to verify

```bash
source .venv/bin/activate
pytest -q                                                # 59 / 59 passing

# One-time macOS env
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export DYLD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")

# Smoke (~5 minutes total)
ccdp train build-yolo-dataset
ccdp train extract-bbox-features --gt
ccdp train xgb --variant b --n-estimators 80 --max-depth 5 --tag xgb_b_smoke
ccdp train detector --epochs 1 --batch 8 --workers 0 --imgsz 320 --tag smoke

# Inference (both variants side-by-side)
IMG=$(find data/raw/car-damage-detection -name '*.jpg' | head -1)
ccdp infer "$IMG" --model both --make honda --model-name civic --year 2018 --body-type sedan

# Full training (~2 hrs wall-clock target):
ccdp train detector --epochs 50 --batch 16 --imgsz 640 --workers 4 --tag yolov8n_v1
ccdp registry promote <detector_run_id> detector
ccdp train extract-bbox-features                         # uses promoted detector
ccdp train xgb --variant b --n-estimators 600 --max-depth 7 --tag xgb_b_v1
ccdp registry promote <xgb_b_run_id> xgb_b
```

## Design points worth remembering

1. **Shared split with Variant A.** `cardd_yolo.build()` reuses `damage_dataset.split_records(..., seed=42)` so A and B train and evaluate on the same images. Variant comparisons in Phase 3 are apples-to-apples.
2. **Image features come from the classifier backbone, not the detector.** Variant B reuses Variant A's 2048-d image embedding (so XGBoost(B) feature schema is a strict superset of XGBoost(A)'s). Adds only the bbox-derived columns (counts + areas).
3. **GT-bbox path is real.** `extract_bbox_features.py --gt` uses CarDD's hand-annotated bboxes. Useful as an **upper bound** on how much bbox info would help if the detector were perfect — gives us a calibration point during model comparison.
4. **Ultralytics paths fixed.** Passing `project=` as an *absolute* path stops ultralytics from prepending `runs/detect/`. Artifacts land where the registry expects them.
5. **Part inference uses bbox center + body-type.** Per-detection: `infer_part_from_damage(damage_type, bbox_center=(xc, yc), damage_location=metadata.body_type)`. This is how Variant B populates the `parts` field that Variant A leaves empty.

## Pending

None blocking. Two open items move to later phases:

- **Pseudo-labeling for low-recall classes** (e.g. `tire_flat` had 319 bboxes total). For now we trust CarDD's annotations. If mAP50 < 0.4 after a full training run, consider a high-confidence pseudo-label pass.
- **YOLOv8s upgrade** if the YOLOv8n full run trails Variant A by >5% on cost RMSE — currently optional.

## Notes for Phase 3 (comparison report + serving)

- The Phase 3 comparison notebook will reuse the same test split (last 400 records of the seed=42 shuffle). XGBoost bundles for `xgb_a` and `xgb_b` both record `training_catalog_id`, so cost predictions are auto-calibratable to whatever catalog is active at inference time.
- `ccdp infer --model both` is the building block for the side-by-side qualitative grid (10 examples per the report spec).
- FastAPI + Gradio in Phase 3 will instantiate both pipelines at startup; switching catalog / FX rate in the UI re-prices without re-running the vision models.
