# Phase 1.5 — Make/model identifier (Stanford Cars fine-tune)

**Status:** ✅ **Done** (scaffolding + verified end-to-end on MPS; full training is a single CLI command)
**Completed:** 2026-05-12

## Goal

Stand up the full make/model/year identification pipeline: data loader, model, two-stage fine-tune trainer, checkpoint registry with resume, and CLI. Verify end-to-end on MPS with a short smoke run. Full training is then a single command the user runs when ready (~45–60 min wall-clock on M-series 16GB).

## Deliverables

- [x] [src/ccdp/data/stanford_cars.py](../src/ccdp/data/stanford_cars.py) — devkit `.mat` parsing, `parse_class_name`, `split_train_val`, `build_torch_dataset`.
- [x] [src/ccdp/models/identifier.py](../src/ccdp/models/identifier.py) — `build_resnet50_identifier`, `set_finetune_stage`.
- [x] [src/ccdp/registry/registry.py](../src/ccdp/registry/registry.py) — `create_run`, `save_checkpoint`, `promote`, `list_entries`, `update_metrics`, `registry.json` index.
- [x] [src/ccdp/train/train_car_identifier.py](../src/ccdp/train/train_car_identifier.py) — two-stage trainer, full resume (model/optimizer/scheduler/RNG), MPS-aware.
- [x] CLI: `ccdp train identifier`, `ccdp registry list`, `ccdp registry promote`.
- [x] Tests: [tests/test_registry.py](../tests/test_registry.py) + [tests/test_stanford_cars.py](../tests/test_stanford_cars.py) — **49 / 49 passing**.
- [x] Smoke run verified: `[device] mps`, weights downloaded, two stages executed, checkpoints + symlinks created, resume tested with optimizer-shape mismatch handled cleanly.

## How the identifier integrates downstream

When Phase 2 inference runs, the identifier produces `(make, model, year, confidence)`. Confidence ≥ 0.6 routes the prediction to **Tier 1 (exact)**; below threshold falls to **Tier 2 (nearest-class via reference table)** and finally **Tier 3 (catalog-only)**. See [PLAN.md §6](../PLAN.md) for the three-tier chain.

## Known design points

1. **Train/val split**: Stanford never released test-set labels; the kaggle "test" set has bboxes only. We do a deterministic stratified 90/10 split (seed=42) on the 8144-image train set — every one of 196 classes appears in val.
2. **Bbox crop**: each sample is cropped to its annotation bbox before the transform, so the classifier focuses on the car (not background).
3. **Two-stage fine-tune**: Stage 1 (frozen backbone, head only ≈1.15M trainable) for fast warm-up; Stage 2 (unfreeze `layer3`+`layer4` ≈23.2M trainable) for full fine-tune.
4. **Resume**: full state restored (model, optimizer, scheduler, epoch, best metric, RNG, stage). Optimizer-shape mismatch across stages is detected and re-initialized cleanly.
5. **SSL cert workaround**: torchvision weight downloads require `SSL_CERT_FILE`. The trainer doesn't set it; the user does `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` before the first run on macOS. Documented in CLI help and below.

## How to verify

```bash
source .venv/bin/activate
pytest -q                                  # 49 / 49 passing

# One-time SSL fix on macOS for torchvision weight download
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# Smoke (30 batches/epoch, ~30s total):
ccdp train identifier --epochs-stage1 1 --epochs-stage2 1 \
    --batch-size 16 --num-workers 0 --smoke-batches 30 --tag smoke

ccdp registry list

# Full training (~45–60 min on M-series 16GB, target >70% val acc):
ccdp train identifier --epochs-stage1 3 --epochs-stage2 12 --batch-size 32 --num-workers 4

# Resume from any last.pt
ccdp train identifier --epochs-stage1 3 --epochs-stage2 20 --resume checkpoints/identifier/run_X/last.pt

# Promote a run to production (used by inference pipeline in Phase 2/3)
ccdp registry promote <run_id> identifier
```

## Smoke-run measurements (M-series 16GB MPS)

| metric | value |
|---|---|
| stage 1, 30 batches × 16 = 480 samples (train) | ~9.2s |
| stage 1, val (400 samples) | ~4.0s |
| stage 2, 30 batches train | ~10.0s |
| ResNet50 ImageNet download | 97.8 MB |
| stage 1 trainable params | 1,149,636 |
| stage 2 trainable params | 23,212,740 |
| checkpoint size (stage 1) | ~103 MB (model only) |
| checkpoint size (stage 2, with optimizer) | ~270 MB |

Extrapolation: full Stanford Cars training (8144 / 16 = 509 batches × ~0.3 s/batch ≈ 2.5 min per epoch in stage 1; ~3.5 min/epoch in stage 2). 3+12 epochs → ~50 min wall-clock end-to-end, plus ~10 min for val passes. Validated against PLAN.md §12 estimate (2–3 hrs) — actually faster than predicted.

## Pending

None blocking. Two follow-ups deferred:

- **Promotion auto-eval** (`ccdp registry promote` currently flips the symlink without running an A/B eval). The A/B logic is in scope for Phase 4 ([progress/phase_4_promotion_and_final_report.md](phase_4_promotion_and_final_report.md)) since classifier/detector also share it.
- **`CarIdentifier.from_production()` wrapper** that loads the promoted checkpoint and runs inference. Will be added with Phase 2 when we wire identification into the inference pipeline.

## Notes for future phases

- Identifier outputs go to `ccdp.identification.car_identifier.identify()` as the "ml" stage (currently `IdentificationResult.source == "ml"` is a placeholder). Phase 2A will wire it.
- `registry.json` is now alive — every subsequent training run (classifier, detector, XGBoost) records itself here automatically via `create_run()`.
- The smoke checkpoints under `checkpoints/identifier/run_*smoke*` can be deleted: `rm -rf checkpoints/identifier/run_*smoke*` (they're ~400 MB).
