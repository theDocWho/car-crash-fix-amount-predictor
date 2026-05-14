# Phase 1 — Datasets, EDA, join, identification, reference table, unidentified bucket

**Status:** ✅ **Done** (with a documented pivot — see "Pivot" section below)
**Completed:** 2026-05-12

## Goal

Bring real datasets onto disk, reconcile their actual schemas with what `PLAN.md` assumed, build the reference table, and stand up the unidentified-cars bucket with auto-naming and a labeling CLI.

## Pivot (vs original Phase 1 plan)

After inspecting real data, three of the four original "cost-bearing" assumptions did not survive contact with the data:

1. **iaai's cost columns are paywalled** in the free sample — all 12,353 rows contain the literal string `"[PREMIUM]"` for `estimatedRepairCost`, `buyNowPrice`, `vin`, `imageUrl`, etc. iaai is therefore used **for metadata distributions only**, not for cost supervision.
2. **ganeshsura** has a broken CSV-to-image join (hashed Roboflow filenames vs sequential CSV keys) and a combinatorially-synthetic `est_cost` column. Dropped from the pipeline entirely.
3. **No dataset has parts-level labels** at the 15-part taxonomy in the seed catalog. Trainable labels are:
   - CarDD's **6 damage TYPES** (dent, scratch, crack, glass_shatter, lamp_broken, tire_flat)
   - comprehensive's **front/rear × {normal, crushed, breakage}** condition labels

The architecture stayed intact (ResNet50 / YOLOv8 / XGBoost / 3-tier estimator / catalog). What changed: the *model output* is damage TYPE + LOCATION, and a heuristic mapping `infer_part_from_damage(damage_type, bbox_center, location)` bridges to the parts-keyed cost catalog. See [PLAN.md §3](../PLAN.md) for the revised dataset table and [CITATIONS.md](../CITATIONS.md) for the up-to-date citations.

## Done

- [x] [scripts/download_datasets.sh](../scripts/download_datasets.sh) — Kaggle CLI downloads for the four datasets we actually use; ganeshsura intentionally omitted; Stanford Cars swapped to `eduardo4jesus` mirror.
- [x] Datasets on disk: CarDD (5.7 GB / 16k files), comprehensive (642 MB / 2300 images), iaai (11 MB / 12353 rows), Stanford Cars (1.9 GB / 16185 images).
- [x] [src/ccdp/data/schema.py](../src/ccdp/data/schema.py) — `Record`, `BBox`, `CANONICAL_PARTS`, `DAMAGE_TYPES`, `infer_part_from_damage`, label mapping.
- [x] [src/ccdp/data/loaders.py](../src/ccdp/data/loaders.py) — `iter_cardd`, `iter_comprehensive`, `iter_iaai` with NaN- and PREMIUM-safe parsing.
- [x] [src/ccdp/identification/car_identifier.py](../src/ccdp/identification/car_identifier.py) — filename / EXIF / OCR stages, color heuristic, segment inference.
- [x] [src/ccdp/identification/unidentified.py](../src/ccdp/identification/unidentified.py) — SQLite bucket with auto-naming + label/consume CLIs.
- [x] [src/ccdp/identification/reference_table.py](../src/ccdp/identification/reference_table.py) — build + nearest() + coverage_report, parquet+csv outputs.
- [x] [src/ccdp/identification/build_reference.py](../src/ccdp/identification/build_reference.py) — `build_from_iaai`.
- [x] [src/ccdp/identification/fallback_estimator.py](../src/ccdp/identification/fallback_estimator.py) — 3-tier degradation chain; NaN-safe scaling.
- [x] [src/ccdp/cli.py](../src/ccdp/cli.py) — added `ccdp data {download, inspect, build-reference-table}`, `ccdp unidentified {list, label, stats}`, `ccdp costing import`.
- [x] [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) — executable EDA reconciling real schemas; runs end-to-end on the current data.
- [x] [CITATIONS.md](../CITATIONS.md) — all dataset attributions with licenses, papers, and use-in-project notes.
- [x] Reference table built from iaai metadata at `data/processed/reference_table.parquet`: 811 unique (make/model/year/body) groups, 43 makes, 260 models, 1978–2026, from 2000 sample rows.
- [x] Tests: **40/40 passing** (`pytest -q`).

## Key real-data numbers (from `notebooks/01_eda.ipynb`)

| Dataset | Records | Trainable labels | Coverage |
|---|---|---|---|
| CarDD | 4,000 images / 8,740 bboxes | dent (3595), scratch (2543), crack (898), lamp_broken (704), glass_shatter (681), tire_flat (319) | full 6/6 categories present in all splits |
| comprehensive | 2,300 images | front (1400) / rear (900) × normal (800) / crushed (700) / breakage (800) | balanced |
| iaai (free) | 12,353 rows | year/make/model/body_type/damage_location | 0 cost rows (paywalled) |
| Stanford Cars | 16,185 images (8144 train / 8041 test) | 196 make/model/year classes | full |

## Pending

None for Phase 1. A few items are scoped into later phases:

- Stanford Cars loader → moved to **Phase 1.5** ([progress/phase_1_5_car_identifier.md](phase_1_5_car_identifier.md)).
- Real-cost catalog import (`ccdp costing import --from-dataset iaai`) → wired but no-ops until real cost data is available; the CLI returns a snapshot of the active catalog and prints the reason.
- TartesiaDS as an external eval set → **not in scope** (user declined).

## Open questions / decisions captured

1. **Cost-supervision honesty**: cost predictions come from the catalog. The training catalog id is recorded on each model checkpoint so the calibrator can scale predictions when the catalog is updated, *without retraining*.
2. **Parts taxonomy stays catalog-keyed**: model output is damage TYPE + LOCATION; the parts-keyed cost catalog is reached via `infer_part_from_damage(damage_type, bbox_center, damage_location)`.
3. **iaai cost data**: research-access program at https://rebrowser.net/free-datasets-for-research may unlock un-paywalled cost data later. When/if available, re-run `ccdp data build-reference-table` with the new file and existing models continue working via the calibrator.

## How to verify

```bash
source .venv/bin/activate
pytest -q                                              # 40 / 40 passing

ccdp data inspect --limit 2                            # streams 2 records from each loader
ccdp data build-reference-table --limit 5000           # writes data/processed/reference_table.parquet

ccdp unidentified list                                 # bucket empty until Phase 2 populates it
ccdp unidentified label --image-id <X> --make M --model Mo --year 2018

ccdp costing import --from-dataset iaai                # snapshots active catalog (no real cost yet)
ccdp costing import --file path/to/parts_prices.csv --tag q2-import

# Run the EDA notebook
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

## Notes for future phases

- **Phase 2A** (ResNet50 multi-label) trains on CarDD damage TYPES. Stratified split per-image; iaai metadata is sampled to inject plausible (year/make/model) tabular features.
- **Phase 2B** (YOLOv8) trains on CarDD bboxes; comprehensive provides auxiliary location head.
- **Phase 1.5** (Stanford Cars fine-tune) feeds the identification pipeline so cost-tier-selection has confidence to work with.
- Cost regression (XGBoost) trains on `(image features ⊕ box stats ⊕ tabular metadata)` against synthetic catalog-derived cost targets. See [PLAN.md §3 honesty statement](../PLAN.md).
