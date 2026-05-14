# Phase 0 — Scaffold, costing, FX, CLI

**Status:** ✅ **Done**
**Completed:** 2026-05-12

## Goal

Stand up the repo, the Python package, and the two pieces of infrastructure that downstream phases will lean on from day one: a **versioned parts-cost catalog** and an **FX module**. Wire both into a `ccdp` CLI so the work is usable before any model is trained.

## Deliverables

- Editable-installable Python package `ccdp` (`pip install -e .`).
- Directory skeleton under `src/ccdp/` for every subpackage referenced in [PLAN.md §10](../PLAN.md).
- Versioned cost catalog with create / list / show / activate / diff / estimate.
- FX module with cache / refresh / manual-set / offline mode.
- `ccdp` CLI with `version`, `costing …`, `fx …` subcommands.
- Tests for both modules.

## Done

- [x] [pyproject.toml](../pyproject.toml) — package metadata + optional `[ml]`, `[serve]`, `[dev]` extras, `ccdp` console script.
- [x] [.gitignore](../.gitignore) — ignores raw data, checkpoints, reports, fx cache; **keeps** catalog YAMLs tracked.
- [x] [README.md](../README.md) — quickstart + Phase 0 surface.
- [x] [src/ccdp/__init__.py](../src/ccdp/__init__.py) + subpackage stubs for `data`, `identification`, `costing`, `models`, `train`, `eval`, `registry`, `infer`, `api`.
- [x] [src/ccdp/costing/catalog.py](../src/ccdp/costing/catalog.py) — `Catalog`, `PartCost`, `build_seed_catalog`, list/load/save/activate/diff/estimate, ISO-UTC catalog ids.
- [x] [src/ccdp/costing/fx.py](../src/ccdp/costing/fx.py) — 3-source fallback chain (exchangerate.host → open.er-api.com → frankfurter.app), 24h staleness, `FX_OFFLINE=1`, manual override.
- [x] [src/ccdp/costing/calibrator.py](../src/ccdp/costing/calibrator.py) — `Calibrator` scaling predictions by `active_median / training_median`.
- [x] [src/ccdp/cli.py](../src/ccdp/cli.py) — Typer CLI; rich tables for `list` / `show` / `diff`.
- [x] [tests/test_costing.py](../tests/test_costing.py) — seed consistency, roundtrip, activate, estimate, diff, calibrator linearity.
- [x] [tests/test_fx.py](../tests/test_fx.py) — manual set, identity convert, explicit rate, offline behaviors, cache format.
- [x] Initial seeded catalog: [data/parts_cost_catalog/catalog_2026-05-12T05-45-11_initial.yaml](../data/parts_cost_catalog/catalog_2026-05-12T05-45-11_initial.yaml) (15 parts, USD, median \$870, mid segment).

## Pending

None for this phase.

## Open questions / decisions needed

None outstanding. Decisions captured in [PLAN.md §2](../PLAN.md).

## How to verify

```bash
source .venv/bin/activate
pytest -q                                                  # expect: 12 passed
ccdp version
ccdp costing list                                          # 1 catalog, active
ccdp costing show active
ccdp fx set 83.25                                          # no network
ccdp costing estimate front_bumper hood --currency INR     # → ~$1985 USD / ₹165251 INR
ls data/parts_cost_catalog/                                # active.yaml symlink + catalog file
```

## Notes for future phases

- Trained models will record `training_catalog_id` in their registry entry so the `Calibrator` can be reconstructed at inference time.
- `build_seed_catalog` is the **bootstrap** seed. Phase 1 will derive a second catalog from iaai + ganeshsura medians via `ccdp costing import --from-dataset iaai` (CLI command to be added in Phase 1).
- `Catalog.estimate()` is the Tier-3 fallback used when neither exact-match nor nearest-class cost data is available (see [PLAN.md §6](../PLAN.md)).
