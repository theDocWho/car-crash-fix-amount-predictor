# Phase 3 — Comparison report + FastAPI + Gradio demo

**Status:** ⏳ Pending
**Estimated effort:** 2 days
**Depends on:** Phases 2A and 2B.

## Goal

Produce the head-to-head **ResNet vs YOLOv8** comparison required by the capstone and ship two ways to actually run the system: a programmatic FastAPI endpoint and an interactive Gradio demo with the catalog / FX / model switchers we promised in [PLAN.md §7-§9](../PLAN.md).

## Deliverables

- `notebooks/05_model_comparison.ipynb` — full comparison table from [PLAN.md §9](../PLAN.md).
- `src/ccdp/eval/comparison_report.py` + `src/ccdp/eval/report_generator.py` — generate HTML + PDF report sections.
- `src/ccdp/api/server.py` — FastAPI app exposing `/estimate` with `?model=resnet|yolov8|both` and `?currency=USD|INR`.
- `src/ccdp/api/demo.py` — Gradio app with:
  - Image upload + optional metadata fields.
  - Model selector (resnet / yolov8 / both).
  - Catalog selector (dropdown of all catalog ids).
  - FX refresh button.
  - "Label this car" tab driven by the unidentified bucket.
- `ccdp serve api` and `ccdp serve demo` CLI subcommands.
- A generated report at `reports/report_<ts>.pdf` and `reports/report_<ts>.html`.

## Done

(none yet)

## Pending

- [ ] **Comparison metrics computation.** All rows from [PLAN.md §9 comparison table](../PLAN.md): per-class F1 (A vs B), mAP@0.5 (B only), regression metrics (RMSE/MAE/MAPE/R²), inference latency (per-image ms on MPS), model size, training wall-clock, Tier 1/2/3 distribution, error slices.
- [ ] **Qualitative grid.** 10 side-by-side examples with Grad-CAM (A) vs detection boxes (B), captioned with both predictions and ground truth.
- [ ] **Decision recommendation table.** "When to use Variant A vs Variant B" matrix in the report.
- [ ] **HTML report.** Built with Jinja2 templates rendering the comparison + slice analyses + provenance breakdown.
- [ ] **PDF export.** Via WeasyPrint or Playwright `page.pdf()` (WeasyPrint preferred — no browser dependency).
- [ ] **FastAPI endpoint.** Pydantic request/response models, multi-part image upload, structured response per [PLAN.md §6](../PLAN.md) with `tier`, `provenance`, `catalog_id`, `fx_snapshot`.
- [ ] **Gradio demo.** All four switchers wired. Catalog dropdown auto-populates from `ccdp.costing.list_catalogs()`; switching it re-prices without rerunning the vision model.
- [ ] **"Label this car" tab.** Lists unidentified images with auto-names; dropdowns for make/model/year; saves to SQLite via `ccdp.identification.unidentified`.
- [ ] **Health endpoint.** `/health` returns `{models_loaded, active_catalog_id, fx_rate, fx_age_hours}`.

## Open questions / decisions needed

1. **PDF backend.** Default to WeasyPrint (pure-Python, but heavy CSS subset). Alt: Playwright (cleaner output but needs a browser binary). Default: WeasyPrint.
2. **Auth on the FastAPI demo.** Capstone scope — no auth. Document this in README.
3. **Gradio share link.** Local-only by default; do not expose via `share=True` unless explicitly run with `--share`.

## How to verify

```bash
ccdp report generate --run-id <id>            # writes reports/report_<ts>.{html,pdf}
ccdp serve api                                # http://127.0.0.1:8000/docs
curl -F "image=@sample.jpg" "http://127.0.0.1:8000/estimate?model=both&currency=INR"
ccdp serve demo                               # http://127.0.0.1:7860
```
