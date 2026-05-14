# Phase 3 — Comparison report + FastAPI + Gradio + HuggingFace deploy

**Status:** ✅ **Done** (scaffold + verified locally; HF Space deploy triggers on next merge to `main`)
**Completed:** 2026-05-14

## Goal

Make the trained v0.1.0 system usable and reviewable by humans rather than only CLI-savvy operators. Three deliverables: a comparison report (HTML + optional PDF), a FastAPI service, and a Gradio demo — all readable from the existing production registry, no new training.

## Pivots vs the original Phase 3 plan

1. **Added image pre-processing (Stage A — downscale + quality report).** Lives in `ccdp.preprocess`. Stage B (super-resolution via Real-ESRGAN) deferred to a later checkpoint per user direction.
2. **Added GitHub→HuggingFace Space auto-deploy.** Free, persistent, Gradio SDK. Workflow lives in `.github/workflows/deploy-hf.yml` and fires on every push to `main`.
3. **Skipped "Label this car" tab.** SQLite bucket is empty in production until we run identification in a batch job; tab would have nothing to show. Deferred to a later checkpoint.

## Deliverables

- [x] [src/ccdp/preprocess/pipeline.py](../src/ccdp/preprocess/pipeline.py) — `preprocess`, `normalize_for_inference`, `quality_report`.
- [x] [src/ccdp/eval/metrics.py](../src/ccdp/eval/metrics.py) — `per_class_prf`, `regression_metrics` (pure functions, no sklearn).
- [x] [src/ccdp/eval/comparison.py](../src/ccdp/eval/comparison.py) — `build_comparison` runs Variants A & B on the seed=42 test split, returns a `Comparison` dataclass.
- [x] [src/ccdp/eval/report.py](../src/ccdp/eval/report.py) — Jinja2 → HTML always; WeasyPrint → PDF when installed.
- [x] [reports/templates/report.html.j2](../reports/templates/report.html.j2) — A4 layout with headline table, per-variant detail, tier/latency, top failures, honest-disclosure footer.
- [x] [src/ccdp/api/schemas.py](../src/ccdp/api/schemas.py) — Pydantic request/response models.
- [x] [src/ccdp/api/server.py](../src/ccdp/api/server.py) — FastAPI app with `/health`, `/catalogs`, `/fx`, `/estimate`. Pipelines loaded once at startup via `lifespan`.
- [x] [src/ccdp/api/demo.py](../src/ccdp/api/demo.py) — Gradio Blocks UI: Estimate / Catalog manager / FX / About tabs.
- [x] CLI: `ccdp serve api`, `ccdp serve demo`, `ccdp report generate`.
- [x] [app.py](../app.py) — HF Space entry; downloads v0.1.0 release assets on first boot, then launches `build_demo`.
- [x] [requirements.txt](../requirements.txt), [packages.txt](../packages.txt) — HF Space build inputs.
- [x] [.github/workflows/deploy-hf.yml](../.github/workflows/deploy-hf.yml) — Mirrors `main` to the HF Space, synthesises the YAML frontmatter so the GitHub README stays clean.
- [x] [README.md](../README.md) — three Mermaid diagrams (local execution, deploy, training).
- [x] Tests: [tests/test_preprocess.py](../tests/test_preprocess.py) + [tests/test_eval_metrics.py](../tests/test_eval_metrics.py) — **76 / 76 passing**.

## Local smoke results

```text
ccdp report generate --limit 5 --no-pdf      →  reports/report_<ts>.html (10.8 KB)
ccdp serve api  →  /health 200 ok            →  variants A & B both loaded on MPS
                  /catalogs 200 [1 catalog]
build_demo()                                  →  gradio Blocks ok
```

## How to verify

```bash
source .venv/bin/activate
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export DYLD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))")

pytest -q                                              # 76 / 76 passing

# Report
ccdp report generate --limit 50 --no-pdf               # ~30 s, writes reports/report_<ts>.html
ccdp report generate                                   # full test split (slower)

# API
ccdp serve api &                                       # http://127.0.0.1:8000/docs
curl http://127.0.0.1:8000/health
curl -F image=@some_car.jpg -F model=both \
     -F make=toyota -F year=2019 -F body_type=sedan \
     http://127.0.0.1:8000/estimate

# Gradio demo
ccdp serve demo                                        # http://127.0.0.1:7860
```

## HuggingFace deployment

One-time setup (already done):
- `HF_USERNAME` repo variable set to `theDocWho`.
- `HF_TOKEN` repo secret set (user-supplied write-scope HF token).

Trigger: every push to `main` (or manual via the Actions tab). The workflow:
1. Checks out the repo at the pushed commit.
2. Prepends the HF Space YAML frontmatter to a temporary copy of README.md.
3. Force-pushes the result to `huggingface.co/spaces/theDocWho/car-crash-fix-amount-predictor`.
4. The Space rebuilds (~2 min); on first boot, `app.py` curl-downloads the v0.1.0 release assets into `checkpoints/production/`.

Once this branch merges to `main`, the Space will appear at:
**https://huggingface.co/spaces/theDocWho/car-crash-fix-amount-predictor**

## Known limitations (documented in app.py)

1. **XGBoost bundle JSON sidecars don't ship in the v0.1.0 release**, so the HF Space's predictions fall through to the Tier-3 catalog-only fallback. Damage detection (Variant A classifier + Variant B detector) is unaffected. Fix: add `bundle.json` sidecars to a future release, no code change needed.
2. **HF Space runs on CPU.** Expected per-image latency: ~3–5 s for `--model both`. Acceptable for a demo.
3. **PDF rendering needs `pango` + `cairo` system libraries** (via `weasyprint`). Not installed on plain macOS without Homebrew. The HTML report always works; the CLI prints a helpful "install hint" when PDF is skipped.
4. **No auth on the API or Gradio demo.** Capstone scope; documented in README.

## Notes for future phases

- **Checkpoint 4** could ship `bundle.json` files into the v0.1.0 release (or a v0.1.1) so the HF Space exercises the full XGBoost path. Pure data, no code change.
- **Checkpoint 5** could add Stage B (Real-ESRGAN) — extra ~65 MB download, hooks into `ccdp.preprocess`.
- **Phase 4** (continued training + promotion harness) remains its own work stream.
