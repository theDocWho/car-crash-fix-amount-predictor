# Phase 4 — Continued-training, promotion workflow, final report

**Status:** ⏳ Pending
**Estimated effort:** 1 day
**Depends on:** Phases 2A, 2B, 3.

## Goal

Prove out the **continued-training** and **promotion** workflows end-to-end on a deliberate, scripted scenario — so a future user (or future-you) extending the system has a paved path. Produce the final capstone report.

## Deliverables

- `src/ccdp/train/continue_training.py` — base-checkpoint + new-data loop with lower LR + optional layer freezing.
- `scripts/promote_model.py` (or `ccdp registry promote`) — A/B eval against current production with guard-metric checks.
- `ccdp registry rollback` — flips active symlink back to previous run.
- `notebooks/06_continued_training_demo.ipynb` — end-to-end demo on a held-out subset masquerading as "new data".
- Final report `reports/final_report.pdf` covering all sections of [PLAN.md §13](../PLAN.md).
- Definition-of-done checklist from [PLAN.md §17](../PLAN.md), all checked.

## Done

(none yet)

## Pending

- [ ] **Continued-training CLI.** `ccdp train continue --base-checkpoint <path> --new-data <path> --lr 1e-5 --freeze-until layer3`. Writes to a new run dir; merges newly-labeled rows from the unidentified bucket automatically.
- [ ] **Catastrophic-forgetting guard.** Hold out a small "anchor" eval set from the original training data; the continued run must not regress >5% on the anchor metric, else abort.
- [ ] **Promotion script.** `ccdp registry promote <run_id>`:
  - Eval candidate vs production on frozen test set.
  - Write `reports/promotion_<ts>.html` with metrics diff + per-class regression check.
  - Flip production symlink only if primary metric improves AND all guard metrics stay within 2%.
  - On failure: keep candidate in `checkpoints/<variant>/run_<ts>_<tag>/`, log why it wasn't promoted.
- [ ] **Rollback CLI.** `ccdp registry rollback` repoints `production/*` to the previous-active entries recorded in `registry.json`.
- [ ] **Continued-training demo notebook.** Take Variant A best.pt, train on a 10% held-out slice with `--freeze-until layer3`, show the promotion run accepting or rejecting it.
- [ ] **Final report.**
  - All sections from [PLAN.md §13](../PLAN.md).
  - Variant A vs Variant B comparison.
  - Provenance breakdown chart (tier distribution) and per-tier RMSE.
  - Limitations section with explicit synthetic-cost caveats and "indicative, not insurable" disclaimer.
  - Unidentified bucket appendix.
  - Catalog + FX snapshot footer.
- [ ] **Definition-of-done sweep.** Walk the checklist in [PLAN.md §17](../PLAN.md), check every box, link the artifact for each.

## Open questions / decisions needed

1. **Promotion guard threshold.** Default ±2% on guard metrics; ±5% on the anchor set for forgetting. Adjustable via config.
2. **Auto-rollback on production crash?** Out of scope for capstone — manual `ccdp registry rollback` is sufficient. Document this.

## How to verify

```bash
# Simulate "new data arrived"
ccdp train continue \
  --base-checkpoint checkpoints/resnet/run_X/best.pt \
  --new-data data/raw/holdout_slice/ \
  --lr 1e-5 --freeze-until layer3

# Try to promote
ccdp registry promote <new_run_id>            # writes a comparison report

# If it took, roll back to verify the rollback path
ccdp registry rollback
ccdp registry list                            # confirms previous run is active again

# Generate the final report
ccdp report generate --final
open reports/final_report.pdf
```
