# Project Status — Car Crash Fix Amount Predictor

**Last updated:** 2026-05-13 (Phase 2B scaffold + smoke end-to-end; both variants now runnable side-by-side)

This folder tracks what has been built and what remains. One file per phase. The full design lives in [../PLAN.md](../PLAN.md).

## Phase index

| Phase | Title | Status | Detail |
|---|---|---|---|
| 0 | Scaffold, costing catalog, FX, CLI | ✅ **Done** | [phase_0_scaffold.md](phase_0_scaffold.md) |
| 1 | Datasets: download, EDA, join, identification, reference table, unidentified bucket | ✅ **Done** | [phase_1_data_and_identification.md](phase_1_data_and_identification.md) |
| 1.5 | Make/model identifier (Stanford Cars fine-tune) | ✅ **Done** (scaffold + smoke-verified; full training is a CLI command) | [phase_1_5_car_identifier.md](phase_1_5_car_identifier.md) |
| 2A | ResNet50 multi-label damage classifier + XGBoost(A) | ✅ **Done** (scaffold + smoke end-to-end; full training is a CLI sequence) | [phase_2a_resnet_classifier.md](phase_2a_resnet_classifier.md) |
| 2B | YOLOv8 damage detector + XGBoost(B) | ✅ **Done** (scaffold + smoke; full training is a CLI sequence) | [phase_2b_yolov8_detector.md](phase_2b_yolov8_detector.md) |
| 3 | Comparison notebook, report generator, FastAPI + Gradio demo | ⏳ Pending | [phase_3_comparison_and_serving.md](phase_3_comparison_and_serving.md) |
| 4 | Continued-training + promotion workflow validation, final report | ⏳ Pending | [phase_4_promotion_and_final_report.md](phase_4_promotion_and_final_report.md) |

## Conventions

Each phase doc carries the same six sections so you can scan any of them quickly:

1. **Goal** — one paragraph.
2. **Deliverables** — file paths + CLI commands that must exist when the phase is done.
3. **Done** — checked items, with file links.
4. **Pending** — unchecked items, in execution order.
5. **Open questions / decisions needed** — anything blocking, or anything I'll default if not answered.
6. **How to verify** — exact commands to confirm the phase works.

When a phase finishes, move all unchecked items to **Done**, update the table above, and bump the "Last updated" date here and in the phase doc.

## Quick verification — what works today

```bash
source .venv/bin/activate
pytest -q                                                  # 12 / 12 passing
ccdp costing list
ccdp costing show active
ccdp costing estimate front_bumper hood --currency INR     # needs cached fx rate
ccdp fx show
```
