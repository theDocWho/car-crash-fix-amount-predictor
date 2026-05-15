"""Gradio Blocks UI for the ccdp inference pipelines.

Layout — three tabs:
    1. Estimate          — upload an image, see Variant A / B side-by-side cost
    2. Catalog manager   — list / view / activate parts-cost catalogs
    3. FX manager        — view / refresh USD↔INR rate

The "Label this car" tab from the original Phase 3 plan is deferred — the
unidentified-cars SQLite bucket is empty in production until we wire identification
into batch processing in a later checkpoint.

The launcher (`build_demo`) is the function the HF Space's ``app.py`` calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gradio as gr
from PIL import Image

from ccdp.costing import activate as activate_catalog
from ccdp.costing import fx as fxmod
from ccdp.costing import list_catalogs
from ccdp.identification.car_identifier import IdentificationResult, infer_segment
from ccdp.preprocess import preprocess
from ccdp.viz import annotate_prediction


# ---------------------------------------------------------------------------
# Pipeline caching (load once, reuse for every UI interaction)
# ---------------------------------------------------------------------------


_pipelines: dict = {}


def _get_pipelines() -> dict:
    """Lazy-load the variant pipelines on first demo interaction."""
    if not _pipelines:
        from ccdp.infer.variant_a import VariantAPipeline
        try:
            _pipelines["a"] = VariantAPipeline()
        except Exception as e:  # noqa: BLE001
            print(f"[demo] Variant A unavailable: {e}")
            _pipelines["a"] = None
        try:
            from ccdp.infer.variant_b import VariantBPipeline
            _pipelines["b"] = VariantBPipeline()
        except Exception as e:  # noqa: BLE001
            print(f"[demo] Variant B unavailable: {e}")
            _pipelines["b"] = None
    return _pipelines


# ---------------------------------------------------------------------------
# Estimate tab handler
# ---------------------------------------------------------------------------


def _build_metadata(make, model_name, year, body_type) -> Optional[IdentificationResult]:
    if not make:
        return None
    return IdentificationResult(
        image_path=Path(""),
        make=make.lower(),
        model=(model_name.lower() if model_name else None),
        year=int(year) if year else None,
        body_type=body_type or "unknown",
        segment=infer_segment(make),
        confidence=1.0,
        source="user",
    )


def _estimate(
    image: Image.Image,
    model_choice: str,
    currency: str,
    classifier_threshold: float,
    detector_conf: float,
    make: str,
    model_name: str,
    year: Optional[int],
    body_type: str,
) -> tuple[Image.Image, str, str, str]:
    """Returns (annotated_image, variant_a_summary, variant_b_summary, full_json).

    The annotated image is the user's upload with Variant B detector boxes
    drawn on it. If the detector finds nothing, the image is overlaid with
    a clear 'No damage detected' banner so the user can tell that the
    detector ran and produced an empty result (vs. failing silently).
    """
    if image is None:
        return None, "Please upload an image.", "", ""
    pipes = _get_pipelines()
    pil_image, preprocessing_meta = preprocess(image)
    metadata = _build_metadata(make, model_name, year, body_type)

    full: dict = {
        "preprocessing": preprocessing_meta,
        "thresholds": {
            "classifier": classifier_threshold,
            "detector_conf": detector_conf,
        },
    }
    a_text, b_text = "Variant A not loaded.", "Variant B not loaded."
    annotated = pil_image  # default: no boxes

    if model_choice in ("Variant A (ResNet50 classifier)", "Both"):
        if pipes.get("a"):
            pred = pipes["a"].predict(
                pil_image, metadata=metadata, currency=currency,
                threshold=classifier_threshold,
            ).to_dict()
            full["variant_a"] = pred
            a_text = _format_prediction("A", pred)

    if model_choice in ("Variant B (YOLOv8 detector)", "Both"):
        if pipes.get("b"):
            pred_b = pipes["b"].predict(
                pil_image, metadata=metadata, currency=currency,
                conf=detector_conf,
            )
            pred = pred_b.to_dict()
            full["variant_b"] = pred
            b_text = _format_prediction("B", pred, n_detections=len(pred_b.detections))
            annotated = annotate_prediction(pil_image, pred_b)
        else:
            b_text = (
                "## Variant B\n"
                "_Detector model not loaded — no boxes available. "
                "See the server logs for the load error._"
            )

    return annotated, a_text, b_text, json.dumps(full, indent=2, default=str)


def _format_prediction(name: str, pred: dict, n_detections: Optional[int] = None) -> str:
    cost = pred.get("cost", 0.0)
    currency = pred.get("currency", "USD")
    types = ", ".join(pred.get("damage_types", [])) or "—"
    parts = ", ".join(pred.get("parts", [])) or "—"
    tier = pred.get("tier", "?")
    prov = pred.get("provenance", "")
    detector_line = ""
    if n_detections is not None:
        if n_detections == 0:
            detector_line = (
                "**Detector:** ran, found **0 boxes** — try lowering the "
                "*Detector confidence* slider, or this car may be undamaged "
                "or out of the training distribution.\n\n"
            )
        else:
            detector_line = f"**Detector:** {n_detections} box(es) above threshold.\n\n"
    return (
        f"## Variant {name}\n"
        f"**Cost:** {cost:.2f} {currency} _(tier: `{tier}`)_\n\n"
        f"{detector_line}"
        f"**Damage types:** {types}\n\n"
        f"**Parts:** {parts}\n\n"
        f"_{prov}_\n"
    )


# ---------------------------------------------------------------------------
# Catalog manager handlers
# ---------------------------------------------------------------------------


def _catalogs_table():
    rows = list_catalogs()
    return [
        [
            "★" if r["is_active"] else "",
            r["catalog_id"],
            r.get("created_at", "") or "",
            r.get("currency", "") or "",
        ]
        for r in rows
    ]


def _activate(catalog_id: str) -> str:
    if not catalog_id:
        return "Pick a catalog id first."
    try:
        activate_catalog(catalog_id.strip())
        return f"Activated: `{catalog_id}`"
    except FileNotFoundError as e:
        return f"Not found: {e}"


# ---------------------------------------------------------------------------
# FX manager
# ---------------------------------------------------------------------------


def _fx_show() -> str:
    try:
        fr = fxmod.get_rate("USD", "INR")
        return f"1 {fr.base} = **{fr.rate:.4f}** {fr.target}  (source: `{fr.source}`, fetched: {fr.fetched_at})"
    except RuntimeError as e:
        return f"_Error: {e}_"


def _fx_refresh() -> str:
    try:
        fr = fxmod.refresh_rate("USD", "INR")
        return f"**Refreshed.** 1 {fr.base} = **{fr.rate:.4f}** {fr.target} ({fr.source})"
    except RuntimeError as e:
        return f"_Error: {e}_"


# ---------------------------------------------------------------------------
# Demo factory
# ---------------------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Build the Gradio app. Returns it without launching; caller decides how to launch."""
    with gr.Blocks(title="ccdp — Car Damage + Repair Cost") as demo:
        gr.Markdown("# Car Crash Fix Amount Predictor")
        gr.Markdown(
            "Upload a damaged-car photo and (optionally) tell us the car's make / "
            "model / year for the most accurate cost. See the GitHub repo "
            "[theDocWho/car-crash-fix-amount-predictor]"
            "(https://github.com/theDocWho/car-crash-fix-amount-predictor) for full docs."
        )

        with gr.Tab("Estimate"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Car damage image")
                    model_choice = gr.Radio(
                        choices=["Variant A (ResNet50 classifier)",
                                 "Variant B (YOLOv8 detector)",
                                 "Both"],
                        value="Both",
                        label="Which model?",
                    )
                    currency = gr.Radio(choices=["USD", "INR"], value="USD", label="Currency")
                    with gr.Accordion("Sensitivity (raise to reduce false positives)", open=False):
                        classifier_threshold = gr.Slider(
                            minimum=0.1, maximum=0.95, step=0.05, value=0.6,
                            label="Classifier threshold",
                            info="Variant A reports a damage class only when its "
                                 "sigmoid probability is above this. Default 0.6 "
                                 "(was 0.5 — raised to suppress false positives "
                                 "on undamaged / out-of-distribution images).",
                        )
                        detector_conf = gr.Slider(
                            minimum=0.05, maximum=0.9, step=0.05, value=0.20,
                            label="Detector confidence",
                            info="Variant B (YOLOv8) keeps boxes above this "
                                 "confidence. Lower = more boxes, more false "
                                 "positives. Raise = fewer, stricter boxes.",
                        )
                    with gr.Accordion("Car metadata (optional but improves cost accuracy)", open=False):
                        make = gr.Textbox(label="Make", placeholder="e.g. Toyota")
                        model_name = gr.Textbox(label="Model", placeholder="e.g. Camry")
                        year = gr.Number(label="Year", value=None, precision=0)
                        body_type = gr.Dropdown(
                            choices=["unknown", "sedan", "suv", "hatchback",
                                     "coupe", "convertible", "wagon", "pickup", "van"],
                            value="unknown",
                            label="Body type",
                        )
                    run_btn = gr.Button("Estimate", variant="primary")
                with gr.Column(scale=1):
                    annotated_out = gr.Image(
                        type="pil",
                        label="Detected damage (Variant B boxes)",
                        interactive=False,
                    )
                    variant_a_out = gr.Markdown(label="Variant A")
                    variant_b_out = gr.Markdown(label="Variant B")
            with gr.Accordion("Full JSON (provenance, probabilities, detections)", open=False):
                json_out = gr.Code(language="json")

            run_btn.click(
                _estimate,
                inputs=[image_in, model_choice, currency,
                        classifier_threshold, detector_conf,
                        make, model_name, year, body_type],
                outputs=[annotated_out, variant_a_out, variant_b_out, json_out],
            )

        with gr.Tab("Catalog manager"):
            gr.Markdown(
                "The active parts-cost catalog backs every cost prediction. "
                "Switching it re-prices the same image **without** retraining the model "
                "via the built-in calibrator."
            )
            catalog_table = gr.Dataframe(
                headers=["active", "catalog_id", "created_at", "currency"],
                value=_catalogs_table,
                interactive=False,
            )
            with gr.Row():
                catalog_pick = gr.Textbox(label="Catalog id to activate")
                activate_btn = gr.Button("Activate")
            activate_msg = gr.Markdown()
            refresh_catalogs_btn = gr.Button("Refresh table")

            activate_btn.click(_activate, inputs=catalog_pick, outputs=activate_msg)
            refresh_catalogs_btn.click(lambda: _catalogs_table(), outputs=catalog_table)

        with gr.Tab("FX (USD ↔ INR)"):
            gr.Markdown("Current FX rate used when you select INR in the Estimate tab.")
            fx_text = gr.Markdown(value=_fx_show)
            fx_refresh_btn = gr.Button("Refresh now")
            fx_refresh_btn.click(_fx_refresh, outputs=fx_text)

        with gr.Tab("About"):
            gr.Markdown(
                "ccdp is a capstone project. Cost predictions are **calibrated triage "
                "estimates** — they are not insurable quotes. The cost target during "
                "training is synthetic (catalog-derived) because no public dataset pairs "
                "car-damage images with real repair invoices.\n\n"
                "See `PLAN.md §3` in the GitHub repo for the full disclosure and "
                "`progress/STATUS.md` for current production metrics.\n\n"
                "## Known limitations\n\n"
                "- **No 'undamaged' class.** The classifier was trained on **CarDD** "
                "(Wang et al. 2023), which contains only damaged-car images. The model "
                "has no concept of 'no damage' — every image will trigger *some* class "
                "above the default threshold. If you photograph an undamaged car, "
                "raise the **Classifier threshold** slider toward `0.8`.\n"
                "- **Domain shift.** CarDD is mostly studio-like Western photos. "
                "Real-world phone photos (varied lighting, bystanders, Indian / Asian "
                "makes) are out-of-distribution and detector recall drops sharply. "
                "Lower the **Detector confidence** slider to surface borderline boxes "
                "or expect zero detections on hard photos.\n"
                "- **No segmentation.** We predict bounding boxes, not pixel masks, "
                "so the area estimate is always an overestimate around the actual "
                "damaged region.\n"
                "- **Synthetic cost target.** The cost regressor was trained on "
                "catalog-derived prices, not real invoices. Treat the dollar amount "
                "as an order-of-magnitude triage estimate, not a quote."
            )

    return demo
