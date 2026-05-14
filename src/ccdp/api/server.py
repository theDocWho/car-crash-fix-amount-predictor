"""FastAPI service exposing the inference pipelines + catalog / FX management.

Endpoints:
    GET  /health                              service + model state
    GET  /catalogs                            list known parts-cost catalogs
    POST /catalogs/{catalog_id}/activate      flip the active symlink
    GET  /fx                                  current cached USD->INR rate
    POST /fx/refresh                          fetch a fresh rate
    POST /estimate                            run Variant A / B / both on an upload

Pipelines are instantiated at startup (one model load) and reused per request.
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ccdp.api.schemas import (
    CatalogEntry,
    Currency,
    FxResponse,
    HealthResponse,
    ModelChoice,
)
from ccdp.costing import activate as activate_catalog
from ccdp.costing import fx as fxmod
from ccdp.costing import list_catalogs, load_active
from ccdp.identification.car_identifier import IdentificationResult, infer_segment
from ccdp.infer.variant_a import VariantAPipeline
from ccdp.preprocess import preprocess
from ccdp.utils import pick_device


# ---------------------------------------------------------------------------
# Lifespan — load pipelines once
# ---------------------------------------------------------------------------


_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipelines at boot; never per request."""
    _state["device"] = str(pick_device())
    _state["variant_a"] = None
    _state["variant_b"] = None
    try:
        _state["variant_a"] = VariantAPipeline()
        print(f"[api] Variant A pipeline loaded on {_state['device']}")
    except Exception as e:  # noqa: BLE001
        print(f"[api] Variant A unavailable: {e}")
    try:
        from ccdp.infer.variant_b import VariantBPipeline
        _state["variant_b"] = VariantBPipeline()
        print(f"[api] Variant B pipeline loaded on {_state['device']}")
    except Exception as e:  # noqa: BLE001
        print(f"[api] Variant B unavailable: {e}")
    yield
    _state.clear()


app = FastAPI(
    title="ccdp",
    description="Car Crash Fix Amount Predictor — damage recognition + cost estimation.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service liveness + which models are loaded + catalog / FX state."""
    try:
        active = load_active()
        catalog_id = active.catalog_id
    except FileNotFoundError:
        catalog_id = None

    fx_rate: Optional[float] = None
    fx_age_hours: Optional[float] = None
    try:
        fr = fxmod.get_rate("USD", "INR", allow_stale=True)
        fx_rate = fr.rate
        fetched = datetime.fromisoformat(fr.fetched_at)
        fx_age_hours = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        pass

    return HealthResponse(
        status="ok",
        active_catalog=catalog_id,
        fx_rate=fx_rate,
        fx_age_hours=fx_age_hours,
        models={
            "variant_a": "loaded" if _state.get("variant_a") else None,
            "variant_b": "loaded" if _state.get("variant_b") else None,
        },
        device=_state.get("device", "unknown"),
    )


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------


@app.get("/catalogs", response_model=list[CatalogEntry])
def catalogs() -> list[CatalogEntry]:
    rows = list_catalogs()
    return [CatalogEntry(
        catalog_id=r["catalog_id"],
        created_at=r.get("created_at"),
        currency=r.get("currency"),
        is_active=r["is_active"],
    ) for r in rows]


@app.post("/catalogs/{catalog_id}/activate")
def catalog_activate(catalog_id: str) -> dict:
    try:
        activate_catalog(catalog_id)
        return {"activated": catalog_id}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------


@app.get("/fx", response_model=FxResponse)
def fx_show() -> FxResponse:
    try:
        fr = fxmod.get_rate("USD", "INR")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return FxResponse(base=fr.base, target=fr.target, rate=fr.rate,
                      source=fr.source, fetched_at=fr.fetched_at)


@app.post("/fx/refresh", response_model=FxResponse)
def fx_refresh() -> FxResponse:
    fr = fxmod.refresh_rate("USD", "INR")
    return FxResponse(base=fr.base, target=fr.target, rate=fr.rate,
                      source=fr.source, fetched_at=fr.fetched_at)


# ---------------------------------------------------------------------------
# Estimate (the main endpoint)
# ---------------------------------------------------------------------------


def _build_identification(make, model_name, year, body_type) -> Optional[IdentificationResult]:
    if not make:
        return None
    return IdentificationResult(
        image_path=Path(""),
        make=make.lower(),
        model=(model_name.lower() if model_name else None),
        year=year,
        body_type=body_type or "unknown",
        segment=infer_segment(make),
        confidence=1.0,
        source="user",
    )


@app.post("/estimate")
async def estimate(
    image: UploadFile = File(..., description="JPEG or PNG car damage image"),
    model: ModelChoice = Form("both"),
    currency: Currency = Form("USD"),
    make: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    body_type: Optional[str] = Form("unknown"),
    refresh_fx: bool = Form(False),
) -> dict:
    """Run the chosen variant(s) on an uploaded image and return a structured response."""
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pil_image, preprocessing_meta = preprocess(raw)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    if refresh_fx:
        try:
            fxmod.refresh_rate("USD", "INR")
        except Exception as e:  # noqa: BLE001
            print(f"[api] FX refresh failed (continuing): {e}")

    metadata = _build_identification(make, model_name, year, body_type)

    response: dict = {
        "preprocessing": preprocessing_meta,
        "active_catalog": load_active().catalog_id,
    }

    if model in ("resnet", "both"):
        pipe = _state.get("variant_a")
        if pipe is None:
            raise HTTPException(status_code=503, detail="Variant A model not loaded")
        response["variant_a"] = pipe.predict(
            pil_image, metadata=metadata, currency=currency,
        ).to_dict()

    if model in ("yolov8", "both"):
        pipe = _state.get("variant_b")
        if pipe is None:
            if model == "yolov8":
                raise HTTPException(status_code=503, detail="Variant B model not loaded")
            # 'both' is best-effort — silently skip B if unavailable
        else:
            response["variant_b"] = pipe.predict(
                pil_image, metadata=metadata, currency=currency,
            ).to_dict()

    return response
