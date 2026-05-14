"""Pydantic models for the FastAPI surface.

Kept separate from `server.py` so they can be imported by tests, by clients
generating types from OpenAPI, and by the Gradio demo without dragging in
FastAPI itself.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ModelChoice = Literal["resnet", "yolov8", "both"]
Currency = Literal["USD", "INR"]


class EstimateMetadata(BaseModel):
    """Optional car metadata supplied by the caller."""
    make: Optional[str] = None
    model_name: Optional[str] = Field(default=None, description="Avoid 'model' clash with pydantic.")
    year: Optional[int] = None
    body_type: Optional[str] = "unknown"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    active_catalog: Optional[str]
    fx_rate: Optional[float]
    fx_age_hours: Optional[float]
    models: dict[str, Optional[str]]
    device: str


class CatalogEntry(BaseModel):
    catalog_id: str
    created_at: Optional[str]
    currency: Optional[str]
    is_active: bool


class FxResponse(BaseModel):
    base: str
    target: str
    rate: float
    source: str
    fetched_at: str
