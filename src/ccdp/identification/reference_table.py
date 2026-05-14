"""Canonical reference table: (make, model, year) -> body_type, segment, avg cost.

Built from the cost-bearing datasets (iaai + ganeshsura) during Phase 1 EDA.
Persisted as Parquet (with CSV mirror for inspection) at:

    data/processed/reference_table.parquet
    data/processed/reference_table.csv

The table supports a graceful-degradation `nearest()` lookup used by the
fallback estimator and the Tier-2 cost path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_PATH = Path("data/processed/reference_table.parquet")


@dataclass
class ReferenceRow:
    make: Optional[str]
    model: Optional[str]
    year: Optional[int]
    body_type: str
    segment: str
    avg_cost_usd: float
    n_samples: int
    datasets: str   # comma-joined list of source dataset names

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --- build --------------------------------------------------------------


def build(
    rows: Iterable[dict],
    out_path: Path = DEFAULT_PATH,
) -> Path:
    """Build reference table from per-record dicts.

    `rows` must contain (at minimum) `make`, `model`, `year`, `body_type`,
    `segment`, `cost_usd`, `dataset`. Missing fields are normalized to None.
    """
    import pandas as pd  # local import — pandas is in [ml] extras

    df = pd.DataFrame(list(rows))
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame(
            columns=["make", "model", "year", "body_type", "segment",
                     "avg_cost_usd", "n_samples", "datasets"]
        )
        empty.to_parquet(out_path, index=False)
        empty.to_csv(out_path.with_suffix(".csv"), index=False)
        return out_path

    for col in ("make", "model", "body_type", "segment"):
        if col in df:
            df[col] = df[col].fillna("unknown").astype(str).str.lower()
        else:
            df[col] = "unknown"
    if "year" not in df:
        df["year"] = None

    grouped = (
        df.groupby(["make", "model", "year", "body_type", "segment"], dropna=False)
          .agg(
              avg_cost_usd=("cost_usd", "mean"),
              n_samples=("cost_usd", "size"),
              datasets=("dataset", lambda s: ",".join(sorted(set(map(str, s))))),
          )
          .reset_index()
    )
    grouped = grouped.sort_values(["make", "model", "year"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(out_path, index=False)
    grouped.to_csv(out_path.with_suffix(".csv"), index=False)
    return out_path


def load(path: Path | None = None):
    import pandas as pd
    if path is None:
        path = DEFAULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Reference table not built yet at {path}. Run Phase 1 EDA notebook "
            f"or `ccdp data build-reference-table`."
        )
    return pd.read_parquet(path)


# --- lookup -------------------------------------------------------------


def nearest(
    make: Optional[str] = None,
    model: Optional[str] = None,
    year: Optional[int] = None,
    body_type: Optional[str] = None,
    segment: Optional[str] = None,
    path: Path | None = None,
) -> Optional[dict]:
    """Graceful-degradation lookup. Returns the matched aggregate row + 'how'.

    Chain:
        1. exact (make, model, year)
        2. (make, model) any year — most recent first
        3. (segment, body_type)
        4. (body_type,) only
        5. (segment,) only
    """
    df = load(path)
    if df.empty:
        return None

    def _result(sub, how: str) -> Optional[dict]:
        if sub.empty:
            return None
        # weighted average by sample count
        if "n_samples" in sub and sub["n_samples"].sum() > 0:
            cost = (sub["avg_cost_usd"] * sub["n_samples"]).sum() / sub["n_samples"].sum()
        else:
            cost = sub["avg_cost_usd"].mean()
        return {
            "match_how": how,
            "n_samples": int(sub["n_samples"].sum()) if "n_samples" in sub else len(sub),
            "avg_cost_usd": float(cost),
            "body_type": _mode(sub.get("body_type")),
            "segment": _mode(sub.get("segment")),
            "example_model": _example_model(sub),
        }

    if make and model and year is not None:
        sub = df[(df["make"] == make.lower()) & (df["model"].astype(str).str.startswith(model.lower())) & (df["year"] == year)]
        r = _result(sub, "exact")
        if r:
            return r

    if make and model:
        sub = df[(df["make"] == make.lower()) & (df["model"].astype(str).str.startswith(model.lower()))]
        r = _result(sub, "make_model_any_year")
        if r:
            return r

    if segment and body_type:
        sub = df[(df["segment"] == segment.lower()) & (df["body_type"] == body_type.lower())]
        r = _result(sub, "segment_body_type")
        if r:
            return r

    if body_type:
        sub = df[df["body_type"] == body_type.lower()]
        r = _result(sub, "body_type")
        if r:
            return r

    if segment:
        sub = df[df["segment"] == segment.lower()]
        r = _result(sub, "segment")
        if r:
            return r

    return None


def _mode(series) -> str:
    if series is None or len(series) == 0:
        return "unknown"
    m = series.mode()
    return str(m.iloc[0]) if not m.empty else "unknown"


def _example_model(sub) -> str:
    if sub.empty:
        return "unknown"
    top = sub.sort_values("n_samples", ascending=False).iloc[0]
    parts = [str(top.get("make", "")), str(top.get("model", ""))]
    return " ".join(p for p in parts if p and p != "unknown").strip() or "unknown"


# --- coverage report ----------------------------------------------------


def coverage_report(path: Path | None = None) -> dict:
    """Quick health summary for the report generator."""
    df = load(path)
    if df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "unique_makes": int(df["make"].nunique()),
        "unique_models": int(df["model"].nunique()),
        "year_range": [int(df["year"].dropna().min()) if df["year"].notna().any() else None,
                       int(df["year"].dropna().max()) if df["year"].notna().any() else None],
        "body_type_counts": json.loads(df["body_type"].value_counts().to_json()),
        "segment_counts": json.loads(df["segment"].value_counts().to_json()),
        "total_samples": int(df["n_samples"].sum()),
        "median_avg_cost_usd": float(df["avg_cost_usd"].median()),
    }
