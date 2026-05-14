"""Versioned, pluggable parts-cost catalog.

Each catalog is a YAML file at `data/parts_cost_catalog/catalog_<iso-utc>_<tag>.yaml`.
`active.yaml` is a symlink to the currently-selected catalog.

Public API:
    list_catalogs(root)         -> list[CatalogMeta]
    load(catalog_id_or_path)    -> Catalog
    load_active(root)           -> Catalog
    save(catalog, root, tag)    -> Catalog (writes a new timestamped file)
    activate(catalog_id, root)  -> Catalog (repoints active.yaml)
    diff(id_a, id_b, root)      -> dict[str, dict]
    estimate(catalog, parts, severity_map) -> float
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOT = Path("data/parts_cost_catalog")
SEGMENTS = ("economy", "mid", "luxury")
SEVERITIES = ("minor", "moderate", "severe")
_ID_RE = re.compile(r"^catalog_(?P<id>[0-9TZ:.\-]+_[A-Za-z0-9._-]+)\.yaml$")


@dataclass
class PartCost:
    base_cost: dict[str, float]              # segment -> $
    severity_multiplier: dict[str, float]    # severity -> multiplier
    labor_hours: dict[str, float]            # severity -> hours

    def cost(self, segment: str, severity: str, labor_rate: float) -> float:
        seg = segment if segment in self.base_cost else "mid"
        sev = severity if severity in self.severity_multiplier else "moderate"
        parts_cost = self.base_cost[seg] * self.severity_multiplier[sev]
        labor_cost = self.labor_hours[sev] * labor_rate
        return parts_cost + labor_cost


@dataclass
class Catalog:
    catalog_id: str
    created_at: str
    created_by: str
    source: str
    currency: str
    parts: dict[str, PartCost]
    labor_rate_per_hour: dict[str, float]
    notes: str = ""
    fx_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def path_name(self) -> str:
        return f"catalog_{self.catalog_id}.yaml"

    # --- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "source": self.source,
            "currency": self.currency,
            "fx_snapshot": self.fx_snapshot,
            "labor_rate_per_hour": self.labor_rate_per_hour,
            "parts": {k: asdict(v) for k, v in self.parts.items()},
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Catalog":
        parts = {k: PartCost(**v) for k, v in d["parts"].items()}
        return cls(
            catalog_id=d["catalog_id"],
            created_at=d["created_at"],
            created_by=d.get("created_by", "unknown"),
            source=d.get("source", ""),
            currency=d.get("currency", "USD"),
            parts=parts,
            labor_rate_per_hour=d["labor_rate_per_hour"],
            notes=d.get("notes", ""),
            fx_snapshot=d.get("fx_snapshot", {}) or {},
        )

    # --- estimation ------------------------------------------------------

    def estimate(
        self,
        parts: list[str] | dict[str, str],
        segment: str = "mid",
    ) -> float:
        """Tier-3 fallback estimate.

        `parts` is either a list of part names (severity defaults to 'moderate')
        or a dict mapping part name -> severity.
        """
        if isinstance(parts, list):
            parts_map = {p: "moderate" for p in parts}
        else:
            parts_map = parts
        rate = self.labor_rate_per_hour.get(segment, self.labor_rate_per_hour["mid"])
        total = 0.0
        for part, sev in parts_map.items():
            pc = self.parts.get(part)
            if pc is None:
                continue
            total += pc.cost(segment, sev, rate)
        return round(total, 2)

    def median_cost(self) -> float:
        """Reference scalar used by the calibrator to detect catalog drift."""
        values = []
        for pc in self.parts.values():
            rate = self.labor_rate_per_hour.get("mid", 100.0)
            values.append(pc.cost("mid", "moderate", rate))
        if not values:
            return 0.0
        values.sort()
        n = len(values)
        return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


# --- file ops -----------------------------------------------------------


def _now_id(tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-") or "untagged"
    return f"{ts}_{safe_tag}"


def _catalog_path(root: Path, catalog_id: str) -> Path:
    return root / f"catalog_{catalog_id}.yaml"


def list_catalogs(root: Path = DEFAULT_ROOT) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    active_id = _active_id(root)
    for p in sorted(root.glob("catalog_*.yaml")):
        m = _ID_RE.match(p.name)
        if not m:
            continue
        cid = m.group("id")
        # peek minimal fields
        with p.open() as f:
            data = yaml.safe_load(f)
        out.append({
            "catalog_id": cid,
            "created_at": data.get("created_at"),
            "currency": data.get("currency"),
            "source": data.get("source"),
            "is_active": cid == active_id,
            "path": str(p),
        })
    return out


def _active_id(root: Path) -> str | None:
    link = root / "active.yaml"
    if not link.is_symlink() and not link.exists():
        return None
    try:
        target = os.readlink(link) if link.is_symlink() else link.name
    except OSError:
        return None
    m = _ID_RE.match(Path(target).name)
    return m.group("id") if m else None


def load(catalog_id_or_path: str, root: Path = DEFAULT_ROOT) -> Catalog:
    root = Path(root)
    p = Path(catalog_id_or_path)
    if not p.exists():
        p = _catalog_path(root, catalog_id_or_path)
    if not p.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_id_or_path}")
    with p.open() as f:
        data = yaml.safe_load(f)
    return Catalog.from_dict(data)


def load_active(root: Path = DEFAULT_ROOT) -> Catalog:
    root = Path(root)
    link = root / "active.yaml"
    if not link.exists():
        raise FileNotFoundError(
            f"No active catalog at {link}. Run `ccdp costing init` first."
        )
    with link.open() as f:
        data = yaml.safe_load(f)
    return Catalog.from_dict(data)


def save(catalog: Catalog, root: Path = DEFAULT_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = _catalog_path(root, catalog.catalog_id)
    with p.open("w") as f:
        yaml.safe_dump(catalog.to_dict(), f, sort_keys=False)
    return p


def activate(catalog_id: str, root: Path = DEFAULT_ROOT) -> Path:
    root = Path(root)
    target = _catalog_path(root, catalog_id)
    if not target.exists():
        raise FileNotFoundError(f"Cannot activate missing catalog: {catalog_id}")
    link = root / "active.yaml"
    if link.is_symlink() or link.exists():
        link.unlink()
    # relative symlink so the repo is portable
    link.symlink_to(target.name)
    return link


def diff(id_a: str, id_b: str, root: Path = DEFAULT_ROOT) -> dict[str, dict[str, Any]]:
    """Return per-part % change in base_cost (mid segment) from a -> b."""
    a = load(id_a, root)
    b = load(id_b, root)
    parts = sorted(set(a.parts) | set(b.parts))
    out: dict[str, dict[str, Any]] = {}
    for part in parts:
        pa = a.parts.get(part)
        pb = b.parts.get(part)
        if pa is None:
            out[part] = {"status": "added", "b_mid": pb.base_cost.get("mid")}
            continue
        if pb is None:
            out[part] = {"status": "removed", "a_mid": pa.base_cost.get("mid")}
            continue
        va = pa.base_cost.get("mid", 0.0)
        vb = pb.base_cost.get("mid", 0.0)
        pct = (vb - va) / va * 100 if va else float("inf")
        out[part] = {
            "status": "changed" if abs(pct) > 1e-9 else "same",
            "a_mid": va,
            "b_mid": vb,
            "pct_change": round(pct, 2),
        }
    return out


# --- seed ---------------------------------------------------------------


def build_seed_catalog(created_by: str = "ccdp init", tag: str = "initial") -> Catalog:
    """Construct the initial data-driven seed catalog.

    Values are bootstrap medians drawn from publicly-reported repair averages
    (iaai / ganeshsura distributions, US body-shop reference points). Replace
    via `ccdp costing import` once you have authoritative tables.
    """
    parts = {
        "front_bumper": PartCost(
            base_cost={"economy": 280, "mid": 520, "luxury": 1450},
            severity_multiplier={"minor": 0.4, "moderate": 1.0, "severe": 1.8},
            labor_hours={"minor": 1.5, "moderate": 4.0, "severe": 8.0},
        ),
        "rear_bumper": PartCost(
            base_cost={"economy": 260, "mid": 490, "luxury": 1380},
            severity_multiplier={"minor": 0.4, "moderate": 1.0, "severe": 1.8},
            labor_hours={"minor": 1.5, "moderate": 4.0, "severe": 8.0},
        ),
        "hood": PartCost(
            base_cost={"economy": 340, "mid": 610, "luxury": 1820},
            severity_multiplier={"minor": 0.5, "moderate": 1.0, "severe": 1.9},
            labor_hours={"minor": 2.0, "moderate": 5.0, "severe": 10.0},
        ),
        "front_door": PartCost(
            base_cost={"economy": 410, "mid": 720, "luxury": 1950},
            severity_multiplier={"minor": 0.5, "moderate": 1.0, "severe": 2.0},
            labor_hours={"minor": 2.0, "moderate": 5.0, "severe": 11.0},
        ),
        "rear_door": PartCost(
            base_cost={"economy": 390, "mid": 690, "luxury": 1880},
            severity_multiplier={"minor": 0.5, "moderate": 1.0, "severe": 2.0},
            labor_hours={"minor": 2.0, "moderate": 5.0, "severe": 11.0},
        ),
        "front_fender": PartCost(
            base_cost={"economy": 220, "mid": 410, "luxury": 1180},
            severity_multiplier={"minor": 0.4, "moderate": 1.0, "severe": 1.8},
            labor_hours={"minor": 1.5, "moderate": 4.0, "severe": 9.0},
        ),
        "rear_quarter_panel": PartCost(
            base_cost={"economy": 520, "mid": 880, "luxury": 2350},
            severity_multiplier={"minor": 0.5, "moderate": 1.0, "severe": 2.1},
            labor_hours={"minor": 3.0, "moderate": 7.0, "severe": 14.0},
        ),
        "headlight": PartCost(
            base_cost={"economy": 180, "mid": 420, "luxury": 1650},
            severity_multiplier={"minor": 0.3, "moderate": 1.0, "severe": 1.0},
            labor_hours={"minor": 0.5, "moderate": 1.5, "severe": 2.5},
        ),
        "taillight": PartCost(
            base_cost={"economy": 130, "mid": 290, "luxury": 1080},
            severity_multiplier={"minor": 0.3, "moderate": 1.0, "severe": 1.0},
            labor_hours={"minor": 0.5, "moderate": 1.5, "severe": 2.5},
        ),
        "windshield": PartCost(
            base_cost={"economy": 320, "mid": 540, "luxury": 1480},
            severity_multiplier={"minor": 0.3, "moderate": 1.0, "severe": 1.0},
            labor_hours={"minor": 1.0, "moderate": 2.0, "severe": 3.0},
        ),
        "side_mirror": PartCost(
            base_cost={"economy": 95, "mid": 210, "luxury": 720},
            severity_multiplier={"minor": 0.3, "moderate": 1.0, "severe": 1.0},
            labor_hours={"minor": 0.5, "moderate": 1.0, "severe": 1.5},
        ),
        "roof": PartCost(
            base_cost={"economy": 580, "mid": 960, "luxury": 2780},
            severity_multiplier={"minor": 0.6, "moderate": 1.0, "severe": 2.2},
            labor_hours={"minor": 3.0, "moderate": 8.0, "severe": 18.0},
        ),
        "trunk": PartCost(
            base_cost={"economy": 380, "mid": 640, "luxury": 1820},
            severity_multiplier={"minor": 0.5, "moderate": 1.0, "severe": 1.9},
            labor_hours={"minor": 2.0, "moderate": 5.0, "severe": 10.0},
        ),
        "wheel": PartCost(
            base_cost={"economy": 160, "mid": 320, "luxury": 980},
            severity_multiplier={"minor": 0.3, "moderate": 1.0, "severe": 1.0},
            labor_hours={"minor": 0.5, "moderate": 1.0, "severe": 1.5},
        ),
        "grille": PartCost(
            base_cost={"economy": 180, "mid": 360, "luxury": 1120},
            severity_multiplier={"minor": 0.4, "moderate": 1.0, "severe": 1.6},
            labor_hours={"minor": 0.5, "moderate": 1.5, "severe": 3.0},
        ),
    }
    labor_rate_per_hour = {"economy": 65.0, "mid": 95.0, "luxury": 165.0}
    return Catalog(
        catalog_id=_now_id(tag),
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by=created_by,
        source=(
            "Bootstrap seed. Bootstrap medians from publicly-reported repair averages; "
            "replace with iaai/ganeshsura-derived medians once raw data is processed, "
            "via `ccdp costing import --from-dataset iaai`."
        ),
        currency="USD",
        parts=parts,
        labor_rate_per_hour=labor_rate_per_hour,
        notes=(
            "Tier 3 fallback values only. Real per-vehicle estimates come from the "
            "trained XGBoost regressor calibrated to this catalog."
        ),
    )


def new_catalog_id(tag: str) -> str:
    """Public helper so import-CLI can mint timestamped IDs."""
    return _now_id(tag)
