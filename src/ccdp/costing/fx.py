"""FX module: USD <-> INR with caching, manual override, and offline mode.

- Cache file: data/fx_cache.json
- Primary source: exchangerate.host
- Fallbacks: open.er-api.com, frankfurter.app
- Offline / override: FX_OFFLINE=1 or pass `rate=` to convert().
- Stale-after: 24h (configurable).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

DEFAULT_CACHE = Path("data/fx_cache.json")
STALE_AFTER_HOURS = 24
REQUEST_TIMEOUT = 8

_SOURCES = [
    # (name, url-template, json-path-tuple)
    ("exchangerate.host", "https://api.exchangerate.host/latest?base={base}&symbols={target}",
     ("rates", "{target}")),
    ("open.er-api.com", "https://open.er-api.com/v6/latest/{base}",
     ("rates", "{target}")),
    ("frankfurter.app", "https://api.frankfurter.app/latest?from={base}&to={target}",
     ("rates", "{target}")),
]


@dataclass
class FXRate:
    base: str
    target: str
    rate: float
    fetched_at: str
    source: str

    @property
    def fetched_dt(self) -> datetime:
        return datetime.fromisoformat(self.fetched_at)

    def is_stale(self, hours: int = STALE_AFTER_HOURS) -> bool:
        return datetime.now(timezone.utc) - self.fetched_dt > timedelta(hours=hours)


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, indent=2)


def _pair_key(base: str, target: str) -> str:
    return f"{base.upper()}_{target.upper()}"


def _read(d: dict, path: tuple[str, ...], target: str) -> Any:
    cur: Any = d
    for key in path:
        cur = cur[key.format(target=target)]
    return cur


def _live_fetch(base: str, target: str) -> FXRate:
    errors = []
    for name, tmpl, path in _SOURCES:
        try:
            url = tmpl.format(base=base.upper(), target=target.upper())
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            rate = float(_read(data, path, target.upper()))
            if rate <= 0:
                raise ValueError(f"non-positive rate {rate}")
            return FXRate(
                base=base.upper(),
                target=target.upper(),
                rate=rate,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                source=name,
            )
        except Exception as e:  # noqa: BLE001 -- try next source
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"All FX sources failed: {'; '.join(errors)}")


def get_rate(
    base: str = "USD",
    target: str = "INR",
    cache_path: Path = DEFAULT_CACHE,
    allow_stale: bool = True,
) -> FXRate:
    """Return a cached rate. Does NOT hit the network unless cache is missing.

    If cache is stale and `allow_stale=False`, refreshes; otherwise returns
    stale value with the staleness implied by `fetched_at`. Callers should
    check `is_stale()` to decide.
    """
    if os.environ.get("FX_OFFLINE") == "1":
        cache = _load_cache(cache_path)
        entry = cache.get(_pair_key(base, target))
        if entry:
            return FXRate(**entry)
        raise RuntimeError("FX_OFFLINE=1 but no cached rate is available.")

    cache = _load_cache(cache_path)
    key = _pair_key(base, target)
    entry = cache.get(key)
    if entry:
        fr = FXRate(**entry)
        if allow_stale or not fr.is_stale():
            return fr
    return refresh_rate(base, target, cache_path)


def refresh_rate(
    base: str = "USD",
    target: str = "INR",
    cache_path: Path = DEFAULT_CACHE,
) -> FXRate:
    """Force a live fetch and update the cache."""
    fr = _live_fetch(base, target)
    cache = _load_cache(cache_path)
    cache[_pair_key(base, target)] = asdict(fr)
    _save_cache(cache, cache_path)
    return fr


def manual_set(
    base: str,
    target: str,
    rate: float,
    cache_path: Path = DEFAULT_CACHE,
) -> FXRate:
    """Write a manual override into the cache. Source recorded as manual_override."""
    fr = FXRate(
        base=base.upper(),
        target=target.upper(),
        rate=float(rate),
        fetched_at=datetime.now(timezone.utc).isoformat(),
        source="manual_override",
    )
    cache = _load_cache(cache_path)
    cache[_pair_key(base, target)] = asdict(fr)
    _save_cache(cache, cache_path)
    return fr


def convert(
    amount: float,
    base: str,
    target: str,
    cache_path: Path = DEFAULT_CACHE,
    rate: float | None = None,
) -> tuple[float, FXRate | None]:
    """Convert amount; returns (converted_amount, FXRate-used-or-None-if-identity)."""
    if base.upper() == target.upper():
        return float(amount), None
    if rate is not None:
        fr = FXRate(
            base=base.upper(), target=target.upper(), rate=float(rate),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            source="manual_override",
        )
        return amount * fr.rate, fr
    fr = get_rate(base, target, cache_path=cache_path)
    return amount * fr.rate, fr
