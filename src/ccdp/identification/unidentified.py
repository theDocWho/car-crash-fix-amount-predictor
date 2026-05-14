"""SQLite-backed bucket for cars whose make/model/year couldn't be auto-identified.

Each row gets an auto-assigned placeholder name (color + body_type + sequence)
which the user can later override via `ccdp unidentified label`. Newly-labeled
rows are queued for the next continued-training run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_DB = Path("data/unidentified_cars.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS unidentified_cars (
    image_id              TEXT PRIMARY KEY,
    image_path            TEXT NOT NULL,
    assigned_name         TEXT NOT NULL,
    predicted_body_type   TEXT,
    predicted_segment     TEXT,
    predicted_color       TEXT,
    user_supplied_make    TEXT,
    user_supplied_model   TEXT,
    user_supplied_year    INTEGER,
    is_labeled            INTEGER NOT NULL DEFAULT 0,
    consumed_in_run       TEXT,
    created_at            TEXT NOT NULL,
    last_updated          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uc_is_labeled ON unidentified_cars(is_labeled);
"""


@dataclass
class UnidentifiedCar:
    image_id: str
    image_path: str
    assigned_name: str
    predicted_body_type: Optional[str] = None
    predicted_segment: Optional[str] = None
    predicted_color: Optional[str] = None
    user_supplied_make: Optional[str] = None
    user_supplied_model: Optional[str] = None
    user_supplied_year: Optional[int] = None
    is_labeled: bool = False
    consumed_in_run: Optional[str] = None
    created_at: str = ""
    last_updated: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_obj(r: sqlite3.Row) -> UnidentifiedCar:
    return UnidentifiedCar(
        image_id=r["image_id"],
        image_path=r["image_path"],
        assigned_name=r["assigned_name"],
        predicted_body_type=r["predicted_body_type"],
        predicted_segment=r["predicted_segment"],
        predicted_color=r["predicted_color"],
        user_supplied_make=r["user_supplied_make"],
        user_supplied_model=r["user_supplied_model"],
        user_supplied_year=r["user_supplied_year"],
        is_labeled=bool(r["is_labeled"]),
        consumed_in_run=r["consumed_in_run"],
        created_at=r["created_at"],
        last_updated=r["last_updated"],
    )


def _next_sequence(conn: sqlite3.Connection, body_type: str, color: str) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM unidentified_cars "
        "WHERE predicted_body_type = ? AND predicted_color = ?",
        (body_type, color),
    )
    return int(cur.fetchone()["n"]) + 1


def auto_name(color: str, body_type: str, seq: int) -> str:
    color = (color or "unknown").lower()
    body_type = (body_type or "unknown").lower()
    return f"unknown_{color}_{body_type}_{seq:03d}"


def add(
    image_id: str,
    image_path: str,
    predicted_body_type: str = "unknown",
    predicted_segment: str = "unknown",
    predicted_color: str = "unknown",
    db_path: Path = DEFAULT_DB,
) -> UnidentifiedCar:
    """Insert (or no-op if image_id exists) and return the row."""
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM unidentified_cars WHERE image_id = ?", (image_id,)
        ).fetchone()
        if existing:
            return _row_to_obj(existing)
        seq = _next_sequence(conn, predicted_body_type, predicted_color)
        name = auto_name(predicted_color, predicted_body_type, seq)
        now = _now()
        conn.execute(
            "INSERT INTO unidentified_cars "
            "(image_id, image_path, assigned_name, predicted_body_type, "
            " predicted_segment, predicted_color, is_labeled, created_at, last_updated) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (image_id, image_path, name, predicted_body_type,
             predicted_segment, predicted_color, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM unidentified_cars WHERE image_id = ?", (image_id,)
        ).fetchone()
        return _row_to_obj(row)


def label(
    image_id: str,
    make: str,
    model: str,
    year: int,
    db_path: Path = DEFAULT_DB,
) -> UnidentifiedCar:
    """Apply a user-supplied label and mark the row labeled."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE unidentified_cars "
            "SET user_supplied_make = ?, user_supplied_model = ?, "
            "    user_supplied_year = ?, is_labeled = 1, last_updated = ? "
            "WHERE image_id = ?",
            (make, model, int(year), _now(), image_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"No unidentified row for image_id={image_id}")
        conn.commit()
        row = conn.execute(
            "SELECT * FROM unidentified_cars WHERE image_id = ?", (image_id,)
        ).fetchone()
        return _row_to_obj(row)


def list_rows(
    only_unlabeled: bool = False,
    limit: Optional[int] = None,
    db_path: Path = DEFAULT_DB,
) -> list[UnidentifiedCar]:
    with _connect(db_path) as conn:
        sql = "SELECT * FROM unidentified_cars"
        if only_unlabeled:
            sql += " WHERE is_labeled = 0"
        sql += " ORDER BY created_at"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [_row_to_obj(r) for r in conn.execute(sql).fetchall()]


def newly_labeled(db_path: Path = DEFAULT_DB) -> list[UnidentifiedCar]:
    """Rows labeled but not yet consumed in any continued-training run."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM unidentified_cars "
            "WHERE is_labeled = 1 AND consumed_in_run IS NULL"
        ).fetchall()
    return [_row_to_obj(r) for r in rows]


def mark_consumed(image_ids: Iterable[str], run_id: str, db_path: Path = DEFAULT_DB) -> int:
    """Mark a batch of labeled rows as included in a training run."""
    image_ids = list(image_ids)
    if not image_ids:
        return 0
    with _connect(db_path) as conn:
        q_marks = ",".join("?" * len(image_ids))
        cur = conn.execute(
            f"UPDATE unidentified_cars SET consumed_in_run = ? "
            f"WHERE image_id IN ({q_marks})",
            (run_id, *image_ids),
        )
        conn.commit()
        return cur.rowcount


def stats(db_path: Path = DEFAULT_DB) -> dict[str, int]:
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM unidentified_cars").fetchone()["n"]
        labeled = conn.execute(
            "SELECT COUNT(*) AS n FROM unidentified_cars WHERE is_labeled = 1"
        ).fetchone()["n"]
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM unidentified_cars "
            "WHERE is_labeled = 1 AND consumed_in_run IS NULL"
        ).fetchone()["n"]
    return {"total": total, "labeled": labeled, "pending_consumption": pending}
