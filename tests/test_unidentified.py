"""Tests for the unidentified-cars SQLite bucket."""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.identification import unidentified as un


def test_add_assigns_unique_name(tmp_path: Path):
    db = tmp_path / "u.db"
    a = un.add("d1/a.jpg", "/x/a.jpg", "sedan", "mid", "red", db_path=db)
    b = un.add("d1/b.jpg", "/x/b.jpg", "sedan", "mid", "red", db_path=db)
    assert a.assigned_name.endswith("001")
    assert b.assigned_name.endswith("002")
    assert a.assigned_name != b.assigned_name
    assert "red_sedan" in a.assigned_name


def test_add_is_idempotent(tmp_path: Path):
    db = tmp_path / "u.db"
    a = un.add("d1/a.jpg", "/x/a.jpg", "sedan", "mid", "red", db_path=db)
    a2 = un.add("d1/a.jpg", "/x/a.jpg", "sedan", "mid", "red", db_path=db)
    assert a.assigned_name == a2.assigned_name
    assert len(un.list_rows(db_path=db)) == 1


def test_label_and_consume(tmp_path: Path):
    db = tmp_path / "u.db"
    un.add("d1/a.jpg", "/x/a.jpg", "sedan", "mid", "red", db_path=db)
    un.add("d1/b.jpg", "/x/b.jpg", "sedan", "mid", "red", db_path=db)

    un.label("d1/a.jpg", make="Honda", model="Civic", year=2018, db_path=db)
    pending = un.newly_labeled(db_path=db)
    assert [p.image_id for p in pending] == ["d1/a.jpg"]

    un.mark_consumed(["d1/a.jpg"], run_id="run_xyz", db_path=db)
    assert un.newly_labeled(db_path=db) == []
    s = un.stats(db_path=db)
    assert s == {"total": 2, "labeled": 1, "pending_consumption": 0}


def test_label_missing_raises(tmp_path: Path):
    db = tmp_path / "u.db"
    with pytest.raises(KeyError):
        un.label("nope", "Honda", "Civic", 2018, db_path=db)
