"""Tests for the 'no damage' negatives integration (checkpoint-10).

The classifier trained on CarDD alone has no concept of 'no damage' because
CarDD only contains damaged cars. We integrate Stanford Cars images as
negatives (empty damage label) so the model learns the negative class.

These tests pin three guarantees:
  1. ``iter_negatives`` produces Records with empty damage_types and the
     ``stanford_cars_negative`` provenance tag.
  2. ``encode_labels([])`` for a negative is the all-zero target vector.
  3. ``mix_negatives`` is deterministic, respects the ratio, doesn't mutate
     inputs, and degrades gracefully when fewer negatives are available than
     requested (no oversampling with replacement).
  4. ``pos_weight`` with negatives correctly raises per-class weights — more
     'no damage' images means each positive should count more.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccdp.data import damage_dataset as dd
from ccdp.data.loaders import iter_negatives
from ccdp.data.schema import DAMAGE_TYPES, Record


def _pos(types: list[str], path: str = "p.jpg") -> Record:
    return Record(image_path=Path(path), dataset="cardd", damage_types=types)


def _neg(path: str = "n.jpg") -> Record:
    return Record(image_path=Path(path), dataset="stanford_cars_negative", damage_types=[])


# ---------- iter_negatives -------------------------------------------------


def test_iter_negatives_missing_dir_yields_nothing():
    # Pointing at a non-existent path must not raise — production trainers
    # may simply not have Stanford Cars on disk.
    out = list(iter_negatives(img_dir=Path("definitely/does/not/exist")))
    assert out == []


def test_iter_negatives_yields_empty_label_records(tmp_path: Path):
    # Build a tiny fake "Stanford Cars" tree.
    for i in range(3):
        (tmp_path / f"car_{i:03d}.jpg").write_bytes(b"fake")
    # Decoy file that should be filtered out.
    (tmp_path / "labels.txt").write_text("ignored")

    records = list(iter_negatives(img_dir=tmp_path))
    assert len(records) == 3
    for r in records:
        assert r.damage_types == []
        assert r.bboxes == []
        assert r.dataset == "stanford_cars_negative"


# ---------- encode_labels --------------------------------------------------


def test_encode_labels_empty_is_all_zero():
    """A negative record's label must be the all-zero vector — that's what
    teaches the model 'no damage'."""
    v = dd.encode_labels([])
    assert v == [0.0] * len(DAMAGE_TYPES)
    assert sum(v) == 0.0


# ---------- mix_negatives --------------------------------------------------


def test_mix_negatives_zero_ratio_is_passthrough():
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(10)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(10)]
    out = dd.mix_negatives(pos, neg, ratio=0.0)
    assert len(out) == 10
    assert all(r.dataset == "cardd" for r in out)


def test_mix_negatives_balanced_ratio():
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(20)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(50)]
    out = dd.mix_negatives(pos, neg, ratio=1.0, seed=0)
    n_pos = sum(1 for r in out if r.damage_types)
    n_neg = sum(1 for r in out if not r.damage_types)
    assert n_pos == 20
    assert n_neg == 20  # ratio 1.0 → one negative per positive


def test_mix_negatives_doubles_when_ratio_two():
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(10)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(50)]
    out = dd.mix_negatives(pos, neg, ratio=2.0, seed=0)
    n_neg = sum(1 for r in out if not r.damage_types)
    assert n_neg == 20


def test_mix_negatives_caps_at_available_no_oversample():
    """If we ask for 100 negatives but only have 5, use 5 — never duplicate.
    Duplicating identical photos would just teach the model to memorise them."""
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(100)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(5)]
    out = dd.mix_negatives(pos, neg, ratio=1.0)
    n_neg = sum(1 for r in out if not r.damage_types)
    assert n_neg == 5  # bounded by availability
    # And every chosen negative is unique.
    neg_paths = [str(r.image_path) for r in out if not r.damage_types]
    assert len(neg_paths) == len(set(neg_paths))


def test_mix_negatives_deterministic():
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(20)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(50)]
    a = dd.mix_negatives(pos, neg, ratio=1.0, seed=42)
    b = dd.mix_negatives(pos, neg, ratio=1.0, seed=42)
    assert [str(r.image_path) for r in a] == [str(r.image_path) for r in b]


def test_mix_negatives_does_not_mutate_inputs():
    pos = [_pos(["dent"], path=f"p{i}.jpg") for i in range(5)]
    neg = [_neg(path=f"n{i}.jpg") for i in range(5)]
    pos_before, neg_before = list(pos), list(neg)
    _ = dd.mix_negatives(pos, neg, ratio=1.0)
    assert pos == pos_before and neg == neg_before


def test_mix_negatives_empty_negatives_returns_positives():
    pos = [_pos(["dent"]) for _ in range(3)]
    out = dd.mix_negatives(pos, [], ratio=1.0)
    assert len(out) == 3 and all(r.damage_types for r in out)


# ---------- pos_weight with mixed records ----------------------------------


def test_pos_weight_increases_when_negatives_added():
    """Adding 'no damage' images should make every class harder to predict
    positively — pos_weight goes up, not down. This is the math we want:
    'most images are clean, so be more confident before saying dent.'

    We use a class that's already a minority (10 'dent' out of 100 positives)
    so the ratio is well above the floor of 1.0 in both cases.
    """
    pos = [_pos(["dent"]) for _ in range(10)] + [_pos(["scratch"]) for _ in range(90)]
    pw_pure = dd.pos_weight(pos)
    mixed = pos + [_neg() for _ in range(100)]
    pw_mixed = dd.pos_weight(mixed)
    i_dent = DAMAGE_TYPES.index("dent")
    # Pure: (100-10)/10 = 9.0; mixed: (200-10)/10 = 19.0
    assert pw_pure[i_dent] == pytest.approx(9.0)
    assert pw_mixed[i_dent] == pytest.approx(19.0)
    assert pw_mixed[i_dent] > pw_pure[i_dent]


def test_pos_weight_negatives_only_safe():
    """Edge case: all-negatives input. No positives anywhere — every class
    should fall back to the floor weight without dividing by zero."""
    recs = [_neg() for _ in range(10)]
    pw = dd.pos_weight(recs)
    assert len(pw) == len(DAMAGE_TYPES)
    assert all(w == 1.0 for w in pw)
