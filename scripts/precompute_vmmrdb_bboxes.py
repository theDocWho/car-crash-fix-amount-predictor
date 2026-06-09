"""Pre-compute car-bounding-boxes for every VMMRdb training image (Option 3).

The VMMRdb Kaggle mirror has no GT bboxes, which means our identifier trains
on full-frame photos while the Stanford baseline uses GT crops. That single
preprocessing mismatch is the biggest contributor to the 0.40 vs 0.77 gap
(see notebook §2.4). This script fixes it: run Mask R-CNN (``ccdp.identification.car_gate``)
once over the corpus and cache the largest-vehicle bbox per image to JSON.

The trainer then crops to the cached bbox before applying transforms — same
input distribution as Stanford got from GT boxes.

Cache format (JSON):
    {
        "schema_version": 1,
        "root": "<absolute path used for relative keys>",
        "bboxes": {
            "honda accord 2007/abc123.jpg": [x1, y1, x2, y2],
            "toyota camry 2014/def456.jpg": null,    # no car detected
            ...
        },
        "stats": {"n_images": N, "n_detected": M, "n_missed": N - M, ...}
    }

Idempotent: keeps already-cached entries on re-run (skip-if-present), so
adding new images or running across multiple sessions costs nothing.

Usage::

    python scripts/precompute_vmmrdb_bboxes.py                       # full corpus
    python scripts/precompute_vmmrdb_bboxes.py --max-samples 200     # quick smoke
    python scripts/precompute_vmmrdb_bboxes.py --score-threshold 0.4 # looser gate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional


def precompute(
    cache_path: Path,
    max_samples: Optional[int] = None,
    score_threshold: float = 0.5,
    save_every: int = 500,
    top_n: Optional[int] = None,
) -> dict:
    from ccdp.data import vmmrdb
    from ccdp.identification.car_gate import CarGate

    root = Path(vmmrdb.ROOT).resolve()

    # Load existing cache (idempotent re-run)
    existing: dict[str, Optional[list[float]]] = {}
    if cache_path.exists():
        try:
            old = json.loads(cache_path.read_text())
            existing = old.get("bboxes", {}) or {}
            if old.get("root") and Path(old["root"]).resolve() != root:
                print(f"  ! cache root mismatch ({old['root']} != {root}); ignoring old cache")
                existing = {}
            else:
                print(f"  loaded existing cache: {len(existing)} entries")
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not read existing cache ({e}); starting fresh")

    samples = vmmrdb.load_train_samples(top_n=top_n)
    print(f"  {len(samples)} VMMRdb images on disk under {root}")
    if max_samples:
        samples = samples[:max_samples]
        print(f"  capped to first {len(samples)} for this run")

    gate = CarGate(score_threshold=score_threshold)

    n_total = len(samples)
    n_done, n_skipped = 0, 0
    n_detected, n_missed, n_errors = 0, 0, 0
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            rel = str(s.image_path.resolve().relative_to(root))
        except ValueError:
            rel = str(s.image_path)
        if rel in existing:
            n_skipped += 1
            continue

        try:
            res = gate.detect(s.image_path)
            if res.has_car and res.box is not None:
                # Round to int — saves bytes in JSON, lossless for crop
                x1, y1, x2, y2 = res.box
                existing[rel] = [int(round(x1)), int(round(y1)),
                                 int(round(x2)), int(round(y2))]
                n_detected += 1
            else:
                existing[rel] = None
                n_missed += 1
        except Exception as e:  # noqa: BLE001
            existing[rel] = None
            n_errors += 1
            if n_errors <= 5:
                print(f"  ! {s.image_path}: {type(e).__name__}: {e}")

        n_done += 1
        if n_done % 50 == 0 or n_done == len(samples) - n_skipped:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            eta = (len(samples) - n_skipped - n_done) / max(rate, 0.01)
            print(f"  ..{n_done}/{n_total - n_skipped} new "
                  f"(detected={n_detected} missed={n_missed} errors={n_errors})  "
                  f"{rate:.1f} img/s  eta {eta/60:.1f} min")
        if n_done and n_done % save_every == 0:
            _save(cache_path, root, existing, n_detected, n_missed, n_errors)
            print(f"  [save] checkpoint -> {cache_path.name}  ({len(existing)} entries)")

    _save(cache_path, root, existing, n_detected, n_missed, n_errors)
    print(f"\n  final cache: {len(existing)} entries  ({n_detected} detected, "
          f"{n_missed} no-car, {n_errors} errored)")
    print(f"  detection rate: {n_detected / max(len(existing), 1):.1%}")
    return existing


def _save(path: Path, root: Path, bboxes: dict, n_det: int, n_mis: int, n_err: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "root": str(root),
        "bboxes": bboxes,
        "stats": {
            "n_images": len(bboxes),
            "n_detected": n_det,
            "n_missed": n_mis,
            "n_errors": n_err,
        },
    }
    path.write_text(json.dumps(payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("data/processed/vmmrdb_bboxes.json"),
                    help="Output JSON cache path.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap to first N samples (debug).")
    ap.add_argument("--score-threshold", type=float, default=0.5,
                    help="Mask R-CNN score threshold for accepting a vehicle (default 0.5).")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Restrict to top-N largest VMMRdb classes (mirrors the trainer).")
    ap.add_argument("--save-every", type=int, default=500,
                    help="Flush cache to disk every N new entries (resumability).")
    args = ap.parse_args()
    precompute(args.cache, args.max_samples, args.score_threshold,
               args.save_every, args.top_n)


if __name__ == "__main__":
    main()
