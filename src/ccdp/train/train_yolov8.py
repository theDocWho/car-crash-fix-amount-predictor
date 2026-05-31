"""Train YOLOv8 on CarDD and register the run.

Thin wrapper around Ultralytics that:

1. Materializes the YOLO-format CarDD dataset if missing.
2. Trains YOLOv8 (default nano) for `epochs` epochs on MPS.
3. Imports the resulting `runs/detect/train*/weights/{best,last}.pt` into our
   registry under `checkpoints/detector/run_<ts>_<tag>/`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ccdp.data import cardd_yolo
from ccdp.registry import create_run, update_metrics


@dataclass
class YoloConfig:
    model: str = "yolov8n.pt"           # nano default; swap to yolov8s if accuracy gap warrants
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    patience: int = 15
    workers: int = 4
    tag: str = "yolov8n"
    optimizer: str = "AdamW"
    lr0: float = 1e-3
    device: Optional[str] = None        # None -> ultralytics auto (mps if available)
    seg: bool = False                   # True -> CarDD YOLOv8-seg (polygon masks)
    variant: str = "detector"           # registry variant: detector | yoloseg | parts
    notes: str = "CarDD YOLOv8 damage-type detector (Variant B)"


def _pick_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def _resolve_data(cfg: YoloConfig, data_yaml):
    """Return the Ultralytics ``data=`` argument.

    - ``None``  -> build/locate the CarDD dataset (seg or detect per ``cfg.seg``).
    - a local file path -> absolute path string.
    - a bare name (e.g. ``"carparts-seg.yaml"``) -> passed through so Ultralytics
      resolves it from its dataset registry and auto-downloads.
    """
    if data_yaml is None:
        if cfg.seg:
            return str(cardd_yolo.build_seg().resolve())
        p = cardd_yolo.DEFAULT_ROOT / "data.yaml"
        if not p.exists():
            p = cardd_yolo.build()
        return str(Path(p).resolve())
    p = Path(data_yaml)
    return str(p.resolve()) if p.exists() else str(data_yaml)


def train(
    cfg: YoloConfig,
    data_yaml: Optional[Path] = None,
    training_catalog_id: Optional[str] = None,
    smoke: bool = False,
) -> Path:
    from ultralytics import YOLO

    data_arg = _resolve_data(cfg, data_yaml)

    run_dir = create_run(
        variant=cfg.variant, tag=cfg.tag,
        training_catalog_id=training_catalog_id,
        notes=cfg.notes,
    )
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in asdict(cfg).items()))

    device = _pick_device(cfg.device)
    print(f"[yolo] data={data_arg}  device={device}  model={cfg.model}  epochs={cfg.epochs}")

    model = YOLO(cfg.model)
    # Run training. Ultralytics writes into `runs/detect/train*`. We point its
    # `project=` at our run_dir so the artifacts land in our registry layout.
    results = model.train(
        data=data_arg,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        patience=cfg.patience,
        workers=cfg.workers,
        device=device,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        project=str(run_dir.resolve()),  # absolute so ultralytics doesn't prefix runs/detect/
        name="ultralytics",
        exist_ok=True,
        verbose=False,
        plots=False,
    )

    # Move/symlink best.pt + last.pt to the run_dir root so the registry
    # convention (best.pt at run_dir root) holds.
    ult_dir = run_dir / "ultralytics" / "weights"
    for fname in ("best.pt", "last.pt"):
        src = ult_dir / fname
        dst = run_dir / fname
        if src.exists():
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.relative_to(run_dir))

    # Collect metrics from results.results_dict if available
    metrics: dict = {}
    try:
        rd = getattr(results, "results_dict", None) or {}
        metrics = {k: float(v) for k, v in rd.items() if isinstance(v, (int, float))}
    except Exception:  # noqa: BLE001
        pass
    if metrics:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        update_metrics(run_dir.name.replace("run_", ""), metrics)
        print(f"[metrics] {metrics}")

    print(f"[done] -> {run_dir / 'best.pt'}")
    return run_dir / "best.pt"
