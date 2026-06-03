"""ccdp CLI — Typer-based entrypoint.

Phase 0 surface: `ccdp costing ...` and `ccdp fx ...`.
Later phases add `train`, `infer`, `registry`, `serve`, etc.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ccdp import __version__
from ccdp.costing import catalog as catmod
from ccdp.costing import fx as fxmod

app = typer.Typer(help="Car Crash Damage Predictor CLI", no_args_is_help=True)
costing_app = typer.Typer(help="Versioned parts-cost catalog commands.")
fx_app = typer.Typer(help="USD<->INR FX rate commands.")
data_app = typer.Typer(help="Dataset commands: download, schema inspection, reference table.")
unidentified_app = typer.Typer(help="Manage the unidentified-cars bucket.")
train_app = typer.Typer(help="Training commands.")
registry_app = typer.Typer(help="Model registry commands.")
serve_app = typer.Typer(help="Serve the inference API or the Gradio demo.")
report_app = typer.Typer(help="Generate the comparison report.")
app.add_typer(costing_app, name="costing")
app.add_typer(fx_app, name="fx")
app.add_typer(data_app, name="data")
app.add_typer(unidentified_app, name="unidentified")
app.add_typer(train_app, name="train")
app.add_typer(registry_app, name="registry")
app.add_typer(serve_app, name="serve")
app.add_typer(report_app, name="report")

console = Console()


@app.command()
def version() -> None:
    """Print the ccdp version."""
    console.print(f"ccdp {__version__}")


# ----------------- costing -----------------------------------------------


@costing_app.command("init")
def costing_init(
    tag: str = typer.Option("initial", help="Tag suffix for the catalog id."),
    root: Path = typer.Option(catmod.DEFAULT_ROOT, help="Catalogs directory."),
    force: bool = typer.Option(False, help="Re-seed even if catalogs already exist."),
) -> None:
    """Create the initial data-driven seed catalog and activate it."""
    existing = catmod.list_catalogs(root)
    if existing and not force:
        console.print(
            f"[yellow]Catalogs already exist ({len(existing)}). "
            f"Use --force to add another seed.[/yellow]"
        )
        raise typer.Exit(0)
    cat = catmod.build_seed_catalog(tag=tag)
    path = catmod.save(cat, root)
    catmod.activate(cat.catalog_id, root)
    console.print(f"[green]Created and activated catalog:[/green] {cat.catalog_id}")
    console.print(f"  path: {path}")
    console.print(f"  parts: {len(cat.parts)}, median_cost(mid/moderate): ${cat.median_cost():.2f}")


@costing_app.command("list")
def costing_list(
    root: Path = typer.Option(catmod.DEFAULT_ROOT, help="Catalogs directory."),
) -> None:
    """List all known catalogs."""
    rows = catmod.list_catalogs(root)
    if not rows:
        console.print("[yellow]No catalogs found. Run `ccdp costing init`.[/yellow]")
        return
    table = Table(title="Cost catalogs")
    table.add_column("active", justify="center")
    table.add_column("catalog_id")
    table.add_column("created_at")
    table.add_column("currency")
    table.add_column("source", overflow="fold")
    for r in rows:
        table.add_row(
            "[bold green]*[/bold green]" if r["is_active"] else "",
            r["catalog_id"],
            r["created_at"] or "",
            r["currency"] or "",
            (r["source"] or "")[:70],
        )
    console.print(table)


@costing_app.command("show")
def costing_show(
    catalog_id: str = typer.Argument(..., help="Catalog id or 'active'."),
    root: Path = typer.Option(catmod.DEFAULT_ROOT, help="Catalogs directory."),
) -> None:
    """Show a catalog's contents."""
    cat = catmod.load_active(root) if catalog_id == "active" else catmod.load(catalog_id, root)
    console.print(f"[bold]{cat.catalog_id}[/bold]  ({cat.currency})  median=${cat.median_cost():.2f}")
    console.print(f"  created_at: {cat.created_at}")
    console.print(f"  source: {cat.source}")
    table = Table(title="Parts (mid segment, moderate severity)")
    table.add_column("part")
    table.add_column("base", justify="right")
    table.add_column("labor h", justify="right")
    table.add_column("cost@mid", justify="right")
    for name, pc in sorted(cat.parts.items()):
        rate = cat.labor_rate_per_hour.get("mid", 95.0)
        table.add_row(
            name,
            f"{pc.base_cost.get('mid', 0):.2f}",
            f"{pc.labor_hours.get('moderate', 0):.1f}",
            f"{pc.cost('mid', 'moderate', rate):.2f}",
        )
    console.print(table)


@costing_app.command("activate")
def costing_activate(
    catalog_id: str = typer.Argument(...),
    root: Path = typer.Option(catmod.DEFAULT_ROOT, help="Catalogs directory."),
) -> None:
    """Repoint active.yaml to a specific catalog."""
    catmod.activate(catalog_id, root)
    console.print(f"[green]Activated:[/green] {catalog_id}")


@costing_app.command("diff")
def costing_diff(
    id_a: str = typer.Argument(...),
    id_b: str = typer.Argument(...),
    root: Path = typer.Option(catmod.DEFAULT_ROOT, help="Catalogs directory."),
) -> None:
    """Show per-part % change between two catalogs (mid segment, base price)."""
    d = catmod.diff(id_a, id_b, root)
    table = Table(title=f"Diff {id_a} -> {id_b}  (mid base_cost)")
    table.add_column("part")
    table.add_column("status")
    table.add_column("a", justify="right")
    table.add_column("b", justify="right")
    table.add_column("% change", justify="right")
    for part, info in d.items():
        table.add_row(
            part,
            info["status"],
            f"{info.get('a_mid', ''):>}",
            f"{info.get('b_mid', ''):>}",
            f"{info.get('pct_change', ''):>}",
        )
    console.print(table)


@costing_app.command("estimate")
def costing_estimate(
    parts: list[str] = typer.Argument(..., help="Damaged part names (space-separated)."),
    segment: str = typer.Option("mid", help="economy | mid | luxury"),
    severity: str = typer.Option("moderate", help="minor | moderate | severe"),
    catalog_id: str = typer.Option("active", help="Catalog id or 'active'."),
    currency: str = typer.Option("USD", help="Output currency: USD or INR."),
) -> None:
    """Tier-3 catalog-only cost estimate for a given parts list."""
    cat = catmod.load_active() if catalog_id == "active" else catmod.load(catalog_id)
    parts_map = {p: severity for p in parts}
    usd = cat.estimate(parts_map, segment=segment)
    if currency.upper() == cat.currency.upper():
        console.print(f"[bold]Estimate:[/bold] {usd:.2f} {cat.currency}  ({cat.catalog_id})")
        return
    out, fr = fxmod.convert(usd, cat.currency, currency)
    console.print(f"[bold]Estimate:[/bold] {usd:.2f} {cat.currency} = {out:.2f} {currency.upper()}")
    if fr:
        console.print(
            f"  fx: 1 {fr.base} = {fr.rate} {fr.target}  "
            f"(source={fr.source}, fetched={fr.fetched_at})"
        )


# ----------------- fx ----------------------------------------------------


@fx_app.command("show")
def fx_show(
    base: str = typer.Option("USD"),
    target: str = typer.Option("INR"),
) -> None:
    """Show the cached FX rate. Will fetch if no cache exists."""
    try:
        fr = fxmod.get_rate(base, target)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    stale = " [yellow](stale)[/yellow]" if fr.is_stale() else ""
    console.print(f"1 {fr.base} = {fr.rate} {fr.target}{stale}")
    console.print(f"  source: {fr.source}, fetched: {fr.fetched_at}")


@fx_app.command("refresh")
def fx_refresh(
    base: str = typer.Option("USD"),
    target: str = typer.Option("INR"),
) -> None:
    """Force a live FX fetch and update cache."""
    fr = fxmod.refresh_rate(base, target)
    console.print(f"[green]Refreshed:[/green] 1 {fr.base} = {fr.rate} {fr.target}")
    console.print(f"  source: {fr.source}, fetched: {fr.fetched_at}")


@fx_app.command("set")
def fx_set(
    rate: float = typer.Argument(..., help="Manual rate, e.g. 83.2 for USD->INR."),
    base: str = typer.Option("USD"),
    target: str = typer.Option("INR"),
) -> None:
    """Set a manual FX override (recorded as manual_override)."""
    fr = fxmod.manual_set(base, target, rate)
    console.print(f"[green]Manual override set:[/green] 1 {fr.base} = {fr.rate} {fr.target}")


# ----------------- data --------------------------------------------------


@data_app.command("download")
def data_download(
    stanford_cars: bool = typer.Option(True, help="Include Stanford Cars (Phase 1.5)."),
) -> None:
    """Run scripts/download_datasets.sh to fetch all primary datasets via Kaggle CLI."""
    import os
    import subprocess
    env = os.environ.copy()
    if stanford_cars:
        env["STANFORD_CARS"] = "1"
    script = Path("scripts/download_datasets.sh")
    if not script.exists():
        console.print(f"[red]Missing {script}[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Running[/bold] {script} (logs at data/raw/_download.log)")
    rc = subprocess.call(["bash", str(script)], env=env)
    raise typer.Exit(rc)


@data_app.command("inspect")
def data_inspect(
    limit: int = typer.Option(5, help="Records to sample per dataset."),
) -> None:
    """Print a few records from each loader to verify on-disk layout."""
    from itertools import islice

    from ccdp.data.loaders import (
        CARDD_ROOT,
        COMPREHENSIVE_ROOT,
        IAAI_ROOT,
        iter_cardd,
        iter_comprehensive,
        iter_iaai,
    )

    sources = [
        ("CarDD (val)", CARDD_ROOT / "annotations" / "instances_val2017.json",
         lambda: iter_cardd(splits=("val",))),
        ("comprehensive", COMPREHENSIVE_ROOT, iter_comprehensive),
        ("iaai", IAAI_ROOT, iter_iaai),
    ]
    for name, probe, gen in sources:
        console.print(f"\n[bold cyan]=== {name} ===[/bold cyan]")
        if not Path(probe).exists():
            console.print(f"  [yellow]missing: {probe}[/yellow]")
            continue
        for r in islice(gen(), limit):
            console.print(f"  • {r.image_id}  dt={r.damage_types}  "
                          f"loc={r.damage_location}  cond={r.damage_condition}  "
                          f"make={r.make}  model={r.model}  year={r.year}  "
                          f"body={r.body_type}")


@data_app.command("build-reference-table")
def data_build_reference_table(
    limit: int = typer.Option(0, help="Optional cap on iaai rows (0=all)."),
    out: Path = typer.Option(None, help="Output path; defaults to data/processed/reference_table.parquet"),
) -> None:
    """Aggregate iaai metadata into the reference table used by Tier-2 lookups."""
    from ccdp.identification import build_reference, reference_table as reftab

    out_path = out or reftab.DEFAULT_PATH
    console.print(f"[bold]Building reference table[/bold] -> {out_path}")
    build_reference.build_from_iaai(out_path=out_path, limit=(limit or None))
    rep = reftab.coverage_report(out_path)
    console.print(f"[green]Done.[/green]  rows={rep.get('rows')}, "
                  f"unique_makes={rep.get('unique_makes')}, "
                  f"unique_models={rep.get('unique_models')}, "
                  f"year_range={rep.get('year_range')}")


# ----------------- unidentified -----------------------------------------


@unidentified_app.command("list")
def un_list(
    only_unlabeled: bool = typer.Option(False, help="Only show rows still missing labels."),
    limit: int = typer.Option(20),
) -> None:
    """List rows in the unidentified-cars bucket."""
    from ccdp.identification.unidentified import list_rows, stats
    rows = list_rows(only_unlabeled=only_unlabeled, limit=limit)
    if not rows:
        console.print("[yellow]Bucket empty.[/yellow]")
        return
    table = Table(title="Unidentified cars")
    for col in ("image_id", "assigned_name", "body_type", "segment", "color",
                "make?", "model?", "year?"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            r.image_id, r.assigned_name,
            r.predicted_body_type or "", r.predicted_segment or "",
            r.predicted_color or "",
            r.user_supplied_make or "", r.user_supplied_model or "",
            str(r.user_supplied_year) if r.user_supplied_year else "",
        )
    console.print(table)
    console.print(f"\n[dim]{stats()}[/dim]")


@unidentified_app.command("label")
def un_label(
    image_id: str = typer.Argument(...),
    make: str = typer.Option(..., "--make"),
    model: str = typer.Option(..., "--model"),
    year: int = typer.Option(..., "--year"),
) -> None:
    """Apply user-supplied (make, model, year) to an unidentified row."""
    from ccdp.identification.unidentified import label
    row = label(image_id, make=make, model=model, year=year)
    console.print(f"[green]Labeled[/green] {row.image_id} -> {make} {model} {year}")


@unidentified_app.command("stats")
def un_stats() -> None:
    from ccdp.identification.unidentified import stats
    console.print(stats())


# ----------------- costing: import --------------------------------------


@costing_app.command("import")
def costing_import(
    from_dataset: str = typer.Option(None, "--from-dataset",
                                     help="One of: iaai (currently no usable cost; reserved for future)."),
    file: Path = typer.Option(None, "--file",
                              help="CSV file with columns: part,economy,mid,luxury,labor_mid_h"),
    tag: str = typer.Option("import"),
) -> None:
    """Build a new timestamped catalog from a CSV or supported dataset.

    NOTE: `--from-dataset iaai` is currently a no-op because the free iaai
    sample has no real cost values. The command is wired so that when real cost
    data lands (e.g., research-access slice), a single change to this function
    enables data-driven catalog generation.
    """
    if not from_dataset and not file:
        console.print("[red]Provide --from-dataset or --file.[/red]")
        raise typer.Exit(2)
    if from_dataset == "iaai":
        console.print(
            "[yellow]iaai free sample has paywalled cost columns; no catalog "
            "can be derived. See CITATIONS.md (§3) for research-access details. "
            "Falling back to a copy of the active catalog.[/yellow]"
        )
        active = catmod.load_active()
        new = active
        new.catalog_id = catmod.new_catalog_id(tag)
        catmod.save(new, catmod.DEFAULT_ROOT)
        console.print(f"[green]Snapshot saved as[/green] {new.catalog_id}")
        return
    if file:
        new = _catalog_from_csv(file, tag=tag)
        catmod.save(new)
        console.print(f"[green]Created[/green] {new.catalog_id}  from {file}")


def _catalog_from_csv(path: Path, tag: str):
    """CSV columns expected: part,economy,mid,luxury,labor_mid_h."""
    import csv
    from datetime import datetime, timezone

    from ccdp.costing.catalog import Catalog, PartCost, build_seed_catalog
    seed = build_seed_catalog()  # baseline for severity multipliers / labor_rate
    parts = dict(seed.parts)
    with path.open() as f:
        for row in csv.DictReader(f):
            name = row["part"].strip().lower()
            economy = float(row.get("economy") or 0)
            mid = float(row.get("mid") or 0)
            luxury = float(row.get("luxury") or 0)
            labor_mid = float(row.get("labor_mid_h") or seed.parts.get(name, list(seed.parts.values())[0]).labor_hours["moderate"])
            existing = parts.get(name)
            severity_mult = (existing.severity_multiplier if existing
                             else {"minor": 0.4, "moderate": 1.0, "severe": 1.8})
            labor_h = (existing.labor_hours if existing
                       else {"minor": labor_mid * 0.4, "moderate": labor_mid, "severe": labor_mid * 2.0})
            parts[name] = PartCost(
                base_cost={"economy": economy, "mid": mid, "luxury": luxury},
                severity_multiplier=severity_mult,
                labor_hours=labor_h,
            )
    return Catalog(
        catalog_id=catmod.new_catalog_id(tag),
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by="ccdp costing import",
        source=f"imported from {path}",
        currency=seed.currency,
        parts=parts,
        labor_rate_per_hour=seed.labor_rate_per_hour,
        notes="Imported via `ccdp costing import --file`.",
    )


# ----------------- train -------------------------------------------------


@train_app.command("identifier")
def train_identifier(
    epochs_stage1: int = typer.Option(3),
    epochs_stage2: int = typer.Option(12),
    batch_size: int = typer.Option(32),
    lr_stage1: float = typer.Option(1e-3),
    lr_stage2: float = typer.Option(1e-4),
    num_workers: int = typer.Option(4),
    image_size: int = typer.Option(224),
    tag: str = typer.Option("identifier"),
    resume: Path = typer.Option(None, help="Path to a last.pt to resume from."),
    smoke_batches: int = typer.Option(0, help="If >0, cap batches/epoch — for smoke runs."),
) -> None:
    """Fine-tune ResNet50 on Stanford Cars 196 for make/model/year identification."""
    from ccdp.train.train_car_identifier import TrainConfig, train as do_train
    from ccdp.costing import load_active

    try:
        active = load_active()
        catalog_id = active.catalog_id
    except FileNotFoundError:
        catalog_id = None

    cfg = TrainConfig(
        epochs_stage1=epochs_stage1, epochs_stage2=epochs_stage2,
        batch_size=batch_size, lr_stage1=lr_stage1, lr_stage2=lr_stage2,
        num_workers=num_workers, image_size=image_size, tag=tag,
    )
    best = do_train(cfg, resume=resume, smoke_batches=(smoke_batches or None),
                    training_catalog_id=catalog_id)
    console.print(f"[green]Best checkpoint:[/green] {best}")


@train_app.command("extract-features")
def train_extract_features(
    checkpoint: Path = typer.Option(None, help="Classifier checkpoint; defaults to promoted run."),
    out: Path = typer.Option(Path("data/processed/cardd_features.parquet")),
    batch_size: int = typer.Option(64),
    num_workers: int = typer.Option(4),
    smoke_batches: int = typer.Option(0, help="If >0, cap batches per split."),
) -> None:
    """Extract 2048-d features for every CarDD image and cache to parquet."""
    from ccdp.train.extract_features import extract_all
    extract_all(checkpoint=checkpoint, out_path=out, batch_size=batch_size,
                num_workers=num_workers, max_batches=(smoke_batches or None))


@train_app.command("detector")
def train_detector(
    model: str = typer.Option("yolov8n.pt", help="yolov8n.pt | yolov8s.pt | ..."),
    epochs: int = typer.Option(50),
    imgsz: int = typer.Option(640),
    batch: int = typer.Option(16),
    patience: int = typer.Option(15),
    workers: int = typer.Option(4),
    tag: str = typer.Option("yolov8n"),
    device: str = typer.Option(None, help="Override device (mps|cpu|0)."),
) -> None:
    """Train YOLOv8 on CarDD (Variant B detector). Materializes the YOLO dataset if missing."""
    from ccdp.train.train_yolov8 import YoloConfig, train as do_train
    from ccdp.costing import load_active
    try:
        active = load_active()
        catalog_id = active.catalog_id
    except FileNotFoundError:
        catalog_id = None
    cfg = YoloConfig(model=model, epochs=epochs, imgsz=imgsz, batch=batch,
                     patience=patience, workers=workers, tag=tag, device=device)
    best = do_train(cfg, training_catalog_id=catalog_id)
    console.print(f"[green]Best:[/green] {best}")


@train_app.command("build-yolo-dataset")
def train_build_yolo_dataset(
    root: Path = typer.Option(Path("data/processed/yolo")),
) -> None:
    """Materialize CarDD as Ultralytics YOLO dataset (train/val/test with labels)."""
    from ccdp.data import cardd_yolo
    p = cardd_yolo.build(root)
    console.print(f"[green]Wrote[/green] {p}")


@train_app.command("extract-bbox-features")
def train_extract_bbox_features(
    weights: Path = typer.Option(None, help="Detector weights; defaults to promoted run."),
    out: Path = typer.Option(Path("data/processed/cardd_bbox_features.parquet")),
    gt: bool = typer.Option(False, "--gt", help="Use ground-truth CarDD bboxes (no detector)."),
    imgsz: int = typer.Option(640),
    conf: float = typer.Option(0.25),
    smoke_per_split: int = typer.Option(0, help="If >0, cap records per split."),
) -> None:
    """Aggregate bbox stats per image for Variant B XGBoost features."""
    from ccdp.train.extract_bbox_features import extract_from_ground_truth, extract_with_detector
    if gt:
        extract_from_ground_truth(out_path=out)
    else:
        extract_with_detector(weights=weights, out_path=out, imgsz=imgsz, conf=conf,
                              max_records_per_split=(smoke_per_split or None))


@train_app.command("extract-seg-features")
def train_extract_seg_features(
    weights: Path = typer.Option(None, help="yoloseg weights; defaults to promoted run."),
    out: Path = typer.Option(Path("data/processed/cardd_seg_features.parquet")),
    gt: bool = typer.Option(False, "--gt", help="Use CarDD polygon areas (no model)."),
    conf: float = typer.Option(0.25),
    smoke_per_split: int = typer.Option(0, help="If >0, cap records per split."),
) -> None:
    """Aggregate YOLOv8-seg mask-area stats per image for Variant C XGBoost features."""
    from ccdp.train.extract_seg_features import extract_from_ground_truth, extract_with_seg_model
    if gt:
        extract_from_ground_truth(out_path=out)
    else:
        extract_with_seg_model(weights=weights, out_path=out, conf=conf,
                               max_records_per_split=(smoke_per_split or None))


@train_app.command("synth-targets")
def train_synth_targets(
    features_path: Path = typer.Option(Path("data/processed/cardd_features.parquet")),
    out: Path = typer.Option(Path("data/processed/cardd_cost_targets.parquet")),
    seed: int = typer.Option(42),
) -> None:
    """Generate synthetic per-image (metadata + cost_usd) targets from the active catalog."""
    from ccdp.train.synthesize_cost import generate_targets
    generate_targets(features_path, out_path=out, seed=seed)


@train_app.command("xgb")
def train_xgb(
    variant: str = typer.Option("a", help="'a' (image only) | 'b' (+ bbox) | 'c' (+ seg mask area)."),
    n_estimators: int = typer.Option(600),
    max_depth: int = typer.Option(7),
    learning_rate: float = typer.Option(0.05),
    tag: str = typer.Option(None, help="Defaults to xgb_<variant>."),
    features_path: Path = typer.Option(Path("data/processed/cardd_features.parquet")),
    targets_path: Path = typer.Option(Path("data/processed/cardd_cost_targets.parquet")),
    bbox_features_path: Path = typer.Option(Path("data/processed/cardd_bbox_features.parquet")),
    seg_features_path: Path = typer.Option(Path("data/processed/cardd_seg_features.parquet")),
) -> None:
    """Train XGBoost — image features (+ bbox for b / seg mask area for c) + tabular -> cost."""
    if variant not in ("a", "b", "c"):
        console.print("[red]variant must be 'a', 'b', or 'c'.[/red]")
        raise typer.Exit(2)
    from ccdp.train.train_xgb import XGBConfig, train as do_train
    cfg = XGBConfig(n_estimators=n_estimators, max_depth=max_depth,
                    learning_rate=learning_rate,
                    tag=(tag or f"xgb_{variant}"), variant=variant)
    best = do_train(cfg, features_path=features_path, targets_path=targets_path,
                    bbox_features_path=bbox_features_path, seg_features_path=seg_features_path)
    console.print(f"[green]Best:[/green] {best}")


@train_app.command("classifier")
def train_classifier(
    epochs_stage1: int = typer.Option(3),
    epochs_stage2: int = typer.Option(12),
    batch_size: int = typer.Option(32),
    lr_stage1: float = typer.Option(1e-3),
    lr_stage2: float = typer.Option(1e-4),
    num_workers: int = typer.Option(4),
    image_size: int = typer.Option(224),
    tag: str = typer.Option("classifier"),
    resume: Path = typer.Option(None, help="Path to a last.pt to resume from."),
    smoke_batches: int = typer.Option(0, help="If >0, cap batches/epoch — for smoke runs."),
    negative_ratio: float = typer.Option(
        0.0,
        help="Ratio of Stanford Cars 'no damage' images to mix into train+val. "
             "0=legacy CarDD-only; 1.0=balanced; 2.0=2x negatives. Fixes the "
             "'always predicts some damage' false-positive failure mode on "
             "undamaged inputs.",
    ),
) -> None:
    """Fine-tune ResNet50 multi-label damage-type classifier on CarDD (Variant A)."""
    from ccdp.train.train_damage_classifier import TrainConfig as ClsConfig, train as do_train
    from ccdp.costing import load_active

    try:
        active = load_active()
        catalog_id = active.catalog_id
    except FileNotFoundError:
        catalog_id = None

    cfg = ClsConfig(
        epochs_stage1=epochs_stage1, epochs_stage2=epochs_stage2,
        batch_size=batch_size, lr_stage1=lr_stage1, lr_stage2=lr_stage2,
        num_workers=num_workers, image_size=image_size, tag=tag,
        negative_ratio=negative_ratio,
    )
    best = do_train(cfg, resume=resume, smoke_batches=(smoke_batches or None),
                    training_catalog_id=catalog_id)
    console.print(f"[green]Best checkpoint:[/green] {best}")


# ----------------- registry ---------------------------------------------


@registry_app.command("list")
def registry_list(
    variant: str = typer.Option(None, help="Filter by variant."),
) -> None:
    from ccdp.registry import list_entries, production_target
    rows = list_entries(variant=variant)
    if not rows:
        console.print("[yellow]Registry empty.[/yellow]")
        return
    table = Table(title="Registry entries")
    for col in ("variant", "run_id", "created_at", "best_val_acc", "training_catalog_id", "production?"):
        table.add_column(col)
    for r in rows:
        prod = production_target(r["variant"])
        is_prod = "*" if prod and r["run_id"] in str(prod) else ""
        bva = r.get("metrics", {}).get("best_val_acc", "")
        table.add_row(
            r["variant"], r["run_id"], r["created_at"][:19],
            f"{bva:.4f}" if isinstance(bva, (int, float)) else "",
            (r.get("training_catalog_id") or "")[:32],
            is_prod,
        )
    console.print(table)


@app.command()
def infer(
    image: Path = typer.Argument(..., help="Path to a car damage image."),
    model: str = typer.Option("resnet", help="resnet (Variant A) | yolov8 (Variant B) | both"),
    currency: str = typer.Option("USD"),
    threshold: float = typer.Option(0.5, help="Variant A sigmoid threshold."),
    conf: float = typer.Option(0.25, help="Variant B detector confidence threshold."),
    make: str = typer.Option(None),
    model_name: str = typer.Option(None, "--model-name"),
    year: int = typer.Option(None),
    body_type: str = typer.Option("unknown"),
) -> None:
    """End-to-end inference. `--model both` runs A and B side-by-side."""
    if model not in ("resnet", "yolov8", "both"):
        console.print("[red]--model must be resnet | yolov8 | both.[/red]")
        raise typer.Exit(2)
    from ccdp.identification.car_identifier import IdentificationResult, infer_segment
    metadata = None
    if make:
        metadata = IdentificationResult(
            image_path=image, make=make.lower(),
            model=(model_name.lower() if model_name else None),
            year=year, body_type=body_type,
            segment=infer_segment(make), confidence=1.0, source="user",
        )
    import json as _json
    out: dict = {}
    if model in ("resnet", "both"):
        from ccdp.infer.variant_a import VariantAPipeline
        pipe_a = VariantAPipeline()
        out["variant_a"] = pipe_a.predict(image, metadata=metadata,
                                          threshold=threshold, currency=currency).to_dict()
    if model in ("yolov8", "both"):
        from ccdp.infer.variant_b import VariantBPipeline
        pipe_b = VariantBPipeline(conf=conf)
        out["variant_b"] = pipe_b.predict(image, metadata=metadata, currency=currency).to_dict()
    console.print_json(_json.dumps(out, default=str))


@registry_app.command("promote")
def registry_promote(
    run_id: str = typer.Argument(...),
    variant: str = typer.Argument(...),
    weights: str = typer.Option("best.pt"),
) -> None:
    from ccdp.registry import promote
    link = promote(run_id, variant=variant, weights_filename=weights)
    console.print(f"[green]Promoted[/green] {run_id} -> {link}")


# ----------------- serve ------------------------------------------------


@serve_app.command("api")
def serve_api(
    host: str = typer.Option("127.0.0.1", help="Bind address. Use 0.0.0.0 to expose."),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="uvicorn auto-reload (dev only)."),
) -> None:
    """Run the FastAPI inference service."""
    import uvicorn
    console.print(f"[bold]Starting ccdp API[/bold] on http://{host}:{port}")
    uvicorn.run("ccdp.api.server:app", host=host, port=port, reload=reload)


@serve_app.command("demo")
def serve_demo(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(7860),
    share: bool = typer.Option(False, help="Gradio public share link."),
) -> None:
    """Run the Gradio demo."""
    from ccdp.api.demo import build_demo
    import gradio as gr
    demo = build_demo()
    demo.launch(
        server_name=host, server_port=port, share=share, show_error=True,
        theme=gr.themes.Soft(),
    )


# ----------------- report -----------------------------------------------


@report_app.command("generate")
def report_generate(
    variant: str = typer.Option("both", help="a | b | both"),
    limit: int = typer.Option(0, help="Cap test images (0 = all). Smoke runs use a small value."),
    no_pdf: bool = typer.Option(False, "--no-pdf"),
) -> None:
    """Build the Variant-A-vs-B comparison report (HTML always, PDF optional)."""
    from ccdp.eval import build_comparison, report as report_mod
    from ccdp.infer.variant_a import VariantAPipeline

    pipe_a = VariantAPipeline() if variant in ("a", "both") else None
    pipe_b = None
    if variant in ("b", "both"):
        try:
            from ccdp.infer.variant_b import VariantBPipeline
            pipe_b = VariantBPipeline()
        except FileNotFoundError as e:
            console.print(f"[yellow]Variant B unavailable: {e}[/yellow]")

    if pipe_a is None and pipe_b is None:
        console.print("[red]No pipelines available — nothing to report.[/red]")
        raise typer.Exit(2)
    # If user asked only for B but B failed, fall back to A-only
    if pipe_a is None and pipe_b is not None:
        pipe_a = pipe_b
        pipe_b = None

    cmp = build_comparison(pipe_a, pipe_b, limit=(limit or None))
    paths = report_mod.generate(cmp, also_pdf=not no_pdf)
    console.print(f"[green]HTML:[/green] {paths['html']}")
    if paths.get("pdf"):
        console.print(f"[green]PDF: [/green] {paths['pdf']}")


if __name__ == "__main__":  # pragma: no cover
    app()
