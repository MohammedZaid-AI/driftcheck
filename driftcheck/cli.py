"""DriftCheck command line interface."""

from __future__ import annotations

import typer

from driftcheck import __version__
from driftcheck.config import load_settings, use_system_certificates
from driftcheck.log import setup_logging

app = typer.Typer(help="Check if an AI generated product image changed the original product.", no_args_is_help=True)


@app.callback()
def main() -> None:
    use_system_certificates()
    setup_logging(load_settings().log_level)


def _not_ready(phase: int) -> None:
    typer.echo(f"Not built yet. This command arrives in phase {phase}.", err=True)
    raise typer.Exit(code=2)


@app.command()
def info() -> None:
    """Show the version, the vision provider and the folders in use."""
    settings = load_settings()
    typer.echo(f"driftcheck {__version__}")
    typer.echo(f"vision provider: {settings.vision.provider}")
    typer.echo(f"vision model:    {settings.vision.model}")
    if settings.vision.provider == "ollama":
        typer.echo(f"ollama host:     {settings.ollama_host}")
    typer.echo(f"data dir:        {settings.data_dir}")
    typer.echo(f"cache dir:       {settings.cache_dir}")


@app.command()
def ping() -> None:
    """Check that the vision provider is running and the model is ready."""
    from driftcheck.vision import VisionUnavailable, get_vision_client

    try:
        client = get_vision_client()
        client.ensure_ready()
    except VisionUnavailable as error:
        typer.echo(f"Not ready: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Ready: {client.provider.name} {client.model}")


@app.command()
def ingest(
    no_vision: bool = typer.Option(False, help="Only move photos. Fill categories later with another ingest."),
    refresh: bool = typer.Option(False, help="Ask again for the category and worn flag of every product."),
) -> None:
    """Move photos from data/inbox into product folders and fill the category."""
    from driftcheck.ingest import ingest_inbox
    from driftcheck.vision import VisionUnavailable, get_vision_client

    settings = load_settings()
    client = None
    if not no_vision:
        client = get_vision_client(settings)
        try:
            client.ensure_ready()
        except VisionUnavailable as error:
            typer.echo(f"Not ready: {error}", err=True)
            raise typer.Exit(code=1)
    result = ingest_inbox(settings.data_dir, client, refresh=refresh)
    typer.echo(f"Added {len(result.added)} products: {' '.join(result.added) or 'none'}")
    if result.categorised:
        typer.echo(f"Categorised {len(result.categorised)} products.")
    if result.duplicates:
        typer.echo(f"Skipped {len(result.duplicates)} duplicates: {' '.join(result.duplicates)}")
    if result.skipped:
        typer.echo(f"Skipped {len(result.skipped)} files that are not jpg, png, webp or avif: {' '.join(result.skipped)}")
    if result.blank:
        typer.echo(f"Skipped {len(result.blank)} blank images that are one flat colour: {' '.join(result.blank)}", err=True)
    if result.category_failed:
        typer.echo(f"No category yet for {' '.join(result.category_failed)}. Run ingest again to retry.", err=True)
    _print_category_summary(settings.data_dir)


def _print_category_summary(data_dir) -> None:
    from collections import Counter

    from driftcheck.dataset import list_product_ids, read_meta

    metas = [read_meta(data_dir, pid) for pid in list_product_ids(data_dir)]
    name = lambda meta: meta.category or "no category yet"  # noqa: E731
    counts = Counter(name(meta) for meta in metas)
    worn = Counter(name(meta) for meta in metas if meta.worn)
    core = Counter(name(meta) for meta in metas if meta.subset == "core")
    typer.echo(f"\n{len(metas)} products, {sum(core.values())} core")
    typer.echo(f"  {'category':16s} {'products':>8s} {'core':>6s} {'worn':>6s}")
    for category, count in counts.most_common():
        typer.echo(f"  {category:16s} {count:8d} {core[category]:6d} {worn[category]:6d}")


@app.command()
def select() -> None:
    """Apply config/selection.json: hand fixes, and the core and extra tags."""
    from driftcheck.selection import apply_selection, load_selection

    settings = load_settings()
    apply_selection(settings.data_dir, load_selection(settings.config_dir))
    _print_category_summary(settings.data_dir)


@app.command()
def jobs(
    generator: str = typer.Option("gemini_app", help="gemini_app for the manual route, qwen_edit for the Kaggle notebook."),
    include_extra: bool = typer.Option(False, "--include-extra", help="Also make jobs for products tagged extra."),
    all_styles: bool = typer.Option(False, "--all-styles", help="Three styles per product instead of two."),
) -> None:
    """Write the generation jobs for the core products: a checklist for the Gemini app and a zip for Kaggle."""
    from driftcheck.dataset import read_meta, selected_product_ids
    from driftcheck.register import export_jobs, make_jobs

    settings = load_settings()
    products = selected_product_ids(settings.data_dir, include_extra)
    missing = [pid for pid in products if read_meta(settings.data_dir, pid).category is None]
    if missing:
        typer.echo(f"Warning: no category for {' '.join(missing)}, so their prompts are generic. Run ingest first.", err=True)
    count = len(make_jobs(settings.data_dir, generator, include_extra, all_styles))
    out = export_jobs(settings.data_dir, generator, include_extra, all_styles)
    typer.echo(f"{count} jobs for {generator} across {len(products)} products{' including extras' if include_extra else ''}.")
    typer.echo(f"Manual checklist: {out / 'manual_checklist.md'}")
    typer.echo(f"Kaggle input: {out / 'kaggle_input.zip'}")


@app.command()
def register(
    source: str = typer.Option(None, "--from", help="Unzipped Kaggle result folder to import first."),
    model: str = typer.Option(None, help="Model name to record when the output has no json of its own."),
) -> None:
    """Record outputs made in Kaggle or the Gemini app."""
    from pathlib import Path

    from driftcheck.register import register_outputs

    settings = load_settings()
    result = register_outputs(settings.data_dir, model=model, source=Path(source) if source else None)
    if source:
        typer.echo(f"Imported {len(result.imported)} images from {source}.")
    typer.echo(f"Registered {len(result.registered)} new outputs. {len(result.already)} were already registered.")
    if result.bad_names:
        typer.echo("These break the naming rule <style>__<generator>.jpg and were skipped:", err=True)
        for name in result.bad_names:
            typer.echo(f"  {name}", err=True)


@app.command()
def label(port: int = typer.Option(7861, help="Port for the labelling app.")) -> None:
    """Open the labelling app in the browser. Every answer is saved to data/labels.json at once."""
    from driftcheck.apps.label_app import build_app

    build_app(load_settings().data_dir).launch(server_port=port, inbrowser=True)


@app.command()
def split(
    seed: int = typer.Option(42, help="Random seed for the shuffle."),
    force: bool = typer.Option(False, help="Overwrite an existing split. Thresholds must then be retuned."),
    include_extra: bool = typer.Option(False, "--include-extra", help="Also split products tagged extra."),
) -> None:
    """Split the core products 50 50 into calibration and test."""
    from driftcheck.split import write_split

    settings = load_settings()
    try:
        result = write_split(settings.data_dir, seed=seed, force=force, include_extra=include_extra)
    except FileExistsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1)
    if force:
        typer.echo("Split replaced. Retune thresholds and say so in PROGRESS.md.", err=True)
    typer.echo(f"calibration: {len(result['calibration'])} products, test: {len(result['test'])} products")


@app.command()
def synth(
    seed: int = typer.Option(42, help="Random seed for the drift parameters."),
    include_extra: bool = typer.Option(False, "--include-extra", help="Also make drift for products tagged extra."),
) -> None:
    """Make synthetic drift with automatic labels in data/synthetic, for the core products."""
    from collections import Counter

    from driftcheck.cache import DiskCache
    from driftcheck.ocr import OcrEngine
    from driftcheck.segment import product_mask
    from driftcheck.synth import run_synth

    settings = load_settings()
    mask_cache = DiskCache(settings.cache_dir, "masks")
    ocr = OcrEngine(DiskCache(settings.cache_dir, "ocr"))
    rows = run_synth(settings.data_dir, lambda image: product_mask(image, mask_cache), ocr.read, seed=seed,
                     include_extra=include_extra)
    counts = Counter(row["output_id"] for row in rows)
    typer.echo(f"Wrote {len(rows)} synthetic outputs to {settings.data_dir / 'synthetic'}")
    for drift, count in sorted(counts.items()):
        typer.echo(f"  {drift}: {count}")


@app.command()
def card(image: str = typer.Argument(..., help="Path to a product photo.")) -> None:
    """Print the product card for one photo as JSON."""
    from PIL import Image

    from driftcheck.cache import DiskCache
    from driftcheck.card import build_card
    from driftcheck.ocr import OcrEngine
    from driftcheck.segment import product_mask
    from driftcheck.vision import get_vision_client

    settings = load_settings()
    masks = DiskCache(settings.cache_dir, "masks")
    ocr = OcrEngine(DiskCache(settings.cache_dir, "ocr"))
    photo = Image.open(image)
    scripts = ocr.detect_scripts(photo)
    result = build_card(
        photo,
        get_vision_client(settings),
        lambda im: product_mask(im, masks),
        lambda im: ocr.read(im, scripts),
        DiskCache(settings.cache_dir, "cards"),
        scripts,
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def scripts(
    include_extra: bool = typer.Option(False, "--include-extra", help="Also detect for products tagged extra."),
    refresh: bool = typer.Option(False, help="Detect again even where text_scripts is set. Hand set values stay."),
) -> None:
    """Detect the text scripts on each product's original and save them as text_scripts."""
    from driftcheck.cache import DiskCache
    from driftcheck.dataset import read_meta
    from driftcheck.ocr import OcrEngine
    from driftcheck.scripts import detect_product_scripts

    settings = load_settings()
    ocr = OcrEngine(DiskCache(settings.cache_dir, "ocr"))
    found = detect_product_scripts(settings.data_dir, ocr.detect_scripts, include_extra, refresh)
    for product_id, value in found.items():
        hand = " (set by hand)" if "text_scripts" in read_meta(settings.data_dir, product_id).manual_fields else ""
        typer.echo(f"  {product_id}  {', '.join(value) or 'no text'}{hand}")


@app.command()
def check(
    original: str = typer.Option(..., help="Path to the original product photo."),
    output: str = typer.Option(..., help="Path to the AI output."),
    out: str = typer.Option("reports/check", help="Folder for report.json and heatmap.png."),
    model: str = typer.Option(None, help="Vision model to use instead of the configured one, for example qwen2.5vl:7b."),
    text_scripts: str = typer.Option(None, "--scripts", help="Scripts on the product, for example en,hi. Detected if not given."),
) -> None:
    """Check one original against one output. Writes report.json and heatmap.png."""
    from pathlib import Path

    from driftcheck.heatmap import draw_heatmap
    from driftcheck.pipeline import Checker, load_image
    from driftcheck.vision import VisionUnavailable

    checker = Checker.from_settings(vision_model=model)
    try:
        checker.client.ensure_ready()
    except VisionUnavailable as error:
        typer.echo(f"Vision model not ready, so shape, logo and the card will be unknown: {error}", err=True)
    report = checker.check(original, output, text_scripts.split(",") if text_scripts else None)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    draw_heatmap(load_image(original), load_image(output), report).save(folder / "heatmap.png")

    typer.echo(f"Verdict: {report.verdict}  ({report.seconds:.1f}s, {report.vision_calls} vision calls)")
    for result in report.checks:
        note = "" if result.applicable else " (not applicable)"
        typer.echo(f"  {result.name:16s} {result.status:8s}{note} {result.reason}")
    typer.echo(f"Wrote {folder / 'report.json'} and {folder / 'heatmap.png'}")


@app.command(name="eval-run")
def eval_run(
    model: str = typer.Option(None, help="Vision model, for example qwen2.5vl:7b. Default is the configured one."),
    test_only: bool = typer.Option(False, "--test-only", help="Only the test split, for a model comparison run."),
) -> None:
    """Run every check and both baselines once on every labelled and synthetic pair, saving raw results.
    Pairs already run are skipped, so it can be stopped and started again."""
    from driftcheck.evaluate import collect_items, recording_thresholds, run_all
    from driftcheck.pipeline import Checker
    from driftcheck.vision import VisionUnavailable

    settings = load_settings()
    items = [i for i in collect_items(settings.data_dir) if not test_only or i.split == "test"]
    checker = Checker.from_settings(settings, vision_model=model)
    checker.thresholds = recording_thresholds(checker.thresholds)
    try:
        checker.client.ensure_ready()
    except VisionUnavailable as error:
        typer.echo(f"Not ready: {error}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{len(items)} pairs with {checker.client.model}")
    run_all(items, checker, checker.features_fn, settings.cache_dir, checker.client.model)
    typer.echo("Done. Run driftcheck eval to tune and write the report.")


@app.command(name="eval")
def evaluate(
    retune: bool = typer.Option(False, help="Tune again on the calibration split even though thresholds are frozen."),
) -> None:
    """Tune on the calibration split if not done yet, then write reports/eval_report.md and .json."""
    import json

    from driftcheck.checks.base import load_thresholds
    from driftcheck.eval_report import build_report, gallery_markdown, render_markdown, write_gallery
    from driftcheck.evaluate import collect_items, load_raw, tune, tune_baseline_1
    from driftcheck.ocr import READERS
    from driftcheck.segment import REMBG_MODEL

    settings = load_settings()
    items = collect_items(settings.data_dir)
    if not any(i.split == "calibration" for i in items) or not any(i.split == "test" for i in items):
        typer.echo("Need labelled or synthetic pairs in both the calibration and test split. Nothing was tuned or written.", err=True)
        raise typer.Exit(code=1)
    primary = settings.vision.model
    raws = load_raw(items, settings.cache_dir, primary)
    missing = [i.key for i in items if i.key not in raws]
    if missing:
        typer.echo(f"{len(missing)} of {len(items)} pairs have no saved run for {primary}. Run driftcheck eval-run first.", err=True)
        raise typer.Exit(code=1)

    thresholds_path = settings.config_dir / "thresholds.json"
    notes_path = settings.root / "reports" / "tuning.json"
    t = load_thresholds(settings.config_dir)
    if not t.tuned or retune:
        if t.tuned:
            typer.echo("Retuning frozen thresholds. Say so in PROGRESS.md.", err=True)
        calibration = [i for i in items if i.split == "calibration"]
        t, notes = tune(calibration, raws, t)
        b1_threshold, b1_note = tune_baseline_1(calibration, raws)
        frozen = {"tuned": True, "tuned_on": f"calibration split, {len(calibration)} pairs, vision model {primary}",
                  "note": "Frozen. Do not change after tuning unless the split changes; then retune and say so in PROGRESS.md.",
                  **{k: v for k, v in t.model_dump().items() if k != "tuned"},
                  "baseline_1": {"similarity": b1_threshold}}
        thresholds_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        notes_path.parent.mkdir(exist_ok=True)
        notes_path.write_text(json.dumps({"checks": notes, "baseline_1": b1_note}, indent=1), encoding="utf-8")
        typer.echo(f"Tuned on {len(calibration)} calibration pairs and froze config/thresholds.json")
    saved = json.loads(thresholds_path.read_text(encoding="utf-8"))
    tuning = json.loads(notes_path.read_text(encoding="utf-8"))

    raws_by_model = {primary: raws}
    for model in settings.vision.compare_models:
        if model != primary:
            other = load_raw([i for i in items if i.split == "test"], settings.cache_dir, model)
            if other:
                raws_by_model[model] = other
    scalio = settings.data_dir / "scalio_examples"
    scalio_note = ("No images in data/scalio_examples, so this part was not run."
                   if not scalio.exists() or not any(scalio.iterdir()) else "Scalio examples are present but not evaluated yet.")
    cost = ("Vision calls run on a local Ollama model, so there is no API charge. Calls per image are in the tables."
            if settings.vision.provider == "ollama" else "Not estimated: no pricing is set in the config.")
    models = {"vision": f"{settings.vision.provider}/{primary}",
              "compared vision models": ", ".join(raws_by_model) or primary,
              "pattern and baseline 1": settings.vision.pattern_model,
              "segmentation": f"rembg/{REMBG_MODEL}", "ocr": "easyocr readers " + ", ".join(READERS),
              "outputs": "Gemini app, model 3.6 Flash, made by hand"}
    report = build_report(settings.root, items, raws_by_model, primary, t, tuning["checks"],
                          saved["baseline_1"]["similarity"], tuning["baseline_1"], models, cost, scalio_note)

    out = settings.root / "reports"
    entries = write_gallery(out / "gallery", items, raws, t)
    report["failure_gallery"] = entries
    (out / "eval_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    markdown = render_markdown(report) + "\n## Failure gallery\n\n" + gallery_markdown(entries)
    (out / "eval_report.md").write_text(markdown, encoding="utf-8")
    typer.echo(f"Wrote {out / 'eval_report.md'} and {out / 'eval_report.json'}")


if __name__ == "__main__":
    app()
