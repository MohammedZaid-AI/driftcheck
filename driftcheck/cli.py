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
    result = ingest_inbox(settings.data_dir, client)
    typer.echo(f"Added {len(result.added)} products: {' '.join(result.added) or 'none'}")
    if result.categorised:
        typer.echo(f"Categorised {len(result.categorised)} products.")
    if result.duplicates:
        typer.echo(f"Skipped {len(result.duplicates)} duplicates: {' '.join(result.duplicates)}")
    if result.skipped:
        typer.echo(f"Skipped {len(result.skipped)} files that are not jpg, png or webp: {' '.join(result.skipped)}")
    if result.category_failed:
        typer.echo(f"No category yet for {' '.join(result.category_failed)}. Run ingest again to retry.", err=True)


@app.command()
def jobs(
    generator: str = typer.Option("qwen_edit", help="qwen_edit for the Kaggle notebook, gemini_app for the manual route."),
) -> None:
    """Write the generation jobs: a zip for Kaggle and a checklist for the Gemini app."""
    from driftcheck.dataset import list_product_ids, read_meta
    from driftcheck.register import export_jobs, make_jobs

    settings = load_settings()
    missing = [pid for pid in list_product_ids(settings.data_dir) if read_meta(settings.data_dir, pid).category is None]
    if missing:
        typer.echo(f"Warning: no category for {' '.join(missing)}, so their prompts are generic. Run ingest first.", err=True)
    count = len(make_jobs(settings.data_dir, generator))
    out = export_jobs(settings.data_dir, generator)
    typer.echo(f"{count} jobs for {generator}.")
    typer.echo(f"Kaggle input: {out / 'kaggle_input.zip'}")
    typer.echo(f"Manual checklist: {out / 'manual_checklist.md'}")


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
) -> None:
    """Split products 50 50 into calibration and test."""
    from driftcheck.split import write_split

    settings = load_settings()
    try:
        result = write_split(settings.data_dir, seed=seed, force=force)
    except FileExistsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1)
    if force:
        typer.echo("Split replaced. Retune thresholds and say so in PROGRESS.md.", err=True)
    typer.echo(f"calibration: {len(result['calibration'])} products, test: {len(result['test'])} products")


@app.command()
def synth(seed: int = typer.Option(42, help="Random seed for the drift parameters.")) -> None:
    """Make synthetic drift with automatic labels in data/synthetic."""
    from collections import Counter

    from driftcheck.cache import DiskCache
    from driftcheck.ocr import OcrEngine
    from driftcheck.segment import product_mask
    from driftcheck.synth import run_synth

    settings = load_settings()
    mask_cache = DiskCache(settings.cache_dir, "masks")
    ocr = OcrEngine(DiskCache(settings.cache_dir, "ocr"))
    rows = run_synth(settings.data_dir, lambda image: product_mask(image, mask_cache), ocr.read, seed=seed)
    counts = Counter(row["output_id"] for row in rows)
    typer.echo(f"Wrote {len(rows)} synthetic outputs to {settings.data_dir / 'synthetic'}")
    for drift, count in sorted(counts.items()):
        typer.echo(f"  {drift}: {count}")


@app.command()
def check() -> None:
    """Check one original against one output."""
    _not_ready(5)


@app.command(name="eval")
def evaluate() -> None:
    """Run the full evaluation."""
    _not_ready(6)


if __name__ == "__main__":
    app()
