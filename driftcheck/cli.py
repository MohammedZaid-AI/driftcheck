"""DriftCheck command line interface."""

from __future__ import annotations

import typer

from driftcheck import __version__
from driftcheck.config import load_settings
from driftcheck.log import setup_logging

app = typer.Typer(help="Check if an AI generated product image changed the original product.", no_args_is_help=True)


@app.callback()
def main() -> None:
    setup_logging(load_settings().log_level)


def _not_ready(phase: int) -> None:
    typer.echo(f"Not built yet. This command arrives in phase {phase}.", err=True)
    raise typer.Exit(code=2)


@app.command()
def info() -> None:
    """Show the version, model names and whether the API key is set."""
    settings = load_settings()
    typer.echo(f"driftcheck {__version__}")
    typer.echo(f"vision model: {settings.models.vision}")
    typer.echo(f"fast model:   {settings.models.fast}")
    typer.echo(f"image model:  {settings.models.image}")
    typer.echo(f"data dir:     {settings.data_dir}")
    typer.echo(f"cache dir:    {settings.cache_dir}")
    typer.echo(f"API key set:  {'yes' if settings.has_api_key else 'no'}")


@app.command()
def ingest() -> None:
    """Move photos from data/inbox into product folders."""
    _not_ready(1)


@app.command()
def generate() -> None:
    """Make AI outputs for every product."""
    _not_ready(1)


@app.command()
def register() -> None:
    """Record outputs saved by hand."""
    _not_ready(1)


@app.command()
def synth() -> None:
    """Make synthetic drift with automatic labels."""
    _not_ready(2)


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
