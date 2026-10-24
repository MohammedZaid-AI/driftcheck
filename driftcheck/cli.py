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
def ingest() -> None:
    """Move photos from data/inbox into product folders."""
    _not_ready(1)


@app.command()
def register() -> None:
    """Record outputs made in Kaggle or the Gemini app."""
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
