from __future__ import annotations

from typer.testing import CliRunner

from driftcheck.cli import app

runner = CliRunner()


def test_info_shows_models_and_hides_key() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "vision model:" in result.output
    assert "API key set:  no" in result.output


def test_unbuilt_command_says_which_phase() -> None:
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "phase 6" in result.output
