from __future__ import annotations

import pytest
from typer.testing import CliRunner

from driftcheck import vision
from driftcheck.cli import app

runner = CliRunner()


def test_info_shows_provider_and_model() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "vision provider: ollama" in result.output
    assert "vision model:    qwen2.5vl:3b" in result.output


def test_ping_reports_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Down:
        def ensure_ready(self) -> None:
            raise vision.VisionUnavailable("Ollama is not running at http://localhost:11434.")

    monkeypatch.setattr(vision, "get_vision_client", lambda: Down())
    result = runner.invoke(app, ["ping"])
    assert result.exit_code == 1
    assert "Ollama is not running" in result.output


def test_unbuilt_command_says_which_phase() -> None:
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "phase 6" in result.output
