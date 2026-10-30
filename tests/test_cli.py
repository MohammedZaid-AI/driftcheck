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


def test_eval_refuses_with_nothing_to_tune_and_writes_nothing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from driftcheck.config import PROJECT_ROOT

    thresholds = PROJECT_ROOT / "config" / "thresholds.json"
    before = thresholds.read_bytes()
    monkeypatch.setenv("DRIFTCHECK_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRIFTCHECK_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "split.json").write_text(json.dumps({"calibration": [], "test": []}))
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 1 and "Nothing was tuned" in result.output
    assert thresholds.read_bytes() == before
