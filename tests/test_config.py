from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from driftcheck.config import PROJECT_ROOT, load_settings


def _make_root(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    vision = {
        "provider": "ollama",
        "models": {"ollama": "local-small", "gemini": "g-model", "hf": "org/hf-model"},
        "compare_models": ["local-small", "local-big"],
        "max_side": 512,
        "timeout_seconds": 30,
    }
    (tmp_path / "config" / "vision.json").write_text(json.dumps(vision))
    return tmp_path


def test_defaults_from_config(tmp_path: Path) -> None:
    settings = load_settings(_make_root(tmp_path))
    assert settings.vision.provider == "ollama"
    assert settings.vision.model == "local-small"
    assert settings.vision.compare_models == ["local-small", "local-big"]
    assert settings.vision.max_side == 512
    assert settings.ollama_host == "http://localhost:11434"


def test_provider_switch_picks_that_providers_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "hf")
    settings = load_settings(_make_root(tmp_path))
    assert settings.vision.provider == "hf"
    assert settings.vision.model == "org/hf-model"


def test_model_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_MODEL", "local-big")
    assert load_settings(_make_root(tmp_path)).vision.model == "local-big"


def test_unknown_provider_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "openai")
    with pytest.raises(ValidationError):
        load_settings(_make_root(tmp_path))


def test_paths_and_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIFTCHECK_CACHE_DIR", "elsewhere")
    monkeypatch.setenv("DRIFTCHECK_LOG_LEVEL", "debug")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    settings = load_settings(_make_root(tmp_path))
    assert settings.data_dir == tmp_path / "data"
    assert settings.cache_dir == tmp_path / "elsewhere"
    assert settings.log_level == "DEBUG"
    assert settings.ollama_host == "http://gpu-box:11434"


def test_max_images_is_per_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_root(tmp_path)
    raw = json.loads((root / "config" / "vision.json").read_text())
    raw["max_images"] = {"groq": 2}
    raw["models"]["groq"] = "g"
    (root / "config" / "vision.json").write_text(json.dumps(raw))
    assert load_settings(root).vision.max_images is None
    monkeypatch.setenv("VISION_PROVIDER", "groq")
    assert load_settings(root).vision.max_images == 2


def test_keys_are_optional(tmp_path: Path) -> None:
    settings = load_settings(_make_root(tmp_path))
    assert settings.gemini_api_key is None
    assert settings.hf_token is None


def test_project_config_defaults_to_local_qwen() -> None:
    settings = load_settings(PROJECT_ROOT)
    assert settings.vision.provider == "ollama"
    assert settings.vision.model == "qwen2.5vl:3b"
    assert "qwen2.5vl:7b" in settings.vision.compare_models
