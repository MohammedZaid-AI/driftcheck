from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftcheck.config import PROJECT_ROOT, load_settings


def _make_root(tmp_path: Path, env_text: str = "") -> Path:
    (tmp_path / "config").mkdir()
    models = {"vision": "v-model", "fast": "f-model", "image": "i-model"}
    (tmp_path / "config" / "models.json").write_text(json.dumps(models))
    if env_text:
        (tmp_path / ".env").write_text(env_text)
    return tmp_path


def test_reads_model_names_from_config(tmp_path: Path) -> None:
    settings = load_settings(_make_root(tmp_path))
    assert settings.models.vision == "v-model"
    assert settings.models.fast == "f-model"
    assert settings.models.image == "i-model"


def test_paths_default_inside_root(tmp_path: Path) -> None:
    settings = load_settings(_make_root(tmp_path))
    assert settings.data_dir == tmp_path / "data"
    assert settings.cache_dir == tmp_path / "cache"


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIFTCHECK_CACHE_DIR", "elsewhere")
    monkeypatch.setenv("DRIFTCHECK_LOG_LEVEL", "debug")
    settings = load_settings(_make_root(tmp_path))
    assert settings.cache_dir == tmp_path / "elsewhere"
    assert settings.log_level == "DEBUG"


def test_missing_key_means_no_key(tmp_path: Path) -> None:
    settings = load_settings(_make_root(tmp_path))
    assert settings.has_api_key is False


def test_project_config_has_all_models() -> None:
    settings = load_settings(PROJECT_ROOT)
    assert settings.models.vision and settings.models.fast and settings.models.image
