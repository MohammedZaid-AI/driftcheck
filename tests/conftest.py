from __future__ import annotations

import os

import pytest

_ENV_NAMES = ("GEMINI_API_KEY", "HF_TOKEN", "VISION_PROVIDER", "VISION_MODEL", "OLLAMA_HOST")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call real APIs, so they never see real keys or local overrides.
    Empty values stop load_dotenv from filling them in from .env."""
    for name in _ENV_NAMES:
        monkeypatch.setenv(name, "")
    for name in list(os.environ):
        if name.startswith("DRIFTCHECK_"):
            monkeypatch.delenv(name)
