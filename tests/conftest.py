from __future__ import annotations

import os
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture
def plain_shirt() -> Path:
    """Real photo of a plain black long sleeve shirt on a hanger. The plain control example."""
    return SAMPLES / "plain_shirt.jpg"


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
