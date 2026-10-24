from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def no_real_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call real APIs, so they never see a real key."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    for name in list(os.environ):
        if name.startswith("DRIFTCHECK_"):
            monkeypatch.delenv(name)
