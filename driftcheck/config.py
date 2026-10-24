"""Settings loaded from .env and config/models.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelNames(BaseModel):
    """Gemini model names. Change them in config/models.json."""

    vision: str  # reads photos: product card and judge
    fast: str  # cheap calls such as the category tag
    image: str  # image generation


class Settings(BaseModel):
    gemini_api_key: str | None
    models: ModelNames
    root: Path
    data_dir: Path
    cache_dir: Path
    config_dir: Path
    log_level: str

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_settings(root: Path | None = None) -> Settings:
    """Load settings. Values already in the environment win over .env."""
    root = root or PROJECT_ROOT
    load_dotenv(root / ".env", override=False)

    config_dir = root / "config"
    models = ModelNames(**json.loads((config_dir / "models.json").read_text(encoding="utf-8")))

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        models=models,
        root=root,
        data_dir=_resolve(root, os.getenv("DRIFTCHECK_DATA_DIR", "data")),
        cache_dir=_resolve(root, os.getenv("DRIFTCHECK_CACHE_DIR", "cache")),
        config_dir=config_dir,
        log_level=os.getenv("DRIFTCHECK_LOG_LEVEL", "INFO").upper(),
    )
