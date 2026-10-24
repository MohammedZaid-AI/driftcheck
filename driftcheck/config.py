"""Settings loaded from .env and config/vision.json."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ProviderName = Literal["ollama", "gemini", "hf"]


class VisionConfig(BaseModel):
    """Which vision model to call. Change it in config/vision.json or with
    VISION_PROVIDER and VISION_MODEL in .env."""

    provider: ProviderName
    model: str  # the model for the chosen provider, after overrides
    models: dict[str, str]
    compare_models: list[str]
    max_side: int
    timeout_seconds: float


class Settings(BaseModel):
    vision: VisionConfig
    ollama_host: str
    gemini_api_key: str | None
    hf_token: str | None
    root: Path
    data_dir: Path
    cache_dir: Path
    config_dir: Path
    log_level: str


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_vision(config_dir: Path) -> VisionConfig:
    raw = json.loads((config_dir / "vision.json").read_text(encoding="utf-8"))
    provider = os.getenv("VISION_PROVIDER") or raw["provider"]
    model = os.getenv("VISION_MODEL") or raw["models"].get(provider, "")
    return VisionConfig(
        provider=provider,
        model=model,
        models=raw["models"],
        compare_models=raw.get("compare_models", []),
        max_side=raw.get("max_side", 1024),
        timeout_seconds=raw.get("timeout_seconds", 600),
    )


def load_settings(root: Path | None = None) -> Settings:
    """Load settings. Values already in the environment win over .env."""
    root = root or PROJECT_ROOT
    load_dotenv(root / ".env", override=False)
    config_dir = root / "config"

    return Settings(
        vision=_load_vision(config_dir),
        ollama_host=os.getenv("OLLAMA_HOST") or "http://localhost:11434",
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        hf_token=os.getenv("HF_TOKEN") or None,
        root=root,
        data_dir=_resolve(root, os.getenv("DRIFTCHECK_DATA_DIR", "data")),
        cache_dir=_resolve(root, os.getenv("DRIFTCHECK_CACHE_DIR", "cache")),
        config_dir=config_dir,
        log_level=os.getenv("DRIFTCHECK_LOG_LEVEL", "INFO").upper(),
    )
