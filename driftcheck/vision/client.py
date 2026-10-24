"""VisionClient adds the disk cache, retries and call logging to any provider."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from pydantic import BaseModel

from driftcheck.cache import DiskCache, make_key
from driftcheck.config import Settings, load_settings
from driftcheck.log import get_logger
from driftcheck.retry import retry_call
from driftcheck.vision.base import ImageInput, Provider, VisionError, VisionUnavailable, prepare_image

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


@dataclass
class CallRecord:
    provider: str
    model: str
    seconds: float
    cached: bool
    ok: bool


def _strip_fences(text: str) -> str:
    """Some models wrap JSON in ```json fences even when asked not to."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


class VisionClient:
    def __init__(
        self,
        provider: Provider,
        cache: DiskCache,
        max_side: int = 1024,
        log_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.max_side = max_side
        self.log_path = log_path
        self.calls: list[CallRecord] = []
        self._sleep = sleep
        self._ready = False

    @property
    def model(self) -> str:
        return self.provider.model

    def ensure_ready(self) -> None:
        """Check the provider once, before the first real call."""
        if not self._ready:
            self.provider.check()
            self._ready = True

    def ask(self, prompt: str, images: Sequence[ImageInput], schema: type[T]) -> T:
        """Ask the model about the images and parse the answer into schema."""
        prepared = [prepare_image(image, self.max_side) for image in images]
        schema_text = json.dumps(schema.model_json_schema(), sort_keys=True)
        key = make_key(self.provider.name, self.provider.model, prompt, schema_text, *prepared)

        cached = self.cache.get(key)
        if cached is not None:
            self._record(0.0, cached=True, ok=True)
            return schema.model_validate(cached)

        self.ensure_ready()
        start = time.perf_counter()
        try:
            text = retry_call(lambda: self.provider.generate(prompt, prepared, schema), sleep=self._sleep)
            result = schema.model_validate_json(_strip_fences(text))
        except VisionError:
            self._record(time.perf_counter() - start, cached=False, ok=False)
            raise
        except Exception as error:
            self._record(time.perf_counter() - start, cached=False, ok=False)
            raise VisionError(f"{self.provider.name} call to {self.provider.model} failed: {error}") from error

        self._record(time.perf_counter() - start, cached=False, ok=True)
        self.cache.set(key, result.model_dump(mode="json"))
        return result

    def _record(self, seconds: float, cached: bool, ok: bool) -> None:
        record = CallRecord(self.provider.name, self.provider.model, round(seconds, 3), cached, ok)
        self.calls.append(record)
        if cached:
            return
        log.info("%s %s call took %.1fs (%s)", record.provider, record.model, seconds, "ok" if ok else "failed")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = {"time": datetime.now(timezone.utc).isoformat(), **asdict(record)}
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")


def make_provider(settings: Settings, model: str | None = None) -> Provider:
    """Build the provider named in settings. Imports are lazy so unused SDKs never load."""
    vision = settings.vision
    model = model or vision.model
    if vision.provider == "ollama":
        from driftcheck.vision.ollama_provider import OllamaProvider

        return OllamaProvider(model, settings.ollama_host, vision.timeout_seconds)
    if vision.provider == "gemini":
        from driftcheck.vision.gemini_provider import GeminiProvider

        return GeminiProvider(model, settings.gemini_api_key)
    if vision.provider == "hf":
        from driftcheck.vision.hf_provider import HuggingFaceProvider

        return HuggingFaceProvider(model, settings.hf_token, vision.timeout_seconds)
    raise VisionUnavailable(f"Unknown VISION_PROVIDER {vision.provider!r}. Use ollama, gemini or hf.")


def get_vision_client(settings: Settings | None = None, model: str | None = None) -> VisionClient:
    settings = settings or load_settings()
    return VisionClient(
        make_provider(settings, model),
        DiskCache(settings.cache_dir, "vision"),
        max_side=settings.vision.max_side,
        log_path=settings.cache_dir / "vision_calls.jsonl",
    )
