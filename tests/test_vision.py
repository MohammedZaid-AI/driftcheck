from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import BaseModel

from driftcheck.cache import DiskCache
from driftcheck.vision import VisionClient, VisionError, VisionUnavailable, prepare_image
from driftcheck.vision.ollama_provider import OllamaProvider


class Garment(BaseModel):
    sleeve_length: str
    main_colour: str


def _png(width: int, height: int, colour: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(out, format="PNG")
    return out.getvalue()


class FakeProvider:
    name = "fake"
    model = "fake-vl"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.checks = 0
        self.seen_images: list[list[bytes]] = []

    def check(self) -> None:
        self.checks += 1

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        self.seen_images.append(images)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _client(tmp_path: Path, provider: FakeProvider) -> VisionClient:
    return VisionClient(provider, DiskCache(tmp_path, "vision"), max_side=1024, log_path=tmp_path / "calls.jsonl", sleep=lambda _: None)


GOOD = json.dumps({"sleeve_length": "sleeveless", "main_colour": "red"})


def test_prepare_image_resizes_to_max_side() -> None:
    out = Image.open(io.BytesIO(prepare_image(_png(3000, 1500))))
    assert out.format == "JPEG"
    assert max(out.size) == 1024
    assert out.size == (1024, 512)


def test_prepare_image_keeps_small_images() -> None:
    out = Image.open(io.BytesIO(prepare_image(_png(400, 300))))
    assert out.size == (400, 300)


def test_ask_parses_schema_and_logs_call(tmp_path: Path) -> None:
    provider = FakeProvider([GOOD])
    client = _client(tmp_path, provider)
    result = client.ask("describe", [_png(2000, 2000)], Garment)
    assert result == Garment(sleeve_length="sleeveless", main_colour="red")
    assert provider.checks == 1
    assert max(Image.open(io.BytesIO(provider.seen_images[0][0])).size) == 1024
    assert len(client.calls) == 1 and client.calls[0].cached is False
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1


def test_second_ask_comes_from_cache(tmp_path: Path) -> None:
    provider = FakeProvider([GOOD])
    image = _png(64, 64)
    _client(tmp_path, provider).ask("describe", [image], Garment)

    fresh = _client(tmp_path, provider)
    assert fresh.ask("describe", [image], Garment).main_colour == "red"
    assert provider.replies == []  # no second real call
    assert fresh.calls[0].cached is True
    assert provider.checks == 1  # cache hits skip the health check


def test_cache_key_changes_with_prompt_and_image(tmp_path: Path) -> None:
    provider = FakeProvider([GOOD, GOOD, GOOD])
    client = _client(tmp_path, provider)
    client.ask("describe", [_png(64, 64)], Garment)
    client.ask("describe again", [_png(64, 64)], Garment)
    client.ask("describe", [_png(64, 64, (0, 0, 255))], Garment)
    assert provider.replies == []


def test_strips_code_fences(tmp_path: Path) -> None:
    provider = FakeProvider([f"```json\n{GOOD}\n```"])
    assert _client(tmp_path, provider).ask("x", [_png(32, 32)], Garment).sleeve_length == "sleeveless"


def test_retries_timeout_then_succeeds(tmp_path: Path) -> None:
    provider = FakeProvider([TimeoutError(), GOOD])
    assert _client(tmp_path, provider).ask("x", [_png(32, 32)], Garment).main_colour == "red"


def test_bad_json_becomes_vision_error(tmp_path: Path) -> None:
    provider = FakeProvider(["the top is red"])
    client = _client(tmp_path, provider)
    with pytest.raises(VisionError):
        client.ask("x", [_png(32, 32)], Garment)
    assert client.calls[0].ok is False


def test_unavailable_provider_raises_before_calling(tmp_path: Path) -> None:
    provider = FakeProvider([GOOD])

    def down() -> None:
        raise VisionUnavailable("Ollama is not running")

    provider.check = down  # type: ignore[method-assign]
    with pytest.raises(VisionUnavailable):
        _client(tmp_path, provider).ask("x", [_png(32, 32)], Garment)
    assert provider.replies == [GOOD]


class FakeOllama:
    def __init__(self, models: list[str] | None = None, down: bool = False) -> None:
        self.models = models or []
        self.down = down
        self.chat_kwargs: dict = {}

    def list(self) -> SimpleNamespace:
        if self.down:
            raise ConnectionError("Failed to connect to Ollama")
        return SimpleNamespace(models=[SimpleNamespace(model=m) for m in self.models])

    def chat(self, **kwargs) -> SimpleNamespace:
        self.chat_kwargs = kwargs
        return SimpleNamespace(message=SimpleNamespace(content=GOOD))


def _ollama(fake: FakeOllama, model: str = "qwen2.5vl:3b") -> OllamaProvider:
    provider = OllamaProvider(model, "http://localhost:11434")
    provider._client = fake  # type: ignore[assignment]
    return provider


def test_ollama_not_running_gives_clear_error() -> None:
    with pytest.raises(VisionUnavailable, match="not running.*ollama serve"):
        _ollama(FakeOllama(down=True)).check()


def test_ollama_model_not_pulled_says_how_to_pull() -> None:
    with pytest.raises(VisionUnavailable, match="ollama pull qwen2.5vl:7b"):
        _ollama(FakeOllama(["qwen2.5vl:3b"]), "qwen2.5vl:7b").check()


def test_ollama_untagged_model_matches_latest() -> None:
    _ollama(FakeOllama(["gemma3:latest"]), "gemma3").check()


def test_ollama_sends_schema_and_images() -> None:
    fake = FakeOllama(["qwen2.5vl:3b"])
    text = _ollama(fake).generate("describe", [b"jpeg-bytes"], Garment)
    assert json.loads(text)["main_colour"] == "red"
    assert fake.chat_kwargs["format"] == Garment.model_json_schema()
    assert fake.chat_kwargs["messages"][0]["images"] == [b"jpeg-bytes"]
    assert fake.chat_kwargs["options"]["temperature"] == 0


def test_optional_providers_need_their_keys() -> None:
    from driftcheck.vision.gemini_provider import GeminiProvider
    from driftcheck.vision.hf_provider import HuggingFaceProvider

    with pytest.raises(VisionUnavailable, match="GEMINI_API_KEY"):
        GeminiProvider("gemini-3.8-flash", None)
    with pytest.raises(VisionUnavailable, match="HF_TOKEN"):
        HuggingFaceProvider("Qwen/Qwen2.5-VL-3B-Instruct", None)
