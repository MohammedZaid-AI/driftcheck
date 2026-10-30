from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from driftcheck.vision import VisionUnavailable
from driftcheck.vision.groq_provider import BASE_URL, GroqProvider

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class Garment(BaseModel):
    sleeve_length: str


def _provider(handler, model: str = MODEL) -> tuple[GroqProvider, list[float]]:
    waits: list[float] = []
    provider = GroqProvider(model, "test-key", sleep=waits.append)
    provider._http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return provider, waits


def _reply(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_needs_a_key() -> None:
    with pytest.raises(VisionUnavailable, match="GROQ_API_KEY"):
        GroqProvider(MODEL, None)


def test_sends_images_schema_and_limits() -> None:
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return _reply('{"sleeve_length": "full"}')

    provider, _ = _provider(handler)
    assert json.loads(provider.generate("describe", [b"jpeg"], Garment)) == {"sleeve_length": "full"}
    content = sent["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert '"sleeve_length"' in content[-1]["text"]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["temperature"] == 0 and sent["max_completion_tokens"] == 1024


def test_falls_back_to_json_mode_when_schema_is_not_supported() -> None:
    formats = []

    def handler(request: httpx.Request) -> httpx.Response:
        kind = json.loads(request.content)["response_format"]["type"]
        formats.append(kind)
        if kind == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})
        return _reply('{"sleeve_length": "short"}')

    provider, _ = _provider(handler)
    assert json.loads(provider.generate("x", [b"j"], Garment))["sleeve_length"] == "short"
    assert formats == ["json_schema", "json_object"]


def test_waits_out_rate_limits() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "7"})
        return _reply('{"sleeve_length": "full"}')

    provider, waits = _provider(handler)
    provider.generate("x", [b"j"], Garment)
    assert waits == [7.5, 7.5]
    assert provider.waited_seconds == 15.0


def test_daily_cap_stops_instead_of_waiting_hours() -> None:
    from driftcheck.vision import QuotaExhausted

    provider, waits = _provider(lambda request: httpx.Response(429, headers={"retry-after": "5400"}))
    with pytest.raises(QuotaExhausted, match="5400"):
        provider.generate("x", [b"j"], Garment)
    assert waits == []


def test_server_errors_are_raised_for_the_client_to_retry() -> None:
    provider, _ = _provider(lambda request: httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        provider.generate("x", [b"j"], Garment)
    assert caught.value.response.status_code == 503


def test_check_reports_unknown_model_and_bad_key() -> None:
    models = httpx.Response(200, json={"data": [{"id": MODEL}, {"id": "llama-3.3-70b-versatile"}]})
    _provider(lambda request: models)[0].check()
    with pytest.raises(VisionUnavailable, match="no model other/model"):
        _provider(lambda request: models, model="other/model")[0].check()
    with pytest.raises(VisionUnavailable, match="rejected"):
        _provider(lambda request: httpx.Response(401))[0].check()
