"""Optional Groq provider, through its OpenAI compatible API. Needs GROQ_API_KEY."""

from __future__ import annotations

import base64
import time

from pydantic import BaseModel

from driftcheck.vision.base import QuotaExhausted, VisionUnavailable, with_schema

BASE_URL = "https://api.groq.com/openai/v1"
MAX_RATE_LIMIT_WAITS = 5
MAX_WAIT_SECONDS = 60.0


class GroqProvider:
    name = "groq"
    version = "1"

    def __init__(self, model: str, api_key: str | None, timeout: float = 120, max_tokens: int = 1024,
                 sleep=time.sleep) -> None:
        if not api_key:
            raise VisionUnavailable("GROQ_API_KEY is not set in .env. It is needed when VISION_PROVIDER=groq.")
        import httpx

        self.model = model
        self.max_tokens = max_tokens
        self.waited_seconds = 0.0  # time spent waiting on rate limits, kept out of speed figures
        self._sleep = sleep
        self._http = httpx.Client(base_url=BASE_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)

    def check(self) -> None:
        try:
            response = self._http.get("/models")
        except Exception as error:
            raise VisionUnavailable(f"Cannot reach Groq: {error}") from error
        if response.status_code == 401:
            raise VisionUnavailable("Groq rejected GROQ_API_KEY. Check the key in .env.")
        response.raise_for_status()
        models = {m["id"] for m in response.json().get("data", [])}
        if self.model not in models:
            raise VisionUnavailable(f"Groq has no model {self.model}. Set VISION_MODEL to one of: {', '.join(sorted(models))}")

    def _post(self, body: dict):
        """POST, waiting out short rate limits for as long as Groq asks. A wait longer than
        MAX_WAIT_SECONDS means a daily cap, so the run stops instead of waiting for hours."""
        for _ in range(MAX_RATE_LIMIT_WAITS):
            response = self._http.post("/chat/completions", json=body)
            if response.status_code != 429:
                return response
            wait = float(response.headers.get("retry-after", "10") or 10)
            if wait > MAX_WAIT_SECONDS:
                raise QuotaExhausted(
                    f"Groq asks to wait {wait:.0f}s. Either the daily cap is reached or one request is bigger than "
                    f"the per minute token budget. Error from Groq: {response.text[:300]}")
            self._sleep(wait + 0.5)
            self.waited_seconds += wait + 0.5
        return response

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}
            for image in images
        ]
        content.append({"type": "text", "text": with_schema(prompt, schema)})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                 "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
        }
        response = self._post(body)
        if response.status_code == 400 and "response_format" in response.text:
            # Not every Groq model takes a JSON schema. Plain JSON mode still works, and the
            # schema is in the prompt and checked by the client.
            body["response_format"] = {"type": "json_object"}
            response = self._post(body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"].get("content") or ""
