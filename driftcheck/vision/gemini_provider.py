"""Optional Gemini provider. Needs GEMINI_API_KEY."""

from __future__ import annotations

from pydantic import BaseModel

from driftcheck.vision.base import VisionUnavailable


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise VisionUnavailable("GEMINI_API_KEY is not set in .env. It is needed when VISION_PROVIDER=gemini.")
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def check(self) -> None:
        # The key is checked in __init__. A bad key fails on the first call.
        return None

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        from google.genai import types

        parts = [types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images]
        response = self._client.models.generate_content(
            model=self.model,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            ),
        )
        return response.text or ""
