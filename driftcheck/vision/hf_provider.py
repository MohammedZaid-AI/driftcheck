"""Optional Hugging Face Inference Providers route for Qwen VL. Needs HF_TOKEN."""

from __future__ import annotations

import base64

from pydantic import BaseModel

from driftcheck.vision.base import VisionUnavailable, with_schema


class HuggingFaceProvider:
    name = "hf"
    version = "2"  # 2: schema is also written into the prompt

    def __init__(self, model: str, token: str | None, timeout: float = 600) -> None:
        if not token:
            raise VisionUnavailable("HF_TOKEN is not set in .env. It is needed when VISION_PROVIDER=hf.")
        from huggingface_hub import InferenceClient

        self.model = model
        self._client = InferenceClient(api_key=token, timeout=timeout)

    def check(self) -> None:
        # The token is checked in __init__. A model with no live provider fails on the first call.
        return None

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}}
            for image in images
        ]
        content.append({"type": "text", "text": with_schema(prompt, schema)})
        response = self._client.chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            },
            temperature=0,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
