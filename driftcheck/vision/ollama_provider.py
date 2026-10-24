"""Local Ollama provider. The default."""

from __future__ import annotations

from pydantic import BaseModel

from driftcheck.vision.base import VisionUnavailable, with_schema


def _with_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaProvider:
    name = "ollama"
    version = "2"  # 2: schema is also written into the prompt

    def __init__(self, model: str, host: str, timeout: float = 600) -> None:
        import ollama

        self.model = model
        self.host = host
        self._client = ollama.Client(host=host, timeout=timeout)

    def check(self) -> None:
        try:
            listed = self._client.list()
        except Exception as error:
            raise VisionUnavailable(
                f"Ollama is not running at {self.host}. Open the Ollama app or run: ollama serve"
            ) from error
        pulled = {_with_tag(m.model) for m in listed.models if m.model}
        if _with_tag(self.model) not in pulled:
            raise VisionUnavailable(f"Ollama model {self.model} is not pulled. Run: ollama pull {self.model}")

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": with_schema(prompt, schema), "images": images}],
            format=schema.model_json_schema(),
            options={"temperature": 0},
        )
        return response.message.content or ""
