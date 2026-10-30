"""Local Ollama provider. The default."""

from __future__ import annotations

from pydantic import BaseModel

from driftcheck.vision.base import VisionUnavailable, with_schema


def _with_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaProvider:
    name = "ollama"
    # 2: schema is also written into the prompt.
    # 3: num_ctx raised from Ollama's 4096, which cut off prompts with several images,
    #    and plain JSON fallback for the grammar error.
    # 4: answers capped at num_predict tokens. Once, qwen2.5vl:3b kept generating until
    #    the 16384 token context was full, which took over 3 minutes and was not valid JSON.
    version = "4"

    def __init__(self, model: str, host: str, timeout: float = 600, num_ctx: int = 16384, num_predict: int = 1024) -> None:
        import ollama

        self.model = model
        self.host = host
        self.num_ctx = num_ctx
        self.num_predict = num_predict
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
        messages = [{"role": "user", "content": with_schema(prompt, schema), "images": images}]
        options = {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": self.num_predict}
        try:
            response = self._client.chat(model=self.model, messages=messages, format=schema.model_json_schema(), options=options)
        except Exception as error:
            # llama.cpp sometimes fails schema constrained decoding with "empty grammar stack".
            # At temperature 0 a retry fails the same way, so fall back to plain JSON mode.
            # The schema is still in the prompt and the answer is still validated by the client.
            if "grammar" not in str(error):
                raise
            response = self._client.chat(model=self.model, messages=messages, format="json", options=options)
        return response.message.content or ""
