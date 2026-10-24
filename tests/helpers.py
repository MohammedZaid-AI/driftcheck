"""Small synthetic images and a fake vision client for tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
from pydantic import BaseModel

from driftcheck.vision import VisionError


def save_image(path: Path, colour: tuple[int, int, int] = (200, 40, 40), size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


class FakeVisionClient:
    """Answers ask() from a list of replies. A VisionError reply is raised."""

    def __init__(self, replies: list[dict[str, Any] | VisionError]) -> None:
        self.replies = replies
        self.provider = SimpleNamespace(name="fake")
        self.model = "fake-vl"
        self.asked: list[str] = []

    def ensure_ready(self) -> None:
        return None

    def ask(self, prompt: str, images: list, schema: type[BaseModel]) -> BaseModel:
        self.asked.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, VisionError):
            raise reply
        return schema.model_validate(reply)
