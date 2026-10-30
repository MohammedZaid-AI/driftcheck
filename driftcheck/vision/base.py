"""Shared types for vision providers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Protocol, Union

from PIL import Image, ImageOps
from pydantic import BaseModel

ImageInput = Union[bytes, Path, str, Image.Image]


class VisionError(RuntimeError):
    """A vision call failed for good. Checks turn this into unknown."""


class VisionUnavailable(VisionError):
    """The provider is not running or not set up."""


class QuotaExhausted(Exception):
    """The provider's rate limit needs a long wait, such as a daily cap.
    Not a VisionError on purpose: it stops a run instead of turning checks into unknown."""


class Provider(Protocol):
    name: str
    model: str

    def check(self) -> None:
        """Raise VisionUnavailable with a clear fix if the provider cannot be used."""

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        """Send JPEG images and a prompt. Return the raw JSON text."""


def with_schema(prompt: str, schema: type[BaseModel]) -> str:
    """Constrained decoding forces the JSON shape, but the model never sees the
    field descriptions or allowed values unless they are in the prompt."""
    return f"{prompt}\n\nAnswer with JSON that follows this schema. Use only the allowed values.\n{json.dumps(schema.model_json_schema())}"


def prepare_image(image: ImageInput, max_side: int = 1024) -> bytes:
    """Turn any image input into RGB JPEG bytes with the longest side at most max_side."""
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.open(Path(image))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()
