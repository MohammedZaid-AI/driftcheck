"""Shared types for vision providers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Protocol, Union

from PIL import Image, ImageOps
from pydantic import BaseModel

ImageInput = Union[bytes, Path, str, Image.Image]


class VisionError(RuntimeError):
    """A vision call failed for good. Checks turn this into unknown."""


class VisionUnavailable(VisionError):
    """The provider is not running or not set up."""


class Provider(Protocol):
    name: str
    model: str

    def check(self) -> None:
        """Raise VisionUnavailable with a clear fix if the provider cannot be used."""

    def generate(self, prompt: str, images: list[bytes], schema: type[BaseModel]) -> str:
        """Send JPEG images and a prompt. Return the raw JSON text."""


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
