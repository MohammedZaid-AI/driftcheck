"""Builds a Pair from synthetic images for check tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from driftcheck.card import OcrItem, ProductCard, VisionCard
from driftcheck.checks.base import Pair, Thresholds, load_thresholds
from driftcheck.config import PROJECT_ROOT
from driftcheck.ocr import TextBox
from driftcheck.palette import palette
from tests.synthetic import simple_mask


def thresholds() -> Thresholds:
    return load_thresholds(PROJECT_ROOT / "config")


def make_card(image: Image.Image, vision: dict | None = None, ocr: list[TextBox] | None = None) -> ProductCard:
    mask = simple_mask(image)
    return ProductCard(
        vision=VisionCard.model_validate(vision) if vision else None,
        vision_model="fake/fake-vl",
        palette=palette(image, mask),
        ocr=[OcrItem(text=b.text, confidence=b.confidence, box=b.box, reader=b.reader) for b in (ocr or [])],
        product_box=None,
        product_share=float(mask.mean()),
    )


def make_pair(original: Image.Image, output: Image.Image, vision: dict | None = None,
              ocr: list[TextBox] | None = None, output_mask: np.ndarray | None = None) -> Pair:
    return Pair(
        original=original,
        output=output,
        original_mask=simple_mask(original),
        output_mask=simple_mask(output) if output_mask is None else output_mask,
        card=make_card(original, vision, ocr),
    )
