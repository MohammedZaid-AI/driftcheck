"""Product card: what the original product looks like, read once and cached.

The vision provider describes shape, pattern, logos and text. easyocr gives the
text items. The pixel palette gives the colours.
"""

from __future__ import annotations

import io
from typing import Callable, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from driftcheck.cache import DiskCache, make_key
from driftcheck.dataset import Category
from driftcheck.log import get_logger
from driftcheck.ocr import TextBox
from driftcheck.palette import PaletteColour, palette
from driftcheck.segment import Box, crop_to_product, pad_box
from driftcheck.vision import VisionClient, VisionError

log = get_logger(__name__)

CARD_VERSION = "1"  # bump when the prompt or schema changes, so old cached cards are not reused
MAX_TEXT_CROPS = 3

SleeveLength = Literal["sleeveless", "short", "elbow", "three_quarter", "full", "not_applicable"]
PatternType = Literal["solid", "stripes", "checks", "floral", "geometric", "print", "embroidery", "mixed", "other"]


class LogoItem(BaseModel):
    description: str = Field(description="what the logo looks like or says")
    position: str = Field(description="where on the product, for example left chest or top of the pack")


class TextItem(BaseModel):
    text: str = Field(description="the exact text as printed")
    language: str = Field(description="English, Hindi, Kannada or other")
    position: str = Field(description="where on the product")


class VisionCard(BaseModel):
    """What the vision model returns about the original product."""

    product_type: str = Field(description="two to four words, for example sleeveless cotton kurti")
    category: Category
    visible_views: list[str] = Field(description="views shown, for example front, back, side, close up")
    sleeve_length: SleeveLength = Field(description="not_applicable if the product is not a garment with arms")
    neckline: str = Field(description="for example round, v neck, collar, boat; not_applicable if none")
    silhouette: str = Field(description="overall cut or shape, for example straight, a line, fitted, box shaped")
    length: str = Field(description="garment length, for example cropped, hip, knee, ankle; not_applicable if none")
    closures: list[str] = Field(description="buttons, zip, hooks, drawstring, lid, cap; empty if none")
    main_colours: list[str] = Field(description="main colours in plain words, largest first")
    pattern_type: PatternType
    pattern_description: str = Field(description="describe the print, embroidery or texture; empty if solid")
    embellishments: list[str] = Field(description="sequins, lace, mirror work, piping; empty if none")
    logos: list[LogoItem] = Field(description="every logo or brand mark; empty if none")
    text_items: list[TextItem] = Field(description="every piece of visible text; empty if none")
    key_features: list[str] = Field(description="other details a shopper would notice")


class OcrItem(BaseModel):
    text: str
    confidence: float
    box: Box  # in the original image
    reader: str


class ProductCard(BaseModel):
    version: str = CARD_VERSION
    vision: VisionCard | None
    vision_model: str
    vision_error: str | None = None
    palette: list[PaletteColour]
    ocr: list[OcrItem]
    product_box: Box | None
    product_share: float  # share of the image covered by the product


CARD_PROMPT = """You are checking product photos for an Indian online store.
The first image is the product cut out from its photo. Any later images are zoomed crops of areas with text, in the same product.
Describe only the product, not the background, hanger, mannequin or person.
Copy every visible text exactly as printed, letter by letter, in its own script. Do not translate.
If something is not there, use an empty list or not_applicable. Do not guess."""


def _jpeg(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.convert("RGB").save(out, format="JPEG", quality=95)
    return out.getvalue()


def text_crops(image: Image.Image, boxes: list[TextBox], limit: int = MAX_TEXT_CROPS) -> list[Image.Image]:
    """Zoomed crops of the largest text areas, so a small model can read them."""
    largest = sorted(boxes, key=lambda b: -(b.box[2] - b.box[0]) * (b.box[3] - b.box[1]))[:limit]
    crops = []
    for box in largest:
        crop = image.crop(pad_box(box.box, image.size, pad=0.5))
        scale = 768 / max(crop.size)
        crops.append(crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS))
    return crops


def build_card(
    image: Image.Image,
    client: VisionClient,
    mask_fn: Callable[[Image.Image], np.ndarray],
    ocr_fn: Callable[[Image.Image], list[TextBox]],
    cache: DiskCache | None = None,
) -> ProductCard:
    """Read the original once. A failed vision call leaves vision empty with the reason."""
    image = image.convert("RGB")
    model_name = f"{client.provider.name}/{client.model}"
    key = make_key("card", CARD_VERSION, model_name, _jpeg(image))
    if cache and (stored := cache.get(key)) is not None:
        return ProductCard.model_validate(stored)

    mask = mask_fn(image)
    cropped = crop_to_product(image, mask)
    boxes = ocr_fn(image)
    vision: VisionCard | None = None
    error: str | None = None
    if cropped is None:
        error = "No product found in the original."
    else:
        product_image = cropped[0]
        try:
            vision = client.ask(CARD_PROMPT, [product_image, *text_crops(image, boxes)], VisionCard)
        except VisionError as failure:
            error = str(failure)
            log.warning("Product card without vision: %s", error)

    card = ProductCard(
        vision=vision,
        vision_model=model_name,
        vision_error=error,
        palette=palette(image, mask),
        ocr=[OcrItem(text=b.text, confidence=round(b.confidence, 3), box=b.box, reader=b.reader) for b in boxes],
        product_box=cropped[2] if cropped else None,
        product_share=round(float(mask.mean()), 4),
    )
    # Do not cache a card whose vision call failed, so the next run retries it.
    if cache and vision is not None:
        cache.set(key, card.model_dump(mode="json"))
    return card
