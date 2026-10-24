"""Small synthetic product images for tests. White background, product in the middle."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from driftcheck.ocr import TextBox

SIZE = (160, 200)
BODY = (40, 50, 120, 180)  # x0, y0, x1, y1 of the garment body
TEXT_BOX = (55, 100, 105, 116)


def garment(colour: tuple[int, int, int] = (190, 30, 40), stripes: bool = False, text: str | None = None,
            sleeves: bool = True, long_sleeves: bool = False) -> Image.Image:
    """A flat top: body, optional sleeves at the sides, optional stripes and text."""
    img = Image.new("RGB", SIZE, "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle(BODY, fill=colour)
    if sleeves:
        bottom = 175 if long_sleeves else 110
        draw.rectangle((15, 50, 40, bottom), fill=colour)
        draw.rectangle((120, 50, 145, bottom), fill=colour)
    if stripes:
        for x in range(BODY[0], BODY[2], 8):
            draw.rectangle((x, BODY[1], x + 3, BODY[3]), fill=(30, 40, 170))
    if text:
        draw.rectangle(TEXT_BOX, fill="white")
        draw.text((TEXT_BOX[0] + 2, TEXT_BOX[1] + 1), text, fill="black", font=ImageFont.load_default(size=12))
    return img


def simple_mask(image: Image.Image) -> np.ndarray:
    """Stands in for rembg: product is anything not close to white, plus white areas inside the body."""
    mask = np.asarray(image.convert("RGB")).astype(int).sum(axis=2) < 700
    x0, y0, x1, y1 = BODY
    if mask[y0:y1, x0:x1].mean() > 0.5:
        mask[y0:y1, x0:x1] = True
    return mask


def text_boxes(text: str = "SALE 50") -> list[TextBox]:
    return [TextBox(text, 0.95, TEXT_BOX, "en_hi")]
