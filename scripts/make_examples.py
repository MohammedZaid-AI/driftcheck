"""Make the public example pairs in examples/ from the user's own sample photo.

Each edited image has one change made on purpose, to show how DriftCheck responds.
They are not AI outputs and not Scalio outputs. Run from the project folder:

    .venv\\Scripts\\python scripts\\make_examples.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from driftcheck.config import use_system_certificates  # noqa: E402

SOURCE = ROOT / "data" / "samples" / "plain_shirt.jpg"
OUT = ROOT / "examples"


def shirt_mask(image: Image.Image) -> np.ndarray:
    """The shirt itself: the rembg product mask, without the light brown hanger."""
    from driftcheck.segment import product_mask

    gray = np.asarray(image.convert("L"))
    return product_mask(image) & (gray < 110)


def recolour(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Black shirt to blue: lighter and bluer in Lab colour space, inside the shirt only."""
    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(np.asarray(image) / 255.0)
    lab[mask, 0] += 22
    lab[mask, 1] += 6
    lab[mask, 2] -= 38
    return Image.fromarray((np.clip(lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8))


def add_word(image: Image.Image, word: str = "SALE") -> Image.Image:
    """A white word printed across the chest."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=44)
    width = draw.textlength(word, font=font)
    draw.text(((out.width - width) / 2, 210), word, fill=(245, 245, 245), font=font)
    return out


def add_stripes(image: Image.Image, mask: np.ndarray, gap: int = 12, width: int = 4) -> Image.Image:
    """Light grey pinstripes over the shirt, where the original is plain."""
    arr = np.asarray(image).copy()
    columns = (np.arange(arr.shape[1]) % gap) < width
    stripes = mask & columns[None, :]
    arr[stripes] = (arr[stripes] * 0.3 + np.array([200, 200, 200]) * 0.7).astype(np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    use_system_certificates()
    OUT.mkdir(exist_ok=True)
    original = Image.open(SOURCE).convert("RGB")
    shutil.copyfile(SOURCE, OUT / "plain_shirt_original.jpg")
    mask = shirt_mask(original)
    edits = {
        "plain_shirt_unchanged.jpg": original,
        "plain_shirt_colour.jpg": recolour(original, mask),
        "plain_shirt_text.jpg": add_word(original),
        "plain_shirt_stripes.jpg": add_stripes(original, mask),
    }
    for name, image in edits.items():
        image.save(OUT / name, quality=95)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
