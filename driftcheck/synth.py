"""Synthetic drift: controlled changes to the originals with automatic labels.

This set is reported on its own and never mixed with real outputs. logo_changed
is always null because synth does not know where logos are.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from driftcheck.dataset import list_product_ids, product_dir, read_meta
from driftcheck.labels import ATTRIBUTES, write_json_atomic
from driftcheck.log import get_logger
from driftcheck.ocr import TextBox
from driftcheck.segment import mask_box

log = get_logger(__name__)

# Std of the high pass signal inside the product. Measured on 2026-10-24:
# the plain shirt sample scores 4.8 and a striped test pattern 12.6.
TEXTURE_MIN = 8.0
SLEEVE_MIN_SHARE = 0.03  # sleeve region must be at least this share of the product

DRIFTS = (
    "colour_shift",
    "text_erase",
    "text_replace",
    "pattern_blur",
    "pattern_swap",
    "sleeve_erase",
    "product_erase",
    "control_light",
    "control_jpeg",
    "control_shift",
)


@dataclass
class SynthItem:
    drift: str
    image: Image.Image
    labels: dict[str, bool | None]
    params: dict[str, Any] = field(default_factory=dict)


def texture_score(image: Image.Image, mask: np.ndarray) -> float:
    gray = image.convert("L")
    high = np.asarray(gray, float) - np.asarray(gray.filter(ImageFilter.GaussianBlur(2)), float)
    inner = binary_erosion(mask, iterations=6)
    return float(high[inner].std()) if inner.any() else 0.0


def _labels(has_text: bool, **changed: bool | None) -> dict[str, bool | None]:
    labels: dict[str, bool | None] = {name: False for name in ATTRIBUTES}
    labels["logo_changed"] = None
    labels["text_changed"] = False if has_text else None
    labels.update(changed)
    return labels


def _background(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    outside = arr[~mask]
    return np.median(outside, axis=0) if len(outside) else np.array([255, 255, 255])


def _paste_where(base: Image.Image, top: Image.Image, mask: np.ndarray) -> Image.Image:
    return Image.composite(top, base, Image.fromarray((mask * 255).astype(np.uint8)))


def text_in_product(boxes: list[TextBox], mask: np.ndarray) -> list[TextBox]:
    inside = []
    for box in boxes:
        cx, cy = (box.box[0] + box.box[2]) // 2, (box.box[1] + box.box[3]) // 2
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
            inside.append(box)
    return inside


def colour_shift(image: Image.Image, mask: np.ndarray, rng: random.Random) -> SynthItem:
    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(np.asarray(image) / 255.0)
    angle, size = rng.uniform(0, 2 * np.pi), rng.uniform(28, 40)
    da, db = size * np.cos(angle), size * np.sin(angle)
    dl = 15.0 if lab[mask, 0].mean() < 25 else 0.0  # very dark products need light to show a hue
    lab[mask, 0] += dl
    lab[mask, 1] += da
    lab[mask, 2] += db
    rgb = (np.clip(lab2rgb(lab), 0, 1) * 255).round().astype(np.uint8)
    params = {"da": round(da, 1), "db": round(db, 1), "dL": dl}
    return SynthItem("colour_shift", Image.fromarray(rgb), {}, params)


def _fill_box(arr: np.ndarray, box: tuple[int, int, int, int], pad: int = 2) -> np.ndarray:
    """Cover a box with the median colour of a thin ring around it."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(w, box[2] + pad), min(h, box[3] + pad)
    rx0, ry0, rx1, ry1 = max(0, x0 - 4), max(0, y0 - 4), min(w, x1 + 4), min(h, y1 + 4)
    ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
    ring[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0] = False
    colour = np.median(arr[ry0:ry1, rx0:rx1][ring], axis=0) if ring.any() else np.array([255, 255, 255])
    arr[y0:y1, x0:x1] = colour
    return colour


def text_erase(image: Image.Image, boxes: list[TextBox]) -> SynthItem:
    arr = np.asarray(image).copy()
    for box in boxes:
        _fill_box(arr, box.box)
    return SynthItem("text_erase", Image.fromarray(arr), {}, {"erased": [b.text for b in boxes]})


def mutate_text(text: str, rng: random.Random) -> str:
    """Change about a third of the latin letters and digits, keeping case and length."""
    chars = list(text)
    spots = [i for i, c in enumerate(chars) if c.isascii() and c.isalnum()]
    for i in rng.sample(spots, max(1, len(spots) // 3)):
        c = chars[i]
        pool = "0123456789" if c.isdigit() else "ABCDEFGHJKLMNPRSTUVWXYZ"
        pool = pool if c.isdigit() or c.isupper() else pool.lower()
        chars[i] = rng.choice([p for p in pool if p != c])
    return "".join(chars)


def text_replace(image: Image.Image, boxes: list[TextBox], rng: random.Random) -> SynthItem | None:
    """Rewrite latin text with a few letters changed. None if no box has latin letters or digits."""
    latin = [b for b in boxes if any(c.isascii() and c.isalnum() for c in b.text)]
    if not latin:
        return None
    arr = np.asarray(image).copy()
    original = arr.copy()
    swaps = []
    fills = []
    for box in latin:
        fills.append(_fill_box(arr, box.box))
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    for box, fill in zip(latin, fills):
        x0, y0, x1, y1 = box.box
        region = original[y0:y1, x0:x1].reshape(-1, 3).astype(float)
        far = np.linalg.norm(region - fill, axis=1)
        ink = tuple(int(v) for v in region[far >= np.percentile(far, 90)].mean(axis=0)) if len(region) else (0, 0, 0)
        new = mutate_text(box.text, rng)
        font = ImageFont.load_default(size=max(8, int((y1 - y0) * 0.8)))
        draw.text((x0, y0), new, fill=ink, font=font)
        swaps.append([box.text, new])
    return SynthItem("text_replace", out, {}, {"swaps": swaps})


def pattern_blur(image: Image.Image, mask: np.ndarray) -> SynthItem:
    box = mask_box(mask)
    assert box is not None
    radius = max(4.0, 0.03 * min(box[2] - box[0], box[3] - box[1]))
    blurred = image.filter(ImageFilter.GaussianBlur(radius))
    return SynthItem("pattern_blur", _paste_where(image, blurred, mask), {}, {"radius": round(radius, 1)})


def pattern_swap(image: Image.Image, mask: np.ndarray, rng: random.Random, grid: int = 4) -> SynthItem:
    """Shuffle tiles of the product area, so the pattern no longer matches."""
    x0, y0, x1, y1 = mask_box(mask)  # type: ignore[misc]
    tw, th = (x1 - x0) // grid, (y1 - y0) // grid
    tiles = [(x0 + c * tw, y0 + r * th) for r in range(grid) for c in range(grid)]
    order = tiles[:]
    while order == tiles:
        rng.shuffle(order)
    shuffled = image.copy()
    for (tx, ty), (sx, sy) in zip(tiles, order):
        shuffled.paste(image.crop((sx, sy, sx + tw, sy + th)), (tx, ty))
    return SynthItem("pattern_swap", _paste_where(image, shuffled, mask), {}, {"grid": grid})


def sleeve_erase(image: Image.Image, mask: np.ndarray) -> SynthItem | None:
    """Paint over the outer upper bands of a garment, where sleeves sit."""
    x0, y0, x1, y1 = mask_box(mask)  # type: ignore[misc]
    band = int(0.22 * (x1 - x0))
    region = np.zeros_like(mask)
    bottom = y0 + int(0.55 * (y1 - y0))
    region[y0:bottom, x0 : x0 + band] = True
    region[y0:bottom, x1 - band : x1] = True
    region &= mask
    share = region.sum() / mask.sum()
    if share < SLEEVE_MIN_SHARE:
        return None
    arr = np.asarray(image).copy()
    arr[region] = _background(arr, mask)
    return SynthItem("sleeve_erase", Image.fromarray(arr), {}, {"share": round(float(share), 3)})


def product_erase(image: Image.Image, mask: np.ndarray) -> SynthItem:
    arr = np.asarray(image).copy()
    arr[binary_dilation(mask, iterations=2)] = _background(arr, mask)
    return SynthItem("product_erase", Image.fromarray(arr), {})


def control_light(image: Image.Image) -> SynthItem:
    gains = np.array([1.06 * 1.04, 1.06, 1.06 * 0.95])  # a bit brighter and warmer
    arr = np.clip(np.asarray(image) * gains, 0, 255).astype(np.uint8)
    return SynthItem("control_light", Image.fromarray(arr), {}, {"gains": gains.round(3).tolist()})


def control_jpeg(image: Image.Image) -> SynthItem:
    small = image.resize((int(image.width * 0.8), int(image.height * 0.8)), Image.Resampling.BILINEAR)
    out = io.BytesIO()
    small.resize(image.size, Image.Resampling.BILINEAR).save(out, format="JPEG", quality=60)
    return SynthItem("control_jpeg", Image.open(io.BytesIO(out.getvalue())).convert("RGB"), {}, {"quality": 60})


def control_shift(image: Image.Image, rng: random.Random) -> SynthItem:
    """Zoom in a little and shift, like a new framing of the same product."""
    w, h = image.size
    zoom = rng.uniform(0.9, 0.95)
    cw, ch = int(w * zoom), int(h * zoom)
    left, top = rng.randint(0, w - cw), rng.randint(0, h - ch)
    out = image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.Resampling.LANCZOS)
    return SynthItem("control_shift", out, {}, {"zoom": round(zoom, 3), "left": left, "top": top})


def make_drifts(
    image: Image.Image, mask: np.ndarray, boxes: list[TextBox], category: str | None, rng: random.Random
) -> list[SynthItem]:
    """Every drift that applies to this product, each with its labels."""
    image = image.convert("RGB")
    text = text_in_product(boxes, mask)
    has_text = bool(text)
    patterned = texture_score(image, mask) >= TEXTURE_MIN
    items: list[SynthItem] = []

    def add(item: SynthItem | None, **changed: bool | None) -> None:
        if item is not None:
            item.labels = _labels(has_text, **changed)
            items.append(item)

    add(colour_shift(image, mask, rng), colour_changed=True)
    if has_text:
        add(text_erase(image, text), text_changed=True)
        add(text_replace(image, text, rng), text_changed=True)
    if patterned:
        add(pattern_blur(image, mask), pattern_changed=True, text_changed=True if has_text else None)
        add(pattern_swap(image, mask, rng), pattern_changed=True, text_changed=True if has_text else None)
    if category == "clothing":
        add(sleeve_erase(image, mask), shape_changed=True)
    add(product_erase(image, mask), product_missing=True, shape_changed=None, colour_changed=None,
        pattern_changed=None, text_changed=None)
    add(control_light(image))
    add(control_jpeg(image))
    add(control_shift(image, rng))
    return items


def run_synth(
    data_dir: Path,
    mask_fn: Callable[[Image.Image], np.ndarray],
    ocr_fn: Callable[[Image.Image], list[TextBox]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Write data/synthetic/<product_id>/<drift>.jpg and .json, plus data/synthetic/labels.json."""
    out_root = data_dir / "synthetic"
    rows: list[dict[str, Any]] = []
    for product_id in list_product_ids(data_dir):
        meta = read_meta(data_dir, product_id)
        image = Image.open(product_dir(data_dir, product_id) / "original.jpg").convert("RGB")
        mask = mask_fn(image)
        if mask.sum() < 0.01 * mask.size:
            log.warning("Skipping %s. No product found in the original.", product_id)
            continue
        rng = random.Random(f"{seed}-{product_id}")
        for item in make_drifts(image, mask, ocr_fn(image), meta.category, rng):
            folder = out_root / product_id
            folder.mkdir(parents=True, exist_ok=True)
            item.image.save(folder / f"{item.drift}.jpg", quality=95)
            row = {"product_id": product_id, "output_id": item.drift, **item.labels, "notes": "synthetic"}
            write_json_atomic(folder / f"{item.drift}.json", {**row, "drift": item.drift, "params": item.params, "seed": seed})
            rows.append(row)
        log.info("%s: %d synthetic outputs", product_id, sum(r["product_id"] == product_id for r in rows))
    write_json_atomic(out_root / "labels.json", rows)
    return rows
