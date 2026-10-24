"""Product masks with rembg, cached on disk."""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

from driftcheck.cache import DiskCache, make_key

REMBG_MODEL = "isnet-general-use"
Box = tuple[int, int, int, int]  # x0, y0, x1, y1 with x1 and y1 exclusive


@lru_cache(maxsize=1)
def _session():
    from rembg import new_session

    return new_session(REMBG_MODEL)


def _rembg_mask(image: Image.Image) -> Image.Image:
    from rembg import remove

    return remove(image, session=_session(), only_mask=True)


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def product_mask(image: Image.Image, cache: DiskCache | None = None) -> np.ndarray:
    """Boolean mask, True where the product is."""
    image = image.convert("RGB")
    key = make_key("rembg", REMBG_MODEL, _png(image))
    stored = cache.get_bytes(key) if cache else None
    if stored is not None:
        mask_image = Image.open(io.BytesIO(stored))
    else:
        mask_image = _rembg_mask(image).convert("L")
        if cache:
            cache.set_bytes(key, _png(mask_image))
    return np.asarray(mask_image) > 127


def mask_box(mask: np.ndarray) -> Box | None:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def pad_box(box: Box, size: tuple[int, int], pad: float = 0.04) -> Box:
    """Grow a box by a share of its size, clipped to the image size (width, height)."""
    x0, y0, x1, y1 = box
    dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
    return max(0, x0 - dx), max(0, y0 - dy), min(size[0], x1 + dx), min(size[1], y1 + dy)


def crop_to_product(image: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray, Box] | None:
    """Crop image and mask to the product with a small margin. None if there is no product."""
    box = mask_box(mask)
    if box is None:
        return None
    box = pad_box(box, image.size)
    x0, y0, x1, y1 = box
    return image.crop(box), mask[y0:y1, x0:x1], box
