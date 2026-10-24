"""Pixel colour palette: k means in Lab colour space on the product mask.

Pixel colours are more reliable than colours the model describes in words.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import binary_erosion

MAX_PIXELS = 20000
MIN_SHARE = 0.02  # clusters smaller than this are edge noise


class PaletteColour(BaseModel):
    lab: tuple[float, float, float]
    hex: str
    share: float


def grey_world(image: Image.Image) -> Image.Image:
    """Scale each channel so its mean matches the overall mean. Removes a colour cast from lighting."""
    arr = np.asarray(image.convert("RGB"), float)
    means = arr.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / np.maximum(means, 1e-6)
    return Image.fromarray(np.clip(arr * gains, 0, 255).astype(np.uint8))


def _hex(lab: np.ndarray) -> str:
    from skimage.color import lab2rgb

    rgb = (np.clip(lab2rgb(lab.reshape(1, 1, 3)), 0, 1) * 255).round().astype(int).ravel()
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def product_pixels_lab(image: Image.Image, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Lab values of product pixels, away from the edge, at most MAX_PIXELS of them."""
    from skimage.color import rgb2lab

    inner = binary_erosion(mask, iterations=2)
    if inner.sum() < 50:
        inner = mask
    pixels = np.asarray(image.convert("RGB"))[inner]
    if len(pixels) > MAX_PIXELS:
        pixels = pixels[np.random.default_rng(seed).choice(len(pixels), MAX_PIXELS, replace=False)]
    return rgb2lab(pixels.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)


def palette(image: Image.Image, mask: np.ndarray, k: int = 5) -> list[PaletteColour]:
    """Main product colours with their share of product pixels, largest first."""
    from sklearn.cluster import KMeans

    if mask.sum() == 0:
        return []
    lab = product_pixels_lab(image, mask)
    k = min(k, len(np.unique(lab.round(1), axis=0)))
    model = KMeans(n_clusters=k, n_init=4, random_state=0).fit(lab)
    counts = np.bincount(model.labels_, minlength=k) / len(lab)
    colours = [
        PaletteColour(lab=tuple(round(float(v), 2) for v in centre), hex=_hex(centre), share=round(float(share), 4))
        for centre, share in zip(model.cluster_centers_, counts)
        if share >= MIN_SHARE
    ]
    total = sum(c.share for c in colours)
    for colour in colours:
        colour.share = round(colour.share / total, 4)
    return sorted(colours, key=lambda c: -c.share)
