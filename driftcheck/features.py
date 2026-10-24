"""DINOv2 patch features on the CPU, cached on disk."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from driftcheck.cache import DiskCache, make_key

SIDE = 448  # 32 by 32 patches of 14 pixels
PATCH = 14
GRID = SIDE // PATCH
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def to_square(image: Image.Image, mask: np.ndarray, fill: str = "white") -> tuple[Image.Image, np.ndarray, int, tuple[int, int]]:
    """Pad to a square without stretching, then resize to SIDE. Returns the image,
    the mask, the padded side in crop pixels and the crop's offset inside the square."""
    w, h = image.size
    side = max(w, h)
    offset = ((side - w) // 2, (side - h) // 2)
    square = Image.new("RGB", (side, side), fill)
    square.paste(image.convert("RGB"), offset)
    square_mask = np.zeros((side, side), bool)
    square_mask[offset[1] : offset[1] + h, offset[0] : offset[0] + w] = mask
    resized_mask = np.asarray(Image.fromarray(square_mask).resize((SIDE, SIDE), Image.Resampling.NEAREST))
    return square.resize((SIDE, SIDE), Image.Resampling.BICUBIC), resized_mask, side, offset


def patch_mask(mask: np.ndarray) -> np.ndarray:
    """A patch is product if at least half its pixels are product."""
    return mask.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)) >= 0.5


class PatchFeatures:
    """Callable that turns a SIDE by SIDE image into a GRID by GRID by D array of unit vectors."""

    def __init__(self, model_name: str, cache: DiskCache | None = None) -> None:
        self.model_name = model_name
        self.cache = cache
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            torch.set_grad_enabled(False)
            self._model = AutoModel.from_pretrained(self.model_name).eval()
        return self._model

    def __call__(self, image: Image.Image) -> np.ndarray:
        png = io.BytesIO()
        image.save(png, format="PNG")
        key = make_key("dinov2", self.model_name, png.getvalue())
        if self.cache and (stored := self.cache.get_bytes(key)) is not None:
            return np.load(io.BytesIO(stored))

        import torch

        arr = (np.asarray(image.convert("RGB"), np.float32) / 255.0 - _MEAN) / _STD
        pixels = torch.from_numpy(arr.transpose(2, 0, 1)[None].astype(np.float32))
        tokens = self._load()(pixel_values=pixels).last_hidden_state[0]
        patches = tokens[-GRID * GRID :].numpy().reshape(GRID, GRID, -1)  # drop the class token
        patches = patches / np.linalg.norm(patches, axis=-1, keepdims=True)
        if self.cache:
            out = io.BytesIO()
            np.save(out, patches.astype(np.float32))
            self.cache.set_bytes(key, out.getvalue())
        return patches
