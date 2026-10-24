"""Text reading with easyocr, the main source for text.

easyocr cannot load every script in one reader, so three readers run:
English with Hindi, English with Kannada, and English with Bengali.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from driftcheck.cache import DiskCache, make_key
from driftcheck.segment import Box

READERS = {"en_hi": ["en", "hi"], "en_kn": ["en", "kn"], "en_bn": ["en", "bn"]}
MIN_CONFIDENCE = 0.3  # below this easyocr is mostly reading noise


@dataclass
class TextBox:
    text: str
    confidence: float
    box: Box
    reader: str


def normalise(text: str) -> str:
    """Casefold, unify unicode forms, drop punctuation and extra spaces.
    Keeps letters, digits and combining marks, so Hindi and Kannada vowel signs survive."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(c if unicodedata.category(c)[0] in "LMN" else " " for c in text)
    return re.sub(r"\s+", " ", text).strip()


def inside_mask(boxes: list, mask: np.ndarray) -> list:
    """Boxes whose centre is on the product. Text on the background is not product text."""
    inside = []
    for box in boxes:
        cx, cy = (box.box[0] + box.box[2]) // 2, (box.box[1] + box.box[3]) // 2
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and mask[cy, cx]:
            inside.append(box)
    return inside


def _iou(a: Box, b: Box) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def merge_boxes(boxes: list[TextBox], iou: float = 0.5) -> list[TextBox]:
    """Both readers often find the same English word. Keep the more confident one."""
    kept: list[TextBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        if all(_iou(box.box, other.box) < iou for other in kept):
            kept.append(box)
    return sorted(kept, key=lambda b: (b.box[1], b.box[0]))


class OcrEngine:
    def __init__(self, cache: DiskCache | None = None, gpu: bool = False) -> None:
        self.cache = cache
        self.gpu = gpu
        self._readers: dict = {}

    def _reader(self, name: str):
        if name not in self._readers:
            import easyocr

            self._readers[name] = easyocr.Reader(READERS[name], gpu=self.gpu, verbose=False)
        return self._readers[name]

    def _run(self, name: str, image: Image.Image) -> list[TextBox]:
        found = []
        for points, text, confidence in self._reader(name).readtext(np.asarray(image)):
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            found.append(TextBox(text, float(confidence), (min(xs), min(ys), max(xs), max(ys)), name))
        return found

    def read(self, image: Image.Image) -> list[TextBox]:
        """Every text item with confidence at least MIN_CONFIDENCE, merged across readers."""
        image = image.convert("RGB")
        png = io.BytesIO()
        image.save(png, format="PNG")
        key = make_key("easyocr", ",".join(READERS), png.getvalue())
        stored = self.cache.get(key) if self.cache else None
        if stored is not None:
            boxes = [TextBox(r["text"], r["confidence"], tuple(r["box"]), r["reader"]) for r in stored]
        else:
            boxes = [box for name in READERS for box in self._run(name, image)]
            if self.cache:
                self.cache.set(key, [asdict(box) for box in boxes])
        usable = [b for b in boxes if b.confidence >= MIN_CONFIDENCE and normalise(b.text)]
        return merge_boxes(usable)
