"""Text reading with easyocr, the main source for text.

easyocr cannot load every script in one reader, so there is one reader per
script, each also reading English: Hindi, Kannada and Bengali. The scripts on a
product are detected once from its original and saved as text_scripts. Checks
then run English plus only those scripts, on both the original and the output.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from driftcheck.cache import DiskCache, make_key
from driftcheck.segment import Box, pad_box

READERS = {"en": ["en"], "en_hi": ["en", "hi"], "en_kn": ["en", "kn"], "en_bn": ["en", "bn"]}
NATIVE_READERS = {"hi": "en_hi", "kn": "en_kn", "bn": "en_bn"}  # script code to the reader that reads it
MIN_CONFIDENCE = 0.3  # below this easyocr is mostly reading noise
SCRIPT_CONFIDENCE = 0.7  # a reading this sure counts as proof the script is there
TITLE_MIN_HEIGHT = 0.03  # unsure boxes at least this share of the image height are read again at 2x


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


# Unicode blocks of the scripts we read. Marathi uses Devanagari, so it counts as hi.
_SCRIPT_RANGES = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "kn": (0x0C80, 0x0CFF)}


def text_script(text: str) -> str | None:
    """The script most letters of text are in: en, hi, kn or bn. None for digits and symbols only."""
    counts: dict[str, int] = {}
    for char in text:
        if char.isascii() and char.isalpha():
            counts["en"] = counts.get("en", 0) + 1
            continue
        for script, (low, high) in _SCRIPT_RANGES.items():
            if low <= ord(char) <= high:
                counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def choose_scripts(boxes: list[TextBox]) -> list[str]:
    """Scripts that really appear, from all readers run on one original.

    The readers share one text detector, so they see the same boxes and only the
    reading differs. Each box goes to the reader most confident about it. A script
    is kept when it wins a box at SCRIPT_CONFIDENCE or more, and wins at least half
    as many boxes as the strongest script. The second rule drops a reader that
    reads another script as made up words, as the Hindi reader does on Bengali.

    This is a first guess only. easyocr reads real Hindi at 0.36 to 0.57 confidence
    on these photos, and once read Hindi as made up Bengali at 0.74. Check the result
    and fix it by hand in config/selection.json.
    """
    winners: list[TextBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        if all(_iou(box.box, other.box) < 0.5 for other in winners):
            winners.append(box)
    wins: dict[str, int] = {}
    for box in winners:
        script = text_script(box.text)
        if script and box.confidence >= SCRIPT_CONFIDENCE:
            wins[script] = wins.get(script, 0) + 1
    native = {s: n for s, n in wins.items() if s != "en"}
    top = max(native.values(), default=0)
    kept = [s for s, n in native.items() if n * 2 >= top]
    return (["en"] if "en" in wins else []) + sorted(kept)


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
        """One reader on one image, cached per reader so detection and checks share results."""
        key = make_key("easyocr", name, _png(image))
        if self.cache and (stored := self.cache.get(key)) is not None:
            return [TextBox(r["text"], r["confidence"], tuple(r["box"]), r["reader"]) for r in stored]
        found = []
        for points, text, confidence in self._reader(name).readtext(np.asarray(image)):
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            found.append(TextBox(text, float(confidence), (min(xs), min(ys), max(xs), max(ys)), name))
        if self.cache:
            self.cache.set(key, [asdict(box) for box in found])
        return found

    def detect_scripts(self, image: Image.Image) -> list[str]:
        """Run every reader once on an original and keep the scripts that really appear."""
        image = image.convert("RGB")
        return choose_scripts([box for name in NATIVE_READERS.values() for box in self._run(name, image)])

    def _readers_for(self, scripts: list[str]) -> list[str]:
        names = [NATIVE_READERS[s] for s in scripts if s in NATIVE_READERS]
        return names or ["en"]

    def _upscaled(self, image: Image.Image, box: TextBox, names: list[str], allowed: set[str]) -> TextBox | None:
        """Read a missed or unsure text area again at twice the size. Large stylised titles
        are often read better this way. Returns the best reading, placed at the original box."""
        x0, y0, x1, y1 = pad_box(box.box, image.size, pad=0.25)
        crop = image.crop((x0, y0, x1, y1))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
        found = []
        for name in names:
            for b in self._run(name, crop):
                # Back to photo coordinates. The padding can pull in neighbouring text,
                # so the reading that overlaps the retried box most is the one for it.
                placed = (x0 + b.box[0] // 2, y0 + b.box[1] // 2, x0 + b.box[2] // 2, y0 + b.box[3] // 2)
                overlap = _iou(placed, box.box)
                if overlap >= 0.3 and text_script(b.text) in allowed:
                    found.append((overlap, b))
        if not found:
            return None
        _, best = max(found, key=lambda pair: (pair[0], pair[1].confidence))
        return TextBox(best.text, best.confidence, box.box, best.reader + "_x2")

    def read(self, image: Image.Image, scripts: list[str] | None = None) -> list[TextBox]:
        """Text items in English plus the given scripts, merged across readers.
        Without scripts, they are detected from this image first."""
        image = image.convert("RGB")
        if scripts is None:
            scripts = self.detect_scripts(image)
        names = self._readers_for(scripts)
        allowed = {"en", *scripts, None}
        boxes = [b for name in names for b in self._run(name, image) if text_script(b.text) in allowed]
        merged = merge_boxes([b for b in boxes if normalise(b.text)])

        result = []
        for box in merged:
            tall = (box.box[3] - box.box[1]) >= TITLE_MIN_HEIGHT * image.height
            if box.confidence < SCRIPT_CONFIDENCE and tall:
                retry = self._upscaled(image, box, names, allowed)
                if retry and retry.confidence > box.confidence:
                    box = retry
            if box.confidence >= MIN_CONFIDENCE and normalise(box.text):
                result.append(box)
        return result
