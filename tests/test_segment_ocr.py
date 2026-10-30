from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from driftcheck import ocr, segment
from driftcheck.cache import DiskCache
from driftcheck.ocr import OcrEngine, TextBox, merge_boxes, normalise
from tests.synthetic import garment


def test_product_mask_is_cached(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_rembg(image: Image.Image) -> Image.Image:
        calls.append(1)
        arr = np.zeros((image.height, image.width), np.uint8)
        arr[10:20, 5:15] = 255
        return Image.fromarray(arr)

    monkeypatch.setattr(segment, "_rembg_mask", fake_rembg)
    cache = DiskCache(tmp_path, "masks")
    first = segment.product_mask(garment(), cache)
    second = segment.product_mask(garment(), cache)
    assert len(calls) == 1
    assert first.dtype == bool and first.sum() == 100
    assert np.array_equal(first, second)


def test_crop_to_product() -> None:
    mask = np.zeros((100, 80), bool)
    mask[20:60, 10:50] = True
    crop, crop_mask, box = segment.crop_to_product(Image.new("RGB", (80, 100)), mask)
    assert box == (9, 19, 51, 61)
    assert crop.size == (42, 42) and crop_mask.shape == (42, 42)
    assert segment.crop_to_product(Image.new("RGB", (80, 100)), np.zeros((100, 80), bool)) is None


def test_normalise() -> None:
    assert normalise("  SALE—50%  OFF! ") == "sale 50 off"
    assert normalise("ＡＢＣ") == "abc"
    assert normalise("नमस्ते!") == "नमस्ते"
    assert normalise("ಹಾಲು,") == "ಹಾಲು"
    assert normalise("রসুই!") == "রসুই"


def test_merge_keeps_more_confident_duplicate() -> None:
    boxes = [
        TextBox("SALE", 0.6, (10, 10, 50, 20), "en_kn"),
        TextBox("SALE", 0.9, (11, 10, 50, 21), "en_hi"),
        TextBox("50", 0.8, (60, 10, 80, 20), "en_hi"),
    ]
    merged = merge_boxes(boxes)
    assert [(b.text, b.reader) for b in merged] == [("SALE", "en_hi"), ("50", "en_hi")]


def _quad(box):
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class FakeReader:
    """Returns fixed (box, text, confidence) rows. Rows given for 'crop' are returned
    for images smaller than the full test image, as on the 2x retry."""

    def __init__(self, rows, crop_rows=None, full_size=(160, 200)):
        self.rows = rows
        self.crop_rows = crop_rows or []
        self.full_size = full_size
        self.calls = 0

    def readtext(self, arr, canvas_size=None):
        self.calls += 1
        self.canvas_size = canvas_size
        rows = self.rows if (arr.shape[1], arr.shape[0]) == self.full_size else self.crop_rows
        return [(_quad(box), text, conf) for box, text, conf in rows]


# Boxes and readings shaped like the real p045 run: the same boxes for every reader,
# real Bengali read well by the Bengali reader, made up Devanagari from the Hindi reader.
P045_LIKE = {
    "en_hi": [((10, 10, 40, 20), "नछ्न", 0.78), ((50, 10, 90, 20), "श्ञम", 0.09), ((10, 30, 60, 40), "टनाज्छ", 0.29),
              ((10, 50, 80, 60), "ROAST MASALA", 0.99), ((10, 70, 40, 80), "कातनि", 0.2)],
    "en_kn": [((10, 10, 40, 20), "F9r", 0.71), ((50, 10, 90, 20), "B3]7", 0.02), ((10, 30, 60, 40), "@೯ೆ", 0.19),
              ((10, 50, 80, 60), "ROAST MASALA", 0.57), ((10, 70, 40, 80), "C", 0.03)],
    "en_bn": [((10, 10, 40, 20), "নতুণ", 0.3), ((50, 10, 90, 20), "ইফাদ", 0.9), ((10, 30, 60, 40), "রোস্ট", 0.94),
              ((10, 50, 80, 60), "ROAST MASALA", 0.98), ((10, 70, 40, 80), "কোরি", 0.94)],
}


def _engine(tmp_path: Path, rows: dict, crop_rows: dict | None = None) -> OcrEngine:
    engine = OcrEngine(DiskCache(tmp_path, "ocr"))
    engine._readers = {name: FakeReader(r, (crop_rows or {}).get(name)) for name, r in rows.items()}
    engine._readers.setdefault("en", FakeReader([]))
    return engine


def test_text_script() -> None:
    assert ocr.text_script("ROAST") == "en"
    assert ocr.text_script("रसोई") == "hi"
    assert ocr.text_script("ರಸ") == "kn"
    assert ocr.text_script("রসুই") == "bn"
    assert ocr.text_script("200 / 50") is None


def test_choose_scripts_drops_made_up_devanagari() -> None:
    boxes = [TextBox(t, c, b, n) for n, rows in P045_LIKE.items() for b, t, c in rows]
    assert ocr.choose_scripts(boxes) == ["en", "bn"]


def test_choose_scripts_needs_a_confident_reading() -> None:
    boxes = [TextBox("मोती", 0.57, (0, 0, 10, 10), "en_hi"), TextBox("Moti", 1.0, (0, 20, 10, 30), "en_hi")]
    assert ocr.choose_scripts(boxes) == ["en"]  # Hindi under 0.7 is missed; fix it by hand in selection.json


def test_read_uses_only_the_product_scripts(tmp_path: Path) -> None:
    engine = _engine(tmp_path, P045_LIKE)
    found = engine.read(garment(), ["en", "bn"])
    assert [b.text for b in found] == ["নতুণ", "ইফাদ", "রোস্ট", "ROAST MASALA", "কোরি"]
    assert all(b.reader == "en_bn" for b in found)
    assert engine._readers["en_hi"].calls == 0 and engine._readers["en_kn"].calls == 0
    assert engine._readers["en_bn"].canvas_size == ocr.CANVAS_SIZE


def test_read_without_native_script_uses_english_reader(tmp_path: Path) -> None:
    engine = _engine(tmp_path, {"en": [((10, 50, 80, 60), "SALE", 0.95)]})
    assert [b.text for b in engine.read(garment(), ["en"])] == ["SALE"]


def test_read_detects_scripts_when_not_given(tmp_path: Path) -> None:
    engine = _engine(tmp_path, P045_LIKE)
    assert engine.detect_scripts(garment()) == ["en", "bn"]
    assert "नछ्न" not in [b.text for b in engine.read(garment())]


def test_unsure_title_is_read_again_at_twice_the_size(tmp_path: Path) -> None:
    title = (20, 80, 140, 150)  # tall box, 35% of the image height
    rows = {"en_bn": [(title, "রস্ুই", 0.3)]}
    # In the 2x crop: a neighbour read very surely, and the title read better than before.
    crop = {"en_bn": [((10, 0, 60, 20), "ইফদ", 0.97), ((30, 30, 270, 170), "রসুই", 0.6)]}
    found = _engine(tmp_path, rows, crop).read(garment(), ["en", "bn"])
    assert [(b.text, b.reader, b.box) for b in found] == [("রসুই", "en_bn_x2", title)]


def test_reader_results_are_cached(tmp_path: Path) -> None:
    first = _engine(tmp_path, P045_LIKE)
    first.read(garment(), ["en", "bn"])
    again = _engine(tmp_path, P045_LIKE)
    again.read(garment(), ["en", "bn"])
    assert again._readers["en_bn"].calls == 0
