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


def test_merge_keeps_more_confident_duplicate() -> None:
    boxes = [
        TextBox("SALE", 0.6, (10, 10, 50, 20), "en_kn"),
        TextBox("SALE", 0.9, (11, 10, 50, 21), "en_hi"),
        TextBox("50", 0.8, (60, 10, 80, 20), "en_hi"),
    ]
    merged = merge_boxes(boxes)
    assert [(b.text, b.reader) for b in merged] == [("SALE", "en_hi"), ("50", "en_hi")]


class FakeReader:
    def __init__(self, found):
        self.found = found
        self.calls = 0

    def readtext(self, arr):
        self.calls += 1
        return self.found


def test_ocr_engine_runs_both_readers_filters_and_caches(tmp_path: Path) -> None:
    hi = FakeReader([([[10, 10], [50, 10], [50, 20], [10, 20]], "SALE", 0.9), ([[0, 0], [5, 0], [5, 5], [0, 5]], "~", 0.2)])
    kn = FakeReader([([[60, 10], [80, 10], [80, 20], [60, 20]], "ಹಾಲು", 0.7)])
    engine = OcrEngine(DiskCache(tmp_path, "ocr"))
    engine._readers = {"en_hi": hi, "en_kn": kn}
    found = engine.read(garment())
    assert [b.text for b in found] == ["SALE", "ಹಾಲು"]
    assert found[0].box == (10, 10, 50, 20)

    again = OcrEngine(DiskCache(tmp_path, "ocr"))
    again._readers = {"en_hi": hi, "en_kn": kn}
    assert [b.text for b in again.read(garment())] == ["SALE", "ಹಾಲು"]
    assert hi.calls == 1 and kn.calls == 1
    assert set(ocr.READERS) == {"en_hi", "en_kn"}
