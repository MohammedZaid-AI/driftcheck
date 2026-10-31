from __future__ import annotations

import numpy as np
from PIL import Image

from driftcheck.garment import garment_cutout, has_person, product_classes

LABELS = {0: "Background", 2: "Hair", 4: "Upper-clothes", 6: "Pants", 7: "Dress", 11: "Face", 14: "Left-arm", 17: "Scarf"}


class FakeParser:
    """Returns a prepared label map per image, picked by the image's red value."""

    labels = LABELS

    def __init__(self, maps: dict[int, np.ndarray]) -> None:
        self.maps = maps

    def __call__(self, image: Image.Image) -> np.ndarray:
        return self.maps[image.getpixel((0, 0))[0]]


def _image(key: int) -> Image.Image:
    return Image.new("RGB", (20, 20), (key, 0, 0))


def _flat_top() -> np.ndarray:
    """A flat lay top on its own: the parser calls it Dress."""
    m = np.zeros((20, 20), np.uint8)
    m[4:16, 4:16] = 7
    return m


def _worn_top() -> np.ndarray:
    """A model wearing the top over pants: face, arms, top and pants."""
    m = np.zeros((20, 20), np.uint8)
    m[0:4, 6:14] = 11  # face
    m[4:12, 6:14] = 4  # top, now called Upper-clothes
    m[4:12, 2:6] = 14  # arm
    m[12:20, 6:14] = 6  # pants
    return m


def test_person_detection() -> None:
    assert has_person(_worn_top(), LABELS)
    assert not has_person(_flat_top(), LABELS)


def test_biggest_group_is_the_product() -> None:
    whole_person = _worn_top() > 0
    assert product_classes(_worn_top(), whole_person, LABELS) == ["Dress", "Upper-clothes"]
    assert product_classes(np.zeros((20, 20), np.uint8), whole_person, LABELS) == []
    assert "Scarf" in product_classes(_flat_top(), whole_person, LABELS, category="accessory")


def test_worn_output_keeps_only_the_garment() -> None:
    parser = FakeParser({1: _flat_top(), 2: _worn_top()})
    original_rembg = _flat_top() > 0
    output_rembg = _worn_top() > 0  # rembg keeps the whole person
    cut = garment_cutout(_image(1), _image(2), original_rembg, output_rembg, parser, "clothing")
    assert cut.info["method"] == "clothes parsing"
    assert cut.info["original_cut"] == "rembg, no person"
    assert np.array_equal(cut.original_mask, original_rembg)
    assert cut.output_mask.sum() == 64  # only the 8 by 8 top
    assert not cut.output_mask[0:4].any() and not cut.output_mask[12:].any()  # no face, no pants


def test_no_garment_on_the_original_keeps_rembg() -> None:
    parser = FakeParser({1: np.zeros((20, 20), np.uint8), 2: _worn_top()})
    rembg = np.ones((20, 20), bool)
    cut = garment_cutout(_image(1), _image(2), rembg, rembg, parser, "clothing")
    assert cut.info["method"] == "rembg" and "no garment class" in cut.info["reason"]
    assert cut.output_mask.all()


def test_tiny_garment_on_the_output_keeps_rembg() -> None:
    tiny = np.zeros((20, 20), np.uint8)
    tiny[0:6, 0:6] = 11  # a face, so a person, but no garment at all
    parser = FakeParser({1: _flat_top(), 2: tiny})
    cut = garment_cutout(_image(1), _image(2), _flat_top() > 0, np.ones((20, 20), bool), parser, "clothing")
    assert cut.info["output_cut"] == "rembg, garment too small"
    assert cut.output_mask.all()
