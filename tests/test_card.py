from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from driftcheck.cache import DiskCache
from driftcheck.card import CARD_PROMPT, VisionCard, build_card, text_crops
from driftcheck.palette import grey_world, palette
from driftcheck.vision import VisionError
from tests.helpers import FakeVisionClient
from tests.synthetic import garment, simple_mask, text_boxes

CARD = {
    "product_type": "striped sleeve top",
    "category": "clothing",
    "visible_views": ["front"],
    "sleeve_length": "short",
    "neckline": "round",
    "silhouette": "straight",
    "length": "hip",
    "closures": [],
    "main_colours": ["red", "blue"],
    "pattern_type": "stripes",
    "pattern_description": "vertical blue stripes",
    "embellishments": [],
    "logos": [],
    "text_items": [{"text": "SALE 50", "language": "English", "position": "chest"}],
    "key_features": [],
}


def _two_colour() -> tuple[Image.Image, np.ndarray]:
    """Left 3/4 red, right 1/4 blue, all product."""
    img = Image.new("RGB", (80, 40), (200, 20, 20))
    ImageDraw.Draw(img).rectangle((60, 0, 79, 39), fill=(20, 20, 200))
    return img, np.ones((40, 80), bool)


def test_palette_finds_colours_and_shares() -> None:
    img, mask = _two_colour()
    colours = palette(img, mask, k=5)
    assert len(colours) == 2
    assert abs(sum(c.share for c in colours) - 1) < 1e-3
    red, blue = colours
    assert 0.70 < red.share < 0.80
    assert red.lab[1] > 50  # positive a* is red
    assert blue.lab[2] < -50  # negative b* is blue
    assert red.hex.startswith("#c")


def test_palette_ignores_background_and_empty_mask() -> None:
    img = garment(colour=(30, 160, 60), sleeves=False)
    colours = palette(img, simple_mask(img))
    assert colours[0].share > 0.95 and colours[0].lab[1] < -30  # green, no white
    assert palette(img, np.zeros((img.height, img.width), bool)) == []


def test_grey_world_removes_a_colour_cast() -> None:
    img = garment(colour=(120, 120, 120), sleeves=False)
    warm = Image.fromarray(np.clip(np.asarray(img) * [1.15, 1.0, 0.85], 0, 255).astype(np.uint8))
    fixed = np.asarray(grey_world(warm), float).reshape(-1, 3).mean(axis=0)
    assert np.ptp(fixed) < 3


def test_text_crops_are_zoomed() -> None:
    crops = text_crops(garment(text="SALE 50"), text_boxes())
    assert len(crops) == 1 and max(crops[0].size) == 768


def test_build_card_sends_product_and_text_crops(tmp_path: Path) -> None:
    client = FakeVisionClient([CARD])
    sent = []
    original_ask = client.ask

    def spy(prompt, images, schema):
        sent.append(images)
        return original_ask(prompt, images, schema)

    client.ask = spy  # type: ignore[method-assign]
    img = garment(stripes=True, text="SALE 50")
    card = build_card(img, client, simple_mask, lambda _: text_boxes(), DiskCache(tmp_path, "cards"))

    assert card.vision is not None and card.vision.sleeve_length == "short"
    assert client.asked == [CARD_PROMPT]
    assert len(sent[0]) == 2  # product crop and one zoomed text crop
    assert sent[0][0].size[0] < img.size[0]  # cropped to the product
    assert [o.text for o in card.ocr] == ["SALE 50"]
    assert len(card.palette) >= 2
    assert card.product_box is not None and 0.3 < card.product_share < 0.6
    assert card.vision_model == "fake/fake-vl"


def test_card_is_cached(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, "cards")
    img = garment()
    build_card(img, FakeVisionClient([CARD]), simple_mask, lambda _: [], cache)
    again = FakeVisionClient([])
    assert build_card(img, again, simple_mask, lambda _: [], cache).vision.product_type == "striped sleeve top"
    assert again.asked == []


def test_failed_vision_keeps_pixels_and_is_not_cached(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, "cards")
    card = build_card(garment(), FakeVisionClient([VisionError("Ollama is down")]), simple_mask, lambda _: [], cache)
    assert card.vision is None and "Ollama is down" in card.vision_error
    assert card.palette
    retry = FakeVisionClient([CARD])
    assert build_card(garment(), retry, simple_mask, lambda _: [], cache).vision is not None


def test_no_product_means_no_vision_call() -> None:
    client = FakeVisionClient([])
    card = build_card(Image.new("RGB", (50, 50), "white"), client, lambda im: np.zeros((50, 50), bool), lambda _: [])
    assert card.vision is None and card.vision_error == "No product found in the original."
    assert client.asked == [] and card.product_box is None


def test_schema_is_accepted_by_vision_card() -> None:
    assert VisionCard.model_validate(CARD).text_items[0].text == "SALE 50"
