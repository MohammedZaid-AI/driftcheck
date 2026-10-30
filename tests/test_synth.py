from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from driftcheck.dataset import ProductMeta, product_dir, write_json
from driftcheck.labels import ATTRIBUTES
from driftcheck.synth import TEXTURE_MIN, make_drifts, mutate_text, run_synth, sleeve_erase, texture_score
from tests.synthetic import BODY, TEXT_BOX, garment, simple_mask, text_boxes


def _by_drift(items):
    return {item.drift: item for item in items}


def _diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    return np.abs(np.asarray(a, int) - np.asarray(b, int)).sum(axis=2)


def test_texture_score_separates_plain_and_striped(plain_shirt: Path) -> None:
    shirt = Image.open(plain_shirt).convert("RGB")
    shirt_mask = np.asarray(shirt).mean(axis=2) < 200
    assert texture_score(shirt, shirt_mask) < TEXTURE_MIN
    striped = garment(stripes=True)
    assert texture_score(striped, simple_mask(striped)) >= TEXTURE_MIN


def test_patterned_clothing_with_text_gets_every_drift() -> None:
    image = garment(stripes=True, text="SALE 50")
    items = _by_drift(make_drifts(image, simple_mask(image), text_boxes(), "clothing", random.Random(0)))
    assert set(items) == {
        "colour_shift", "text_erase", "text_replace", "pattern_blur", "pattern_swap",
        "sleeve_erase", "product_erase", "control_light", "control_jpeg", "control_shift",
    }
    for item in items.values():
        assert set(item.labels) == set(ATTRIBUTES)
        assert item.labels["logo_changed"] is None
        assert item.image.size == image.size


def test_labels_mark_only_the_drift() -> None:
    image = garment(stripes=True, text="SALE 50")
    items = _by_drift(make_drifts(image, simple_mask(image), text_boxes(), "clothing", random.Random(0)))
    assert items["colour_shift"].labels["colour_changed"] is True
    assert items["colour_shift"].labels["text_changed"] is False
    assert items["text_replace"].labels["text_changed"] is True
    assert items["pattern_swap"].labels["pattern_changed"] is True
    assert items["sleeve_erase"].labels["shape_changed"] is True
    assert items["product_erase"].labels["product_missing"] is True
    assert items["product_erase"].labels["colour_changed"] is None
    for control in ("control_light", "control_jpeg", "control_shift"):
        assert not any(v for v in items[control].labels.values())


def test_plain_product_without_text_skips_pattern_and_text() -> None:
    image = garment()
    items = _by_drift(make_drifts(image, simple_mask(image), [], "packaged_food", random.Random(0)))
    assert "pattern_blur" not in items and "pattern_swap" not in items
    assert "text_erase" not in items and "sleeve_erase" not in items
    assert items["colour_shift"].labels["text_changed"] is None  # no text, so not applicable


def test_drifts_change_the_right_pixels() -> None:
    image = garment(stripes=True, text="SALE 50")
    mask = simple_mask(image)
    items = _by_drift(make_drifts(image, mask, text_boxes(), "clothing", random.Random(0)))

    colour = _diff(image, items["colour_shift"].image)
    assert colour[mask].mean() > 30 and colour[~mask].max() == 0

    erased = _diff(image, items["text_erase"].image)
    x0, y0, x1, y1 = TEXT_BOX
    assert erased[y0:y1, x0:x1].sum() > 0
    assert erased[:y0 - 3].sum() == 0

    long = garment(long_sleeves=True)
    cut = sleeve_erase(long, simple_mask(long))
    assert cut is not None
    sleeves = _diff(long, cut.image)
    assert sleeves[130:170, 18:32].mean() > 100  # lower left sleeve painted out
    assert sleeves[52:90, 18:32].sum() == 0  # top of the sleeve stays, like a short sleeve
    assert sleeves[150:175, 60:100].sum() == 0  # body untouched

    gone = np.asarray(items["product_erase"].image)
    assert gone[BODY[1] + 5 : BODY[3] - 5, BODY[0] + 5 : BODY[2] - 5].min() > 240


def test_mutate_text_keeps_shape_but_changes_it() -> None:
    new = mutate_text("SALE 50", random.Random(1))
    assert new != "SALE 50" and len(new) == 7 and new[4] == " "


def test_run_synth_writes_images_json_and_labels(tmp_path: Path) -> None:
    folder = product_dir(tmp_path, "p001")
    folder.mkdir(parents=True)
    garment(stripes=True).save(folder / "original.jpg", quality=95)
    write_json(folder / "meta.json", ProductMeta(
        product_id="p001", category="clothing", source_file="x.jpg", sha256="1", ingested_at="2026-10-24",
    ))
    rows = run_synth(tmp_path, simple_mask, lambda image, scripts: [])
    assert {row["output_id"] for row in rows} >= {"colour_shift", "pattern_swap", "control_jpeg"}
    labels = json.loads((tmp_path / "synthetic" / "labels.json").read_text())
    assert labels == rows
    detail = json.loads((tmp_path / "synthetic" / "p001" / "colour_shift.json").read_text())
    assert detail["drift"] == "colour_shift" and "da" in detail["params"]
    assert (tmp_path / "synthetic" / "p001" / "colour_shift.jpg").exists()
