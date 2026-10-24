from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from driftcheck import pipeline
from driftcheck.cli import app
from driftcheck.heatmap import draw_heatmap, drift_pixels
from tests.test_card import CARD
from tests.test_check_judge import SHIRT
from tests.test_check_pattern import dotted
import numpy as np

from tests.test_pipeline import _checker
from tests.synthetic import TEXT_BOX, garment, text_boxes

BLUE = (30, 90, 170)  # not red, so red pixels in the heatmap are drawn marks only


def _report(original: Image.Image, output: Image.Image):
    return _checker([CARD, SHIRT, SHIRT]).check(original, output)


def ocr_if_text(image: Image.Image, scripts: list[str] | None = None):
    """Fake OCR: finds SALE 50 only when there are dark text pixels in the text box."""
    x0, y0, x1, y1 = TEXT_BOX
    return text_boxes() if (np.asarray(image)[y0:y1, x0:x1] < 60).all(axis=2).any() else []


def test_heatmap_for_same_product_has_no_red_drift() -> None:
    image = garment(colour=BLUE, stripes=True, text="SALE 50")
    report = _report(image, image.copy())
    heatmap = draw_heatmap(image, image.copy(), report)
    assert heatmap.height > 512 and heatmap.width >= 900
    assert drift_pixels(heatmap) < 50  # only status text, no painted patches or red boxes


def test_heatmap_paints_pattern_drift_on_the_output() -> None:
    original, output = garment(stripes=True), dotted()
    report = _report(original, output)
    assert report.check("pattern").status == "fail"
    assert drift_pixels(draw_heatmap(original, output, report)) > 5000


def test_heatmap_marks_missing_text_in_red() -> None:
    original = garment(colour=BLUE, text="SALE 50")
    output = garment(colour=BLUE)
    checker = _checker([CARD, {"visible_text": []}, SHIRT, SHIRT])
    checker.ocr_fn = ocr_if_text
    report = checker.check(original, output)
    assert report.check("text").status == "fail"
    with_box = draw_heatmap(original, output, report)
    report.check("text").details["items"] = []
    without_box = draw_heatmap(original, output, report)
    assert drift_pixels(with_box) > drift_pixels(without_box) + 100


def test_check_command_writes_report_and_heatmap(tmp_path: Path, monkeypatch) -> None:
    original, output = tmp_path / "o.jpg", tmp_path / "u.jpg"
    garment(stripes=True).save(original)
    garment(colour=(30, 140, 60), stripes=True).save(output)
    fake = _checker([CARD, SHIRT, SHIRT])
    fake.client.ensure_ready = lambda: None
    monkeypatch.setattr(pipeline.Checker, "from_settings", classmethod(lambda cls, settings=None, vision_model=None: fake))

    result = CliRunner().invoke(app, ["check", "--original", str(original), "--output", str(output), "--out", str(tmp_path / "r")])
    assert result.exit_code == 0, result.output
    assert "Verdict: drifted" in result.output
    report = json.loads((tmp_path / "r" / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "drifted" and len(report["checks"]) == 6
    assert Image.open(tmp_path / "r" / "heatmap.png").width >= 900
