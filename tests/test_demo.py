from __future__ import annotations

from pathlib import Path

from driftcheck.apps.demo_app import build_demo, check_word, examples, explain, guess_image_type, run_check
from tests.synthetic import garment
from tests.test_card import CARD
from tests.test_check_judge import SHIRT
from tests.test_pipeline import _checker


def test_auto_image_type() -> None:
    assert guess_image_type("auto", "x/studio__gemini_app.jpg") == "flat"
    assert guess_image_type("auto", "x/model_shot__gemini_app.jpg") == "scene"
    assert guess_image_type("auto", "C:/Temp/gradio/upload123.png") == "scene"  # no style name, so scene
    assert guess_image_type("flat", "x/model_shot__gemini_app.jpg") == "flat"  # an explicit choice wins


def test_check_words() -> None:
    assert check_word("pass", True) == "fine"
    assert check_word("fail", True) == "changed"
    assert check_word("unknown", True) == "not sure"
    assert check_word("unknown", False) == "not applicable"


def _save(tmp_path: Path, name: str, image) -> str:
    path = tmp_path / name
    image.save(path)
    return str(path)


def test_run_check_on_a_changed_flat_image(tmp_path: Path) -> None:
    original = _save(tmp_path, "original.png", garment(stripes=True, text="SALE 50"))
    changed = _save(tmp_path, "studio__gemini_app.png", garment(colour=(30, 140, 60), stripes=True, text="SALE 50"))
    summary, rows, heatmap, read_o, read_g, matches, timing = run_check(_checker([CARD, SHIRT, SHIRT]), original, changed, "auto")
    assert "Changed" in summary and "Colour" in summary
    assert ["Colour", "changed"] == rows[1][:2]
    assert heatmap is not None  # the pattern check ran on this flat image
    assert "SALE 50" in read_o and "SALE 50" in read_g
    assert matches[0][-1] == "found"


def test_run_check_on_a_scene_skips_colour_and_pattern(tmp_path: Path) -> None:
    original = _save(tmp_path, "original.png", garment(stripes=True, text="SALE 50"))
    upload = _save(tmp_path, "upload.png", garment(colour=(30, 140, 60), stripes=True, text="SALE 50"))
    summary, rows, heatmap, *_rest, timing = run_check(_checker([CARD, SHIRT, SHIRT]), original, upload, "auto")
    words = {row[0]: row[1] for row in rows}
    assert words["Colour"] == words["Pattern"] == "not applicable"
    assert "Looks fine" in summary
    assert heatmap is None and "did not run" in timing


def test_explain_lists_fired_checks() -> None:
    report = _checker([CARD, SHIRT, SHIRT]).check(garment(stripes=True), garment(colour=(30, 140, 60), stripes=True))
    words = explain(report)
    assert words["summary"].startswith("## Changed") and "Colour" in words["summary"]
    assert len(words["rows"]) == 6


def test_missing_upload_is_explained() -> None:
    assert run_check(None, None, None, "auto")[0] == "Upload both images first."


def test_examples_only_when_files_exist(tmp_path: Path) -> None:
    assert examples(tmp_path) == []
    garment().save(tmp_path / "plain_shirt_original.jpg")
    garment(colour=(30, 90, 170)).save(tmp_path / "plain_shirt_colour.jpg")
    found = examples(tmp_path)
    assert [e.slug for e in found] == ["colour"] and found[0].image_type == "flat"


def test_repo_ships_its_examples_outside_data() -> None:
    from driftcheck.apps.demo_app import EXAMPLES_DIR

    found = examples()
    assert [e.slug for e in found] == ["unchanged", "colour", "text", "stripes"]
    assert all("data" not in e.generated.relative_to(EXAMPLES_DIR.parent).parts for e in found)


def test_demo_builds(tmp_path: Path) -> None:
    assert build_demo(lambda: _checker([]), "Vision model: test.", tmp_path) is not None
