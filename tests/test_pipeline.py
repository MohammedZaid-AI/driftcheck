from __future__ import annotations

from PIL import Image

from driftcheck.checks.base import CheckResult
from driftcheck.checks.verdict import CHECK_ATTRIBUTE, overall_verdict
from driftcheck.pipeline import CHECK_ORDER, Checker
from driftcheck.vision import VisionError
from tests.checks_helpers import thresholds
from tests.helpers import FakeVisionClient
from tests.synthetic import garment, simple_mask, text_boxes
from tests.test_card import CARD
from tests.test_check_judge import SHIRT
from tests.test_check_pattern import fake_features


def _result(status: str) -> CheckResult:
    return CheckResult(name="x", status=status, reason="r")


def test_verdict_rules() -> None:
    assert overall_verdict([_result("pass"), _result("pass")]) == "faithful"
    assert overall_verdict([_result("pass"), _result("fail"), _result("unknown")]) == "drifted"
    assert overall_verdict([_result("pass"), _result("unknown")]) == "unknown"
    assert overall_verdict([]) == "unknown"
    not_applicable = CheckResult(name="logo", status="unknown", reason="no logo", applicable=False)
    assert overall_verdict([_result("pass"), not_applicable]) == "faithful"
    assert overall_verdict([not_applicable]) == "unknown"


def test_every_check_maps_to_a_label_attribute() -> None:
    assert set(CHECK_ORDER) == set(CHECK_ATTRIBUTE)


def _checker(replies: list) -> Checker:
    client = FakeVisionClient(replies)
    client.calls = []  # the pipeline counts real calls from this list
    return Checker(thresholds=thresholds(), client=client, mask_fn=simple_mask,
                   ocr_fn=lambda image, scripts=None: text_boxes(), features_fn=fake_features, pattern_model="fake/dino")


def test_faithful_pair() -> None:
    image = garment(stripes=True, text="SALE 50")
    report = _checker([CARD, SHIRT, SHIRT]).check(image, image.copy())
    assert [c.name for c in report.checks] == list(CHECK_ORDER)
    statuses = {c.name: c.status for c in report.checks}
    assert statuses == {"product_missing": "pass", "colour": "pass", "text": "pass", "pattern": "pass",
                        "shape": "pass", "logo": "unknown"}
    assert report.check("logo").applicable is False
    assert report.verdict == "faithful"  # logo does not apply, everything else passed
    assert report.thresholds_tuned is False
    assert report.models["pattern"] == "fake/dino"
    assert set(report.check_seconds) >= {"card", "colour", "pattern"}


def test_colour_drift_is_drifted() -> None:
    original = garment(stripes=True, text="SALE 50")
    output = garment(colour=(30, 140, 60), stripes=True, text="SALE 50")
    report = _checker([CARD, SHIRT, SHIRT]).check(original, output)
    assert report.check("colour").status == "fail"
    assert report.verdict == "drifted"


def test_scene_image_skips_colour_and_pattern() -> None:
    original = garment(stripes=True, text="SALE 50")
    output = garment(colour=(30, 140, 60), stripes=True, text="SALE 50")  # a colour change
    flat = _checker([CARD, SHIRT, SHIRT]).check(original, output)
    scene = _checker([CARD, SHIRT, SHIRT]).check(original, output, image_type="scene")
    assert flat.check("colour").status == "fail" and flat.verdict == "drifted"
    for name in ("colour", "pattern"):
        assert scene.check(name).status == "unknown" and scene.check(name).applicable is False
        assert "scene" in scene.check(name).reason
    assert scene.verdict == "faithful"  # text and shape passed; skipped checks do not count


def test_missing_product_skips_other_checks() -> None:
    blank = Image.new("RGB", garment().size, "white")
    checker = _checker([CARD])
    report = checker.check(garment(), blank)
    assert report.check("product_missing").status == "fail"
    assert all(c.status == "unknown" for c in report.checks if c.name != "product_missing")
    assert report.verdict == "drifted"
    assert checker.client.asked == [checker.client.asked[0]]  # only the card call


def test_large_images_are_capped_on_load() -> None:
    from PIL import Image

    from driftcheck.pipeline import MAX_CHECK_SIDE, load_image

    big = Image.new("RGB", (3000, 4500), "white")
    assert max(load_image(big).size) == MAX_CHECK_SIDE
    assert big.size == (3000, 4500)  # the caller's image is not changed
    small = Image.new("RGB", (400, 300), "white")
    assert load_image(small).size == (400, 300)


def test_vision_failure_gives_unknown_not_crash() -> None:
    image = garment(stripes=True)
    report = _checker([VisionError("down"), VisionError("down")]).check(image, image.copy())
    assert report.check("shape").status == "unknown"
    assert report.card.vision is None
    assert report.verdict == "unknown"
