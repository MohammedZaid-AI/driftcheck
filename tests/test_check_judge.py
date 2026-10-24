from __future__ import annotations

from PIL import Image

from driftcheck.checks.base import JudgeThresholds
from driftcheck.checks.judge import DESCRIBE_PROMPT, check_shape_and_logo
from driftcheck.vision import VisionError
from tests.checks_helpers import make_pair, thresholds
from tests.helpers import FakeVisionClient
from tests.synthetic import garment
from tests.test_card import CARD

T = thresholds().judge
SHIRT = {"sleeve_length": "full", "neckline": "collar", "length": "hip", "silhouette": "fitted", "logo_visible": False}


def _shape(**changes) -> dict:
    return {**SHIRT, **changes}


def _run(original: dict | VisionError, output: dict | VisionError | None = None, card: dict = CARD,
         out_image: Image.Image | None = None, t: JudgeThresholds = T):
    replies = [original] if output is None else [original, output]
    client = FakeVisionClient(replies)
    pair = make_pair(garment(), out_image or garment(), vision=card)
    return client, check_shape_and_logo(pair, t, client)


def test_pass_when_descriptions_match() -> None:
    client, (shape, logo) = _run(SHIRT, SHIRT)
    assert shape.status == "pass" and shape.score == 0
    assert logo.status == "unknown"  # no logo on the original
    assert client.asked == [DESCRIBE_PROMPT, DESCRIBE_PROMPT]


def test_fail_when_full_sleeves_become_short() -> None:
    _, (shape, _) = _run(SHIRT, _shape(sleeve_length="short"))
    assert shape.status == "fail" and "sleeves full to short" in shape.reason
    assert shape.details["sleeves"]["steps"] == 3


def test_neighbouring_sleeve_lengths_are_not_a_change() -> None:
    _, (shape, _) = _run(SHIRT, _shape(sleeve_length="three_quarter"))
    assert shape.status == "pass"


def test_silhouette_counts_when_switched_on() -> None:
    on = JudgeThresholds(min_ordinal_steps=2, min_changed_parts=1, use_silhouette=True)
    _, (shape, _) = _run(SHIRT, _shape(silhouette="boxy"), t=on)
    assert shape.status == "fail"


def test_neckline_change_fails() -> None:
    _, (shape, _) = _run(SHIRT, _shape(neckline="v_neck"))
    assert shape.status == "fail"


def test_silhouette_can_be_switched_off() -> None:
    off = JudgeThresholds(min_ordinal_steps=2, min_changed_parts=1, use_silhouette=False)
    _, (shape, _) = _run(SHIRT, _shape(silhouette="boxy"), t=off)
    assert shape.status == "pass" and "silhouette" not in shape.details


def test_unknown_when_nothing_applies() -> None:
    na = {"sleeve_length": "not_applicable", "neckline": "not_applicable", "length": "not_applicable",
          "silhouette": "not_applicable", "logo_visible": True}
    _, (shape, _) = _run(na, na)
    assert shape.status == "unknown"


def test_logo_missing_fails_and_kept_passes() -> None:
    with_logo = {**CARD, "logos": [{"description": "leaf", "position": "left chest"}]}
    _, (_, gone) = _run(_shape(logo_visible=True), _shape(logo_visible=False), card=with_logo)
    assert gone.status == "fail"
    _, (_, kept) = _run(_shape(logo_visible=True), _shape(logo_visible=True), card=with_logo)
    assert kept.status == "pass"


def test_vision_failure_gives_unknown_for_both() -> None:
    _, results = _run(VisionError("Ollama is down"))
    assert [r.status for r in results] == ["unknown", "unknown"]
    assert all("Ollama is down" in r.reason for r in results)


def test_no_product_in_output_skips_the_calls() -> None:
    client, results = _run(SHIRT, SHIRT, out_image=Image.new("RGB", garment().size, "white"))
    assert [r.status for r in results] == ["unknown", "unknown"]
    assert client.asked == []
