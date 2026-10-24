from __future__ import annotations

from driftcheck.checks.base import TextThresholds
from driftcheck.checks.text import check_text, item_score
from driftcheck.ocr import TextBox
from driftcheck.vision import VisionError
from tests.checks_helpers import make_pair, thresholds
from tests.helpers import FakeVisionClient
from tests.synthetic import TEXT_BOX, garment

T = thresholds().text


def _ocr(*texts: str):
    """Fake OCR that finds the given texts on the product, ignoring the image."""
    return lambda image: [TextBox(text, 0.9, (TEXT_BOX[0], TEXT_BOX[1] + 20 * i, TEXT_BOX[2], TEXT_BOX[3] + 20 * i), "en_hi")
                          for i, text in enumerate(texts)]


def _pair(*original_texts: str):
    image = garment(text="SALE 50")
    return make_pair(image, garment(text="SALE 50"), ocr=_ocr(*original_texts)(image))


def test_pass_when_text_matches() -> None:
    result = check_text(_pair("SALE 50"), T, _ocr("sale 50!"))
    assert result.status == "pass" and result.score == 1.0
    assert result.details["items"][0]["found_by"] == "ocr"


def test_fail_when_text_changed() -> None:
    result = check_text(_pair("SALE 50"), T, _ocr("SOLE 80"), client=None)
    assert result.status == "fail" and "SALE 50" in result.reason
    assert result.score < T.min_match / 100


def test_fail_when_one_of_two_items_is_missing() -> None:
    result = check_text(_pair("SALE 50", "Pure Cotton"), T, _ocr("SALE 50"))
    assert result.status == "fail" and "1 of 2" in result.reason


def test_unknown_when_original_has_no_text() -> None:
    result = check_text(_pair(), T, _ocr("SALE 50"))
    assert result.status == "unknown" and "No text" in result.reason
    assert result.applicable is False


def test_hindi_and_kannada_items_match() -> None:
    result = check_text(_pair("शुद्ध घी", "ಹಾಲು"), T, _ocr("शुद्ध घी", "ಹಾಲು"))
    assert result.status == "pass"


def test_vision_backup_rescues_an_ocr_miss() -> None:
    client = FakeVisionClient([{"visible_text": ["SALE 50"]}])
    result = check_text(_pair("SALE 50"), T, _ocr(), client)
    assert result.status == "pass"
    assert result.details["items"][0]["found_by"] == "vision"


def test_vision_backup_does_not_rescue_changed_text() -> None:
    client = FakeVisionClient([{"visible_text": ["SOLE 80"]}])
    assert check_text(_pair("SALE 50"), T, _ocr(), client).status == "fail"


def test_vision_failure_is_recorded_not_raised() -> None:
    client = FakeVisionClient([VisionError("Ollama is down")])
    result = check_text(_pair("SALE 50"), T, _ocr(), client)
    assert result.status == "fail" and "Ollama is down" in result.details["vision_error"]


def test_backup_is_off_when_config_says_so() -> None:
    client = FakeVisionClient([])
    off = TextThresholds(min_match=T.min_match, min_item_length=T.min_item_length, vision_backup=False)
    check_text(_pair("SALE 50"), off, _ocr(), client)
    assert client.asked == []


def test_item_score_joined_text_for_split_words() -> None:
    score, _ = item_score("pure cotton", ["pure", "cotton"])
    assert score == 100
    short, _ = item_score("50", ["500"])
    assert short < 100
