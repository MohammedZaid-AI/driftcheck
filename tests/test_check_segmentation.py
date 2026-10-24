from __future__ import annotations

import numpy as np
from PIL import Image

from driftcheck.checks.segmentation import check_segmentation
from tests.checks_helpers import make_pair, thresholds
from tests.synthetic import garment

T = thresholds().segmentation


def test_pass_when_product_is_there() -> None:
    result = check_segmentation(make_pair(garment(), garment()), T)
    assert result.status == "pass" and result.name == "product_missing"
    assert result.score > 0.3


def test_fail_when_product_is_gone() -> None:
    blank = Image.new("RGB", garment().size, "white")
    result = check_segmentation(make_pair(garment(), blank), T)
    assert result.status == "fail" and "missing" in result.reason


def test_unknown_when_product_is_tiny() -> None:
    tiny = np.zeros((200, 160), bool)
    tiny[100:125, 80:105] = True  # 625 of 32000 pixels, about 2%, between missing and small
    result = check_segmentation(make_pair(garment(), garment(), output_mask=tiny), T)
    assert result.status == "unknown" and "Too small" in result.reason


def test_unknown_when_original_has_no_product() -> None:
    blank = Image.new("RGB", garment().size, "white")
    assert check_segmentation(make_pair(blank, garment()), T).status == "unknown"


def test_threshold_comes_from_config() -> None:
    assert check_segmentation(make_pair(garment(), garment()), T).threshold == T.missing_share
