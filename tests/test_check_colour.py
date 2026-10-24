from __future__ import annotations

import random

import numpy as np
from PIL import Image

from driftcheck.checks.base import ColourThresholds
from driftcheck.checks.colour import check_colour
from driftcheck.synth import colour_shift, control_light
from tests.checks_helpers import make_pair, thresholds
from tests.synthetic import garment, simple_mask

T = thresholds().colour


def test_pass_for_same_product() -> None:
    result = check_colour(make_pair(garment(stripes=True), garment(stripes=True)), T)
    assert result.status == "pass" and result.score < 2


def test_pass_under_warm_lighting_with_white_balance() -> None:
    original = garment(colour=(150, 150, 150), sleeves=False)
    warm = control_light(original).image
    assert check_colour(make_pair(original, warm), T).status == "pass"


def test_fail_for_different_colour() -> None:
    result = check_colour(make_pair(garment(colour=(190, 30, 40)), garment(colour=(30, 140, 60))), T)
    assert result.status == "fail" and result.score > T.max_weighted_delta_e
    assert result.details["matches"][0]["delta_e"] > 30


def test_fail_for_synthetic_colour_shift() -> None:
    original = garment(stripes=True)
    shifted = colour_shift(original, simple_mask(original), random.Random(3)).image
    assert check_colour(make_pair(original, shifted), T).status == "fail"


def test_extra_output_colours_do_not_count() -> None:
    original = garment(sleeves=False)
    output = garment(sleeves=False)
    arr = np.asarray(output).copy()
    arr[5:45, 60:100] = (224, 172, 140)  # a skin coloured area added next to the product
    output = Image.fromarray(arr)
    assert check_colour(make_pair(original, output), T).status == "pass"


def test_unknown_when_no_product_pixels() -> None:
    blank = Image.new("RGB", garment().size, "white")
    result = check_colour(make_pair(garment(), blank), T)
    assert result.status == "unknown"


def test_threshold_comes_from_config() -> None:
    strict = ColourThresholds(max_weighted_delta_e=0.0, white_balance=False)
    result = check_colour(make_pair(garment(), control_light(garment()).image), strict)
    assert result.status == "fail" and result.threshold == 0.0
