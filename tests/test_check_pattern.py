from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from driftcheck.cache import DiskCache
from driftcheck.checks.pattern import check_pattern
from driftcheck.features import GRID, PATCH, SIDE, PatchFeatures, patch_mask, to_square
from driftcheck.synth import control_shift
from tests.checks_helpers import make_pair, thresholds
from tests.synthetic import garment

T = thresholds().pattern


def fake_features(image: Image.Image) -> np.ndarray:
    """Stands in for DINOv2: per patch histogram over 8 coarse colours, as unit vectors."""
    arr = np.asarray(image.convert("RGB")) > 127
    codes = arr[..., 0] * 4 + arr[..., 1] * 2 + arr[..., 2]
    patches = codes.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3).reshape(GRID, GRID, -1)
    feats = np.stack([(patches == code).mean(axis=-1) for code in range(8)], axis=-1)
    return feats / np.maximum(np.linalg.norm(feats, axis=-1, keepdims=True), 1e-9)


def dotted() -> Image.Image:
    """A green top with yellow dots: a different motif in different colours."""
    img = garment(colour=(30, 160, 60))
    draw = ImageDraw.Draw(img)
    for y in range(55, 175, 10):
        for x in range(45, 115, 10):
            draw.ellipse((x, y, x + 5, y + 5), fill=(250, 230, 40))
    return img


def test_pass_for_same_pattern() -> None:
    result = check_pattern(make_pair(garment(stripes=True), garment(stripes=True)), T, fake_features)
    assert result.status == "pass" and result.score == 0


def test_pass_when_reframed() -> None:
    original = garment(stripes=True)
    moved = control_shift(original, random.Random(0)).image
    assert check_pattern(make_pair(original, moved), T, fake_features).status == "pass"


def test_fail_when_pattern_replaced() -> None:
    result = check_pattern(make_pair(garment(stripes=True), dotted()), T, fake_features)
    assert result.status == "fail" and result.score > T.max_drift_share
    grid = result.details["similarity_grid"]
    assert len(grid) == GRID and any(v is not None and v < T.patch_similarity for row in grid for v in row)


def test_unknown_when_product_too_small() -> None:
    tiny = np.zeros((200, 160), bool)
    tiny[90:96, 70:76] = True
    result = check_pattern(make_pair(garment(), garment(), output_mask=tiny), T, fake_features)
    assert result.status == "unknown"


def test_to_square_keeps_aspect_and_mask() -> None:
    mask = np.ones((100, 50), bool)
    image, square_mask, side, offset = to_square(Image.new("RGB", (50, 100), "red"), mask)
    assert image.size == (SIDE, SIDE) and side == 100 and offset == (25, 0)
    patches = patch_mask(square_mask)
    assert patches.shape == (GRID, GRID)
    assert patches[:, GRID // 2].all() and not patches[:, 0].any()


def test_patch_features_are_cached(tmp_path: Path) -> None:
    calls = []

    class FakeModel:
        def __call__(self, pixel_values):
            calls.append(1)
            import torch

            out = torch.ones((1, 1 + GRID * GRID, 8))
            return type("Out", (), {"last_hidden_state": out})()

    features = PatchFeatures("fake/dino", DiskCache(tmp_path, "features"))
    features._model = FakeModel()
    first = features(Image.new("RGB", (SIDE, SIDE), "red"))
    second = features(Image.new("RGB", (SIDE, SIDE), "red"))
    assert first.shape == (GRID, GRID, 8) and np.allclose(np.linalg.norm(first, axis=-1), 1)
    assert np.array_equal(first, second) and len(calls) == 1
