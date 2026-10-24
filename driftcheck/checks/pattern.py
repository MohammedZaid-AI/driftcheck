"""Pattern: DINOv2 patch features inside the product masks.

For each output patch, take the best cosine similarity to any original patch.
Low similarity means drift. The score is the share of output product patches
below the patch threshold. No alignment is needed, so pose changes are fine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image

from driftcheck.checks.base import CheckResult, Pair, PatternThresholds, unknown
from driftcheck.features import PATCH, patch_mask, to_square

NAME = "pattern"


def check_pattern(pair: Pair, t: PatternThresholds, features_fn: Callable[[Image.Image], np.ndarray]) -> CheckResult:
    if pair.original_crop is None or pair.output_crop is None:
        return unknown(NAME, "No product to compare in one of the images.")
    # Crops are scaled to a fixed size, so judge size by real pixels, not by patches after scaling.
    min_pixels = t.min_patches * PATCH * PATCH
    if pair.original_mask.sum() < min_pixels or pair.output_mask.sum() < min_pixels:
        return unknown(NAME, f"The product has fewer than {min_pixels} pixels. Too small to compare patterns.")

    original_image, original_mask, _, _ = to_square(pair.original_crop[0], pair.original_crop[1])
    output_image, output_mask, side, offset = to_square(pair.output_crop[0], pair.output_crop[1])
    original_patches, output_patches = patch_mask(original_mask), patch_mask(output_mask)
    if original_patches.sum() < t.min_patches or output_patches.sum() < t.min_patches:
        return unknown(NAME, f"Fewer than {t.min_patches} product patches. The product is too small to compare patterns.")

    original_features = features_fn(original_image)[original_patches]
    output_features = features_fn(output_image)[output_patches]
    similarity = output_features @ original_features.T
    best = similarity.max(axis=1)
    drift_share = float((best < t.patch_similarity).mean())
    # The other direction: original patches with no match in the output. Reported, not scored.
    missing_share = float((similarity.max(axis=0) < t.patch_similarity).mean())

    grid = np.full(output_patches.shape, np.nan)
    grid[output_patches] = best
    details = {
        "drift_share": round(drift_share, 4),
        "missing_share": round(missing_share, 4),
        "median_similarity": round(float(np.median(best)), 4),
        "patch_similarity": t.patch_similarity,
        "similarity_grid": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in grid],
        "crop_box": list(pair.output_crop[2]),
        "square_side": side,
        "square_offset": list(offset),
    }
    if drift_share > t.max_drift_share:
        reason = f"{drift_share:.0%} of the output product has no close match in the original pattern, above {t.max_drift_share:.0%}."
        return CheckResult(name=NAME, status="fail", score=round(drift_share, 4), threshold=t.max_drift_share,
                           reason=reason, details=details)
    return CheckResult(name=NAME, status="pass", score=round(drift_share, 4), threshold=t.max_drift_share,
                       reason=f"{drift_share:.0%} of the output product differs from the original pattern.", details=details)
