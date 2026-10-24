"""Is the product still there? Compares the rembg product area in both images."""

from __future__ import annotations

from driftcheck.checks.base import CheckResult, Pair, SegmentationThresholds, unknown

NAME = "product_missing"


def check_segmentation(pair: Pair, t: SegmentationThresholds) -> CheckResult:
    original_share = float(pair.original_mask.mean())
    output_share = float(pair.output_mask.mean())
    details = {"original_share": round(original_share, 4), "output_share": round(output_share, 4)}

    if original_share < t.missing_share:
        return unknown(NAME, "No product found in the original, so there is nothing to compare.", **details)
    if output_share < t.missing_share:
        return CheckResult(
            name=NAME, status="fail", score=output_share, threshold=t.missing_share,
            reason=f"The product covers {output_share:.1%} of the output, below {t.missing_share:.0%}. It is missing.",
            details=details,
        )
    if output_share < t.small_share:
        return CheckResult(
            name=NAME, status="unknown", score=output_share, threshold=t.small_share,
            reason=f"The product covers only {output_share:.1%} of the output. Too small to check reliably.",
            details=details,
        )
    return CheckResult(
        name=NAME, status="pass", score=output_share, threshold=t.missing_share,
        reason=f"The product covers {output_share:.1%} of the output.", details=details,
    )


def product_visible(result: CheckResult) -> bool:
    """Other checks only run when this check passed."""
    return result.status == "pass"
