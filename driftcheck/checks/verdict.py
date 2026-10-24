"""Overall verdict from all checks. Unknown is its own outcome, never a pass."""

from __future__ import annotations

from typing import Literal

from driftcheck.checks.base import CheckResult

Verdict = Literal["drifted", "faithful", "unknown"]

# Which label attribute each check answers.
CHECK_ATTRIBUTE = {
    "product_missing": "product_missing",
    "colour": "colour_changed",
    "text": "text_changed",
    "pattern": "pattern_changed",
    "shape": "shape_changed",
    "logo": "logo_changed",
}


def overall_verdict(results: list[CheckResult]) -> Verdict:
    """Any fail means drifted. All pass means faithful. Anything else is unknown.
    Checks that do not apply, such as logo on a product without a logo, are left out."""
    statuses = [result.status for result in results if result.applicable]
    if "fail" in statuses:
        return "drifted"
    if statuses and all(status == "pass" for status in statuses):
        return "faithful"
    return "unknown"
