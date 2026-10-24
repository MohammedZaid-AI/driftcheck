"""Colour: Lab palettes of both products, matched with CIEDE2000, weighted by colour share.

Each original colour is matched to its nearest output colour. Extra colours in
the output, such as a model's skin, do not count against it.
"""

from __future__ import annotations

import numpy as np

from driftcheck.checks.base import CheckResult, ColourThresholds, Pair, unknown
from driftcheck.palette import PaletteColour, grey_world, palette

NAME = "colour"


def match_palettes(original: list[PaletteColour], output: list[PaletteColour]) -> tuple[float, list[dict]]:
    """Share weighted mean of each original colour's distance to its nearest output colour."""
    from skimage.color import deltaE_ciede2000

    out_lab = np.array([c.lab for c in output])
    matches, total = [], 0.0
    for colour in original:
        distances = deltaE_ciede2000(np.tile(colour.lab, (len(out_lab), 1)), out_lab)
        best = int(np.argmin(distances))
        total += colour.share * float(distances[best])
        matches.append({
            "original": colour.hex, "share": colour.share,
            "output": output[best].hex, "delta_e": round(float(distances[best]), 2),
        })
    return total, matches


def check_colour(pair: Pair, t: ColourThresholds) -> CheckResult:
    original, output = pair.original, pair.output
    if t.white_balance:
        original, output = grey_world(original), grey_world(output)
    original_palette = palette(original, pair.original_mask)
    output_palette = palette(output, pair.output_mask)
    if not original_palette or not output_palette:
        return unknown(NAME, "No product pixels to compare.")

    score, matches = match_palettes(original_palette, output_palette)
    details = {
        "matches": matches,
        "output_palette": [c.model_dump() for c in output_palette],
        "white_balance": t.white_balance,
    }
    worst = max(matches, key=lambda m: m["share"] * m["delta_e"])
    if score > t.max_weighted_delta_e:
        reason = (f"Colours moved by {score:.1f} CIEDE2000 on average, above {t.max_weighted_delta_e:g}. "
                  f"Biggest change: {worst['original']} became {worst['output']}.")
        return CheckResult(name=NAME, status="fail", score=round(score, 2), threshold=t.max_weighted_delta_e,
                           reason=reason, details=details)
    return CheckResult(name=NAME, status="pass", score=round(score, 2), threshold=t.max_weighted_delta_e,
                       reason=f"Colours moved by {score:.1f} CIEDE2000 on average.", details=details)
