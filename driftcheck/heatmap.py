"""heatmap.png: where the output drifted from the original.

Top left: the original, with each original text item boxed green if found and
red if missing. Top right: the output, with patches whose pattern has no close
match in the original painted red. Bottom: each original colour next to the
output colour it matched, with the CIEDE2000 distance. The two photos are not
aligned, so a pixel by pixel difference would not mean anything.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from driftcheck.features import GRID
from driftcheck.pipeline import Report

PANEL_HEIGHT = 512
RED = (220, 30, 30)
GREEN = (30, 160, 60)
STATUS_COLOUR = {"pass": GREEN, "fail": RED, "unknown": (150, 150, 150)}


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _scale(image: Image.Image) -> tuple[Image.Image, float]:
    scale = PANEL_HEIGHT / image.height
    return image.resize((max(1, round(image.width * scale)), PANEL_HEIGHT), Image.Resampling.LANCZOS), scale


def draw_text_boxes(original: Image.Image, report: Report, scale: float) -> Image.Image:
    text = report.check("text")
    out = original.copy()
    draw = ImageDraw.Draw(out)
    for item in text.details.get("items", []):
        x0, y0, x1, y1 = (round(v * scale) for v in item["box"])
        colour = GREEN if item["found_by"] else RED
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=colour, width=3)
    return out


def draw_pattern(output: Image.Image, report: Report, scale: float) -> Image.Image:
    pattern = report.check("pattern")
    grid = pattern.details.get("similarity_grid")
    if not grid:
        return output
    overlay = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, _, _ = pattern.details["crop_box"]
    ox, oy = pattern.details["square_offset"]
    cell = pattern.details["square_side"] / GRID
    threshold = pattern.details["patch_similarity"]
    for row in range(GRID):
        for col in range(GRID):
            similarity = grid[row][col]
            if similarity is None or similarity >= threshold:
                continue
            strength = min(1.0, (threshold - similarity) / 0.2)
            left = (x0 + col * cell - ox) * scale
            top = (y0 + row * cell - oy) * scale
            draw.rectangle((left, top, left + cell * scale, top + cell * scale), fill=(*RED, int(80 + 120 * strength)))
    return Image.alpha_composite(output.convert("RGBA"), overlay).convert("RGB")


def draw_colour_strip(report: Report, width: int) -> Image.Image:
    colour = report.check("colour")
    matches = colour.details.get("matches", [])
    strip = Image.new("RGB", (width, 44 + 34 * max(1, len(matches))), "white")
    draw = ImageDraw.Draw(strip)
    draw.text((10, 10), f"Colour: original -> output, CIEDE2000. Limit {colour.threshold}", fill="black", font=_font(18))
    if not matches:
        draw.text((10, 44), colour.reason, fill="black", font=_font(16))
    for i, match in enumerate(matches):
        y = 44 + 34 * i
        draw.rectangle((10, y, 70, y + 26), fill=match["original"], outline="black")
        draw.rectangle((80, y, 140, y + 26), fill=match["output"], outline="black")
        delta = match["delta_e"]
        flag = RED if colour.threshold is not None and delta > colour.threshold else (60, 60, 60)
        draw.text((155, y + 4), f"{match['share']:.0%} of product   delta E {delta:.1f}", fill=flag, font=_font(16))
    return strip


def draw_header(report: Report, width: int) -> Image.Image:
    header = Image.new("RGB", (width, 70), "white")
    draw = ImageDraw.Draw(header)
    draw.text((10, 8), f"Verdict: {report.verdict}", fill=STATUS_COLOUR.get(
        {"drifted": "fail", "faithful": "pass"}.get(report.verdict, "unknown")), font=_font(24))
    x = 10
    for check in report.checks:
        label = f"{check.name}: {check.status}" + ("" if check.applicable else " (n/a)")
        draw.text((x, 42), label, fill=STATUS_COLOUR[check.status], font=_font(16))
        x += int(draw.textlength(label, font=_font(16))) + 24
    return header


def draw_heatmap(original: Image.Image, output: Image.Image, report: Report) -> Image.Image:
    original_panel, original_scale = _scale(original.convert("RGB"))
    output_panel, output_scale = _scale(output.convert("RGB"))
    original_panel = draw_text_boxes(original_panel, report, original_scale)
    output_panel = draw_pattern(output_panel, report, output_scale)

    width = max(original_panel.width + output_panel.width + 30, 900)
    header = draw_header(report, width)
    strip = draw_colour_strip(report, width)
    canvas = Image.new("RGB", (width, header.height + PANEL_HEIGHT + 30 + strip.height), "white")
    canvas.paste(header, (0, 0))
    top = header.height + 24
    ImageDraw.Draw(canvas).text((10, header.height + 2), "Original", fill="black", font=_font(16))
    ImageDraw.Draw(canvas).text((original_panel.width + 20, header.height + 2), "Output", fill="black", font=_font(16))
    canvas.paste(original_panel, (10, top))
    canvas.paste(output_panel, (original_panel.width + 20, top))
    canvas.paste(strip, (0, top + PANEL_HEIGHT + 6))
    return canvas


def drift_pixels(image: Image.Image) -> int:
    """Count strongly red pixels. Used by tests to see that drift was drawn."""
    arr = np.asarray(image.convert("RGB")).astype(int)
    return int(((arr[..., 0] > 180) & (arr[..., 1] < 90) & (arr[..., 2] < 90)).sum())
