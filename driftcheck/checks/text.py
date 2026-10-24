"""Text: every text item on the original product must show up correctly in the output.

easyocr reads both images and rapidfuzz matches the normalised text. When an
item seems missing, the vision model reads the output as a backup. The model
only reads; the match is still done here.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from driftcheck.card import text_crops
from driftcheck.checks.base import CheckResult, Pair, TextThresholds, unknown
from driftcheck.ocr import TextBox, inside_mask, normalise
from driftcheck.vision import VisionClient, VisionError

NAME = "text"

READ_PROMPT = """Read every piece of text printed on this product, exactly as printed, letter by letter, in its own script.
The first image is the product. Later images are zoomed crops of text areas on it.
Do not translate. Do not guess text you cannot see. If there is no text, return an empty list."""


class VisionReading(BaseModel):
    visible_text: list[str] = Field(description="every piece of text on the product, exactly as printed")


def item_score(item: str, candidates: list[str]) -> tuple[float, str]:
    """Best fuzzy match of one normalised item against the output texts.
    Long items may also match inside the joined text, for words OCR split or merged."""
    best_score, best = 0.0, ""
    for candidate in candidates:
        score = fuzz.ratio(item, candidate)
        if score > best_score:
            best_score, best = score, candidate
    joined = " ".join(candidates)
    if len(item) >= 4 and joined:
        partial = fuzz.partial_ratio(item, joined)
        if partial > best_score:
            best_score, best = partial, joined
    return float(best_score), best


def check_text(
    pair: Pair,
    t: TextThresholds,
    ocr_fn: Callable,
    client: VisionClient | None = None,
) -> CheckResult:
    items = [o for o in inside_mask(pair.card.ocr, pair.original_mask) if len(normalise(o.text)) >= t.min_item_length]
    if not items:
        return unknown(NAME, "No text on the original product, so there is nothing to compare.", applicable=False)
    if pair.output_crop is None:
        return unknown(NAME, "No product in the output.")

    output_boxes: list[TextBox] = inside_mask(ocr_fn(pair.output), pair.output_mask)
    output_texts = [normalise(b.text) for b in output_boxes]
    rows = []
    for item in items:
        score, best = item_score(normalise(item.text), output_texts)
        rows.append({
            "text": item.text, "box": list(item.box), "score": score, "best_match": best,
            "found_by": "ocr" if score >= t.min_match else None,
        })

    details: dict = {"items": rows, "output_text": [b.text for b in output_boxes]}
    missing = [row for row in rows if row["found_by"] is None]
    if missing and t.vision_backup and client is not None:
        try:
            images = [pair.output_crop[0], *text_crops(pair.output, output_boxes)]
            reading = client.ask(READ_PROMPT, images, VisionReading)
            vision_texts = [normalise(text) for text in reading.visible_text if normalise(text)]
            details["vision_text"] = reading.visible_text
            for row in missing:
                score, best = item_score(normalise(row["text"]), vision_texts)
                if score >= t.min_match:
                    row.update(score=score, best_match=best, found_by="vision")
        except VisionError as error:
            details["vision_error"] = str(error)

    still_missing = [row["text"] for row in rows if row["found_by"] is None]
    score = min(row["score"] for row in rows) / 100
    threshold = t.min_match / 100
    if still_missing:
        reason = f"{len(still_missing)} of {len(rows)} text items not found or changed: " + ", ".join(still_missing)
        return CheckResult(name=NAME, status="fail", score=score, threshold=threshold, reason=reason, details=details)
    return CheckResult(name=NAME, status="pass", score=score, threshold=threshold,
                       reason=f"All {len(rows)} text items found.", details=details)
