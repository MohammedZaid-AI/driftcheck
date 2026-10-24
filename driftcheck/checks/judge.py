"""Shape and logo: describe each product the same way, then compare in code.

A first version asked the vision model to compare the two images directly. On
the shirt sample, qwen2.5vl:3b missed a sleeve change, called a colour shift a
change of every shape part, and gave 0.9 confidence to every answer. Small
models describe one image far more consistently than they compare two, so the
judge now describes the original and the output with the same prompt and fixed
choices, and the comparison happens here. The original's description is cached
and reused for every output of that product.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from driftcheck.checks.base import CheckResult, JudgeThresholds, Pair, unknown
from driftcheck.vision import VisionClient, VisionError

SHAPE = "shape"
LOGO = "logo"
NA = "not_applicable"

SLEEVES = ("sleeveless", "short", "elbow", "three_quarter", "full")
LENGTHS = ("cropped", "waist", "hip", "thigh", "knee", "midi", "ankle", "floor")
NECKLINES = ("round", "v_neck", "collar", "mandarin", "boat", "square", "high", "off_shoulder", "halter", "other")
SILHOUETTES = ("fitted", "straight", "a_line", "flared", "boxy", "wrap", "other")


class ShapeDescription(BaseModel):
    sleeve_length: Literal[SLEEVES + (NA,)]  # type: ignore[valid-type]
    neckline: Literal[NECKLINES + (NA,)]  # type: ignore[valid-type]
    length: Literal[LENGTHS + (NA,)] = Field(description="garment length on the body")  # type: ignore[valid-type]
    silhouette: Literal[SILHOUETTES + (NA,)]  # type: ignore[valid-type]
    logo_visible: bool = Field(description="true if any logo or brand mark is visible on the product")


DESCRIBE_PROMPT = """Describe the shape of the product in this photo for an Indian online store.
Only the product, not the hanger, background, mannequin or person wearing it.
Look carefully at the sleeves: how far down the arm do they reach?
Use not_applicable for parts the product does not have, for example sleeves on a spice pack."""


class PartChange(BaseModel):
    changed: bool | None  # None when either side is not applicable
    original: str
    output: str
    steps: int | None = None  # for ordered parts, how many steps apart
    reason: str


def _ordered(name: str, scale: tuple[str, ...], a: str, b: str, min_steps: int) -> PartChange:
    if NA in (a, b):
        return PartChange(changed=None, original=a, output=b, reason=f"{name} not applicable")
    steps = abs(scale.index(a) - scale.index(b))
    return PartChange(changed=steps >= min_steps, original=a, output=b, steps=steps,
                      reason=f"{name} {a} to {b}" if steps else f"{name} {a} in both")


def _categorical(name: str, a: str, b: str) -> PartChange:
    if NA in (a, b):
        return PartChange(changed=None, original=a, output=b, reason=f"{name} not applicable")
    return PartChange(changed=a != b, original=a, output=b, reason=f"{name} {a} to {b}" if a != b else f"{name} {a} in both")


def compare_shapes(original: ShapeDescription, output: ShapeDescription, t: JudgeThresholds) -> dict[str, PartChange]:
    parts = {
        "sleeves": _ordered("sleeves", SLEEVES, original.sleeve_length, output.sleeve_length, t.min_ordinal_steps),
        "length": _ordered("length", LENGTHS, original.length, output.length, t.min_ordinal_steps),
        "neckline": _categorical("neckline", original.neckline, output.neckline),
    }
    if t.use_silhouette:
        parts["silhouette"] = _categorical("silhouette", original.silhouette, output.silhouette)
    return parts


def _shape_result(parts: dict[str, PartChange], t: JudgeThresholds) -> CheckResult:
    details = {name: part.model_dump() for name, part in parts.items()}
    seen = [part for part in parts.values() if part.changed is not None]
    changed = [part for part in seen if part.changed]
    if not seen:
        return unknown(SHAPE, "No shape part applies to this product.", **details)
    if len(changed) >= t.min_changed_parts:
        return CheckResult(name=SHAPE, status="fail", score=len(changed), threshold=t.min_changed_parts,
                           reason="Shape changed: " + "; ".join(p.reason for p in changed) + ".", details=details)
    return CheckResult(name=SHAPE, status="pass", score=len(changed), threshold=t.min_changed_parts,
                       reason="Shape kept: " + "; ".join(p.reason for p in seen) + ".", details=details)


def _logo_result(pair: Pair, original: ShapeDescription, output: ShapeDescription) -> CheckResult:
    details = {"original_logo_visible": original.logo_visible, "output_logo_visible": output.logo_visible}
    card_logos = bool(pair.card.vision and pair.card.vision.logos)
    if not (card_logos or original.logo_visible):
        return unknown(LOGO, "No logo on the original, so there is nothing to compare.", **details)
    if not output.logo_visible:
        return CheckResult(name=LOGO, status="fail", score=1.0, threshold=1.0,
                           reason="The original has a logo but none is visible in the output.", details=details)
    return CheckResult(name=LOGO, status="pass", score=0.0, threshold=1.0,
                       reason="A logo is visible in both. Logo text is checked by the text check.", details=details)


def check_shape_and_logo(pair: Pair, t: JudgeThresholds, client: VisionClient) -> list[CheckResult]:
    if pair.original_crop is None or pair.output_crop is None:
        reason = "No product to compare in one of the images."
        return [unknown(SHAPE, reason), unknown(LOGO, reason)]
    try:
        original = client.ask(DESCRIBE_PROMPT, [pair.original_crop[0]], ShapeDescription)
        output = client.ask(DESCRIBE_PROMPT, [pair.output_crop[0]], ShapeDescription)
    except VisionError as error:
        reason = f"The vision judge failed: {error}"
        return [unknown(SHAPE, reason), unknown(LOGO, reason)]
    return [_shape_result(compare_shapes(original, output, t), t), _logo_result(pair, original, output)]
