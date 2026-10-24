"""What every check returns, the thresholds it reads, and the image pair it looks at."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from driftcheck.card import ProductCard
from driftcheck.segment import Box, crop_to_product

Status = Literal["pass", "fail", "unknown"]


class CheckResult(BaseModel):
    name: str
    status: Status
    score: float | None = None
    threshold: float | None = None
    reason: str
    details: dict[str, Any] = {}


def unknown(name: str, reason: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, status="unknown", reason=reason, details=details)


class SegmentationThresholds(BaseModel):
    missing_share: float
    small_share: float


class ColourThresholds(BaseModel):
    max_weighted_delta_e: float
    white_balance: bool


class TextThresholds(BaseModel):
    min_match: float
    min_item_length: int
    vision_backup: bool


class PatternThresholds(BaseModel):
    patch_similarity: float
    max_drift_share: float
    min_patches: int


class JudgeThresholds(BaseModel):
    min_confidence: float


class Thresholds(BaseModel):
    tuned: bool
    segmentation: SegmentationThresholds
    colour: ColourThresholds
    text: TextThresholds
    pattern: PatternThresholds
    judge: JudgeThresholds


def load_thresholds(config_dir: Path) -> Thresholds:
    return Thresholds.model_validate(json.loads((config_dir / "thresholds.json").read_text(encoding="utf-8")))


@dataclass
class Pair:
    """The original, the output, their product masks and the product card."""

    original: Image.Image
    output: Image.Image
    original_mask: np.ndarray
    output_mask: np.ndarray
    card: ProductCard

    @cached_property
    def original_crop(self) -> tuple[Image.Image, np.ndarray, Box] | None:
        return crop_to_product(self.original, self.original_mask)

    @cached_property
    def output_crop(self) -> tuple[Image.Image, np.ndarray, Box] | None:
        return crop_to_product(self.output, self.output_mask)
