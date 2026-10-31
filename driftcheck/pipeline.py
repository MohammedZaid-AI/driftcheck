"""Runs every check on one original and output pair and builds the report."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel

from driftcheck.cache import DiskCache
from driftcheck.card import ProductCard, build_card
from driftcheck.checks.base import CheckResult, Pair, Thresholds, load_thresholds, unknown
from driftcheck.checks.colour import check_colour
from driftcheck.checks.judge import LOGO, SHAPE, check_shape_and_logo
from driftcheck.checks.pattern import check_pattern
from driftcheck.checks.segmentation import check_segmentation, product_visible
from driftcheck.checks.text import check_text
from driftcheck.checks.verdict import Verdict, overall_verdict
from driftcheck.config import Settings, load_settings
from driftcheck.ocr import READERS, OcrEngine
from driftcheck.segment import REMBG_MODEL, product_mask
from driftcheck.vision import VisionClient, get_vision_client

CHECK_ORDER = ("product_missing", "colour", "text", "pattern", SHAPE, LOGO)


class Report(BaseModel):
    original: str
    output: str
    verdict: Verdict
    checks: list[CheckResult]
    card: ProductCard
    seconds: float
    check_seconds: dict[str, float]
    vision_calls: int  # real calls only, cache hits are free
    models: dict[str, str]
    thresholds_tuned: bool
    text_scripts: list[str]  # scripts OCR read on both images, besides English
    created: str
    cutout: dict = {}  # which cut out the masks came from, see driftcheck/garment.py

    def check(self, name: str) -> CheckResult:
        return next(result for result in self.checks if result.name == name)


MAX_CHECK_SIDE = 2048  # same cap as originals at ingest; full size AI outputs ran a 12 GB laptop out of memory in OCR


def load_image(source: Path | str | Image.Image) -> Image.Image:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_CHECK_SIDE:
        image = image.copy()
        image.thumbnail((MAX_CHECK_SIDE, MAX_CHECK_SIDE), Image.Resampling.LANCZOS)
    return image


@dataclass
class Checker:
    thresholds: Thresholds
    client: VisionClient
    mask_fn: Callable[[Image.Image], np.ndarray]
    ocr_fn: Callable  # (image, scripts) -> text boxes in English plus those scripts
    features_fn: Callable[[Image.Image], np.ndarray]
    scripts_fn: Callable[[Image.Image], list[str]] = lambda image: ["en"]  # used when a product has no text_scripts
    card_cache: DiskCache | None = None
    pattern_model: str = ""
    clothes_parser: Callable | None = None  # garment cut out for worn clothing; None keeps rembg

    def masks(self, original: Image.Image, output: Image.Image, category: str | None) -> tuple[np.ndarray, np.ndarray, dict]:
        """rembg masks, or garment only masks for clothing and accessories."""
        from driftcheck.garment import GARMENT_CATEGORIES, garment_cutout

        original_mask, output_mask = self.mask_fn(original), self.mask_fn(output)
        if self.clothes_parser is None or category not in GARMENT_CATEGORIES:
            return original_mask, output_mask, {"method": "rembg", "reason": f"category {category}"}
        cut = garment_cutout(original, output, original_mask, output_mask, self.clothes_parser, category)
        return cut.original_mask, cut.output_mask, cut.info

    @classmethod
    def from_settings(cls, settings: Settings | None = None, vision_model: str | None = None) -> "Checker":
        from driftcheck.features import PatchFeatures
        from driftcheck.garment import ClothesParser

        settings = settings or load_settings()
        masks = DiskCache(settings.cache_dir, "masks")
        ocr = OcrEngine(DiskCache(settings.cache_dir, "ocr"))
        return cls(
            thresholds=load_thresholds(settings.config_dir),
            client=get_vision_client(settings, vision_model),
            mask_fn=lambda image: product_mask(image, masks),
            ocr_fn=ocr.read,
            scripts_fn=ocr.detect_scripts,
            features_fn=PatchFeatures(settings.vision.pattern_model, DiskCache(settings.cache_dir, "features")),
            card_cache=DiskCache(settings.cache_dir, "cards"),
            pattern_model=settings.vision.pattern_model,
            clothes_parser=ClothesParser(cache=DiskCache(settings.cache_dir, "clothes")),
        )

    def check(
        self,
        original: Path | str | Image.Image,
        output: Path | str | Image.Image,
        text_scripts: list[str] | None = None,
        category: str | None = None,
    ) -> Report:
        """Run every check. text_scripts and category come from the product's meta.json;
        without them the scripts are detected and the category is taken from the product card."""
        start = time.perf_counter()
        calls_before = len(self.client.calls)
        timings: dict[str, float] = {}

        def timed(name: str, run: Callable):
            began = time.perf_counter()
            result = run()
            timings[name] = round(time.perf_counter() - began, 3)
            return result

        original_image, output_image = load_image(original), load_image(output)
        scripts = text_scripts if text_scripts is not None else timed("scripts", lambda: self.scripts_fn(original_image))
        read_text = lambda image: self.ocr_fn(image, scripts)  # noqa: E731
        card = timed("card", lambda: build_card(original_image, self.client, self.mask_fn, read_text,
                                                self.card_cache, scripts))
        if category is None and card.vision is not None:
            category = card.vision.category
        original_mask, output_mask, cutout = timed("masks", lambda: self.masks(original_image, output_image, category))
        pair = Pair(original_image, output_image, original_mask, output_mask, card)
        t = self.thresholds
        segmentation = timed("product_missing", lambda: check_segmentation(pair, t.segmentation))
        results = [segmentation]
        if product_visible(segmentation):
            results.append(timed("colour", lambda: check_colour(pair, t.colour)))
            results.append(timed("text", lambda: check_text(pair, t.text, read_text, self.client)))
            results.append(timed("pattern", lambda: check_pattern(pair, t.pattern, self.features_fn)))
            results.extend(timed("shape_and_logo", lambda: check_shape_and_logo(pair, t.judge, self.client)))
        else:
            reason = f"Skipped because the product is not visible in the output. {segmentation.reason}"
            results.extend(unknown(name, reason) for name in CHECK_ORDER[1:])

        new_calls = self.client.calls[calls_before:]
        return Report(
            original=str(original) if not isinstance(original, Image.Image) else "upload",
            output=str(output) if not isinstance(output, Image.Image) else "upload",
            verdict=overall_verdict(results),
            checks=sorted(results, key=lambda r: CHECK_ORDER.index(r.name)),
            card=card,
            seconds=round(time.perf_counter() - start, 3),
            check_seconds=timings,
            vision_calls=sum(not call.cached for call in new_calls),
            models={
                "vision": f"{self.client.provider.name}/{self.client.model}",
                "pattern": self.pattern_model,
                "segmentation": f"rembg/{REMBG_MODEL}",
                "ocr": "easyocr/" + "+".join(READERS),
            },
            thresholds_tuned=t.tuned,
            text_scripts=scripts,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            cutout=cutout,
        )
