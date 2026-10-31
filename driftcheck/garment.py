"""Garment only cutout for worn clothing.

rembg cuts out the whole foreground. In a model shot that is the whole person, so
colour and pattern compared the garment with the model's face, skin and other
clothes. A clothes parsing model labels every pixel as a garment class or a body
part. For a clothing product we keep the garment classes found on the original
product and drop face, hair, arms and legs. Images without a person, such as flat
lays and studio shots, keep the rembg mask, which is already tight there.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from PIL import Image

from driftcheck.cache import DiskCache, make_key

PARSER_MODEL = "mattmdjaga/segformer_b2_clothes"
GARMENT_CATEGORIES = {"clothing", "accessory"}
PERSON_LABELS = {"Hair", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm"}
# The parser often calls a flat top "Dress" but the same top on a model "Upper-clothes",
# so classes are grouped. A product is the biggest group on the original: a shirt worn
# with trousers is the shirt.
GARMENT_GROUPS = {
    "top or dress": {"Upper-clothes", "Dress"},
    "bottom": {"Pants", "Skirt"},
    "hat": {"Hat"},
    "shoes": {"Left-shoe", "Right-shoe"},
    "bag": {"Bag"},
    "belt": {"Belt"},
    "scarf": {"Scarf"},
}
# Accessories in the core set are scarves. On a flat lay or mannequin the parser does not
# see them as scarves, so on a model only accessory classes are kept.
ACCESSORY_CLASSES = {"Scarf", "Hat", "Bag", "Belt"}
PERSON_MIN_SHARE = 0.02  # this much of the image as face, hair, arms or legs means a person is in it
MIN_GARMENT_SHARE = 0.005  # a garment cut out smaller than this share of the image is not trusted


class ClothesParser:
    """Callable: image to a label map of class names per pixel, cached on disk."""

    def __init__(self, model_name: str = PARSER_MODEL, cache: DiskCache | None = None) -> None:
        self.model_name = model_name
        self.cache = cache

    @cached_property
    def _model(self):
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        torch.set_grad_enabled(False)
        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(self.model_name).eval()
        return processor, model

    @cached_property
    def labels(self) -> dict[int, str]:
        return {int(k): v for k, v in self._model[1].config.id2label.items()}

    def __call__(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        png = io.BytesIO()
        image.save(png, format="PNG")
        key = make_key("clothes_parse", self.model_name, png.getvalue())
        if self.cache and (stored := self.cache.get_bytes(key)) is not None:
            return np.load(io.BytesIO(stored))

        import torch

        processor, model = self._model
        logits = model(**processor(images=image, return_tensors="pt")).logits
        upsampled = torch.nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        label_map = upsampled.argmax(dim=1)[0].numpy().astype(np.uint8)
        if self.cache:
            out = io.BytesIO()
            np.save(out, label_map)
            self.cache.set_bytes(key, out.getvalue())
        return label_map


def _ids(labels: dict[int, str], names: set[str]) -> list[int]:
    return [i for i, name in labels.items() if name in names]


def has_person(label_map: np.ndarray, labels: dict[int, str]) -> bool:
    return bool(np.isin(label_map, _ids(labels, PERSON_LABELS)).mean() >= PERSON_MIN_SHARE)


def product_classes(label_map: np.ndarray, product_mask: np.ndarray, labels: dict[int, str],
                    category: str | None = None) -> list[str]:
    """Garment classes to keep: the biggest garment group inside the original product mask,
    or the accessory classes for an accessory."""
    if category == "accessory":
        return sorted(ACCESSORY_CLASSES)
    sizes = {group: int((np.isin(label_map, _ids(labels, names)) & product_mask).sum())
             for group, names in GARMENT_GROUPS.items()}
    biggest = max(sizes, key=sizes.get)
    return sorted(GARMENT_GROUPS[biggest]) if sizes[biggest] else []


@dataclass
class Cutout:
    original_mask: np.ndarray
    output_mask: np.ndarray
    info: dict = field(default_factory=dict)


def garment_cutout(
    original: Image.Image,
    output: Image.Image,
    original_rembg: np.ndarray,
    output_rembg: np.ndarray,
    parser: ClothesParser,
    category: str | None = None,
) -> Cutout:
    """Garment masks for both images. An image with a person in it gets only the product's
    garment classes. An image without a person keeps its rembg mask. If the original's garment
    classes cannot be found, both keep rembg and the info says why."""
    labels = parser.labels
    original_map, output_map = parser(original), parser(output)
    keep = product_classes(original_map, original_rembg, labels, category)
    info = {"method": "rembg", "garment_classes": keep,
            "original_has_person": has_person(original_map, labels),
            "output_has_person": has_person(output_map, labels)}
    if not keep:
        info["reason"] = "no garment class found on the original product"
        return Cutout(original_rembg, output_rembg, info)

    keep_ids = _ids(labels, set(keep))
    masks = []
    for name, label_map, rembg_mask in (("original", original_map, original_rembg), ("output", output_map, output_rembg)):
        if info[f"{name}_has_person"]:
            garment = np.isin(label_map, keep_ids)
            if garment.mean() >= MIN_GARMENT_SHARE:
                masks.append(garment)
                info[f"{name}_cut"] = "garment classes"
                continue
            info[f"{name}_cut"] = "rembg, garment too small"
        else:
            info[f"{name}_cut"] = "rembg, no person"
        masks.append(rembg_mask)
    info["method"] = "clothes parsing" if "garment classes" in (info["original_cut"], info["output_cut"]) else "rembg"
    return Cutout(masks[0], masks[1], info)
