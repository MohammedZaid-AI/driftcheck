"""The two simple baselines DriftCheck is compared with.

Baseline 1: whole image DINOv2 similarity with one threshold.
Baseline 2: one vision call that asks whether the product changed, yes or no.
Neither says which attribute changed, so they are compared on overall drift only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from driftcheck.features import to_square
from driftcheck.vision import VisionClient, VisionError

B2_PROMPT = """Image 1 is the original product photo. Image 2 is an AI generated photo that must show the same product.
Did the product itself change? Count changes to its shape, colour, print or pattern, logo or text.
Ignore pose, camera angle, background, lighting, the model person and fabric folds."""


class ChangedAnswer(BaseModel):
    changed: bool = Field(description="true if the product itself changed")
    reason: str = Field(description="one short sentence")


def whole_image_similarity(original: Image.Image, output: Image.Image,
                           features_fn: Callable[[Image.Image], np.ndarray]) -> float:
    """Cosine similarity of the mean DINOv2 patch features of the two whole images."""
    vectors = []
    for image in (original, output):
        image = image.convert("RGB")
        square, _, _, _ = to_square(image, np.ones((image.height, image.width), bool))
        features = features_fn(square)
        mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
        vectors.append(mean / np.linalg.norm(mean))
    return float(vectors[0] @ vectors[1])


def ask_changed(original: Image.Image, output: Image.Image, client: VisionClient) -> tuple[bool | None, str]:
    """True or False from the vision model, or None with the error when the call fails."""
    try:
        answer = client.ask(B2_PROMPT, [original, output], ChangedAnswer)
    except VisionError as error:
        return None, str(error)
    return answer.changed, answer.reason
