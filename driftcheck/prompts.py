"""Generation prompts. Three styles, worded to fit the product category."""

from __future__ import annotations

from driftcheck.dataset import Style

WEARABLE = {"clothing", "footwear", "bag", "jewellery", "accessory"}

KEEP_PRODUCT = (
    "Keep the product exactly as it is in the photo: same shape, cut, sleeves, neckline, length, "
    "colours, print, embroidery, logo and every letter of text."
)

_WEARABLE_STYLES: dict[Style, str] = {
    "model_shot": "Show the {thing} worn by an Indian model in a natural standing pose, full product visible, soft daylight.",
    "studio": "Show the {thing} on a clean light grey studio background with soft even lighting, as a catalogue photo.",
    "lifestyle": "Show the {thing} worn by a person in an everyday Indian street or cafe scene, natural light.",
}

_OBJECT_STYLES: dict[Style, str] = {
    "model_shot": "Show a person holding the {thing} toward the camera, product in sharp focus, front label facing the camera.",
    "studio": "Show the {thing} on a clean white studio background with soft shadows, front label facing the camera.",
    "lifestyle": "Show the {thing} on a wooden table in a bright Indian home kitchen or living room, front label facing the camera.",
}


def build_prompt(category: str | None, product_type: str | None, style: Style) -> str:
    """Prompt for one product and style. Clothes go on a model, packaged goods go on a table."""
    thing = product_type or "product"
    templates = _WEARABLE_STYLES if category in WEARABLE else _OBJECT_STYLES
    return f"{templates[style].format(thing=thing)} {KEEP_PRODUCT}"
