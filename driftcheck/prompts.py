"""Generation prompts. Three styles, worded to fit the product category."""

from __future__ import annotations

from driftcheck.dataset import WEARABLE, Style

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


KEEP_SHOP = (
    "Keep the shop front exactly as it is in the photo: the shop name with every letter in its own script, "
    "the sign's colours, logo and layout. Do not translate, respell or shorten the shop name."
)

_STOREFRONT_STYLES: dict[Style, str] = {
    "festival_poster": "Make a Diwali festival offer poster featuring this shop, with the shop front clearly visible.",
    "instagram_post": "Make a square Instagram post featuring this shop front, bright and inviting, shop sign clearly visible.",
    "reel_cover": "Make a vertical Instagram Reel cover image with this shop front, shop sign clearly visible.",
}


def build_prompt(category: str | None, product_type: str | None, style: Style) -> str:
    """Prompt for one product and style. Clothes go on a model, packaged goods go on a table,
    shop fronts go into marketing posts where the shop name must stay exact."""
    thing = product_type or "product"
    if category == "storefront":
        return f"{_STOREFRONT_STYLES[style]} {KEEP_SHOP}"
    templates = _WEARABLE_STYLES if category in WEARABLE else _OBJECT_STYLES
    return f"{templates[style].format(thing=thing)} {KEEP_PRODUCT}"
