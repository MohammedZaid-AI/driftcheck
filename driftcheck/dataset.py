"""Data layout under data/ and the json records stored next to images.

data/products/<product_id>/original.jpg
data/products/<product_id>/meta.json
data/products/<product_id>/outputs/<output_id>.jpg
data/products/<product_id>/outputs/<output_id>.json

An output_id is <style>__<generator>, for example studio__gemini_app.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Category = Literal[
    "clothing",
    "footwear",
    "bag",
    "jewellery",
    "accessory",
    "beauty",
    "packaged_food",
    "home",
    "electronics",
    "toy",
    "other",
]

Style = Literal["model_shot", "studio", "lifestyle"]
STYLES: tuple[Style, ...] = ("model_shot", "studio", "lifestyle")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
_OUTPUT_ID = re.compile(r"^(model_shot|studio|lifestyle)__([a-z0-9][a-z0-9_]*)$")
_PRODUCT_ID = re.compile(r"^p\d{3,}$")


class ProductMeta(BaseModel):
    product_id: str
    category: Category | None = None  # None until the vision model has filled it
    product_type: str | None = None
    worn: bool | None = None  # True if the original photo shows a person wearing the product
    notes: str = ""
    source_file: str
    sha256: str
    ingested_at: str
    category_by: str | None = None  # provider/model that filled the category


class OutputMeta(BaseModel):
    product_id: str
    output_id: str
    style: Style
    prompt: str
    generator: str
    model: str
    date: str


def products_dir(data_dir: Path) -> Path:
    return data_dir / "products"


def product_dir(data_dir: Path, product_id: str) -> Path:
    return products_dir(data_dir) / product_id


def output_id(style: str, generator: str) -> str:
    return f"{style}__{generator}"


def parse_output_id(value: str) -> tuple[Style, str] | None:
    """Split an output_id into style and generator, or None if it breaks the naming rule."""
    match = _OUTPUT_ID.match(value)
    if not match:
        return None
    return match.group(1), match.group(2)  # type: ignore[return-value]


def list_product_ids(data_dir: Path) -> list[str]:
    root = products_dir(data_dir)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and _PRODUCT_ID.match(p.name))


def next_product_id(data_dir: Path) -> str:
    numbers = [int(pid[1:]) for pid in list_product_ids(data_dir)]
    return f"p{max(numbers, default=0) + 1:03d}"


def list_outputs(data_dir: Path) -> list[tuple[str, str]]:
    """Every registered output as (product_id, output_id). Registered means it has its json."""
    found = []
    for product_id in list_product_ids(data_dir):
        outputs = product_dir(data_dir, product_id) / "outputs"
        if outputs.exists():
            found += [(product_id, p.stem) for p in sorted(outputs.glob("*.jpg")) if p.with_suffix(".json").exists()]
    return found


def output_image(data_dir: Path, product_id: str, output_id: str) -> Path:
    return product_dir(data_dir, product_id) / "outputs" / f"{output_id}.jpg"


def read_meta(data_dir: Path, product_id: str) -> ProductMeta:
    path = product_dir(data_dir, product_id) / "meta.json"
    return ProductMeta.model_validate_json(path.read_text(encoding="utf-8"))


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
