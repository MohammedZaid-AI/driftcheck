"""Detects the text scripts on each product once and saves them as text_scripts in meta.json."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from driftcheck.dataset import product_dir, read_meta, selected_product_ids, write_json


def detect_product_scripts(
    data_dir: Path,
    detect: Callable[[Image.Image], list[str]],
    include_extra: bool = False,
    refresh: bool = False,
) -> dict[str, list[str]]:
    """Fill text_scripts for every selected product that has none yet, or all with refresh.
    Scripts set by hand in config/selection.json are never changed. Returns product id to scripts."""
    found = {}
    for product_id in selected_product_ids(data_dir, include_extra):
        meta = read_meta(data_dir, product_id)
        if "text_scripts" not in meta.manual_fields and (meta.text_scripts is None or refresh):
            original = Image.open(product_dir(data_dir, product_id) / "original.jpg")
            meta.text_scripts = detect(original)
            write_json(product_dir(data_dir, product_id) / "meta.json", meta)
        found[product_id] = meta.text_scripts or []
    return found
