"""Applies config/selection.json: hand fixes to meta.json and the core and extra tags."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from driftcheck.dataset import ProductMeta, list_product_ids, product_dir, read_meta, write_json

ALLOWED_OVERRIDES = {"category", "product_type", "worn"}


def load_selection(config_dir: Path) -> dict:
    return json.loads((config_dir / "selection.json").read_text(encoding="utf-8"))


def apply_selection(data_dir: Path, selection: dict) -> Counter:
    """Set the hand fixed fields, then tag every product core or extra. Returns core counts per category."""
    known = set(list_product_ids(data_dir))
    missing = (set(selection["overrides"]) | set(selection["core"])) - known
    if missing:
        raise ValueError(f"selection.json names products that are not in data/products: {sorted(missing)}")

    for product_id in sorted(known):
        meta = read_meta(data_dir, product_id)
        for field, value in selection["overrides"].get(product_id, {}).items():
            if field not in ALLOWED_OVERRIDES:
                raise ValueError(f"{product_id}: {field} cannot be set by hand")
            setattr(meta, field, value)
            if field not in meta.manual_fields:
                meta.manual_fields.append(field)
        reason = selection["core"].get(product_id)
        meta.subset = "core" if reason else "extra"
        meta.subset_reason = reason or ""
        ProductMeta.model_validate(meta.model_dump())  # catches a bad category name
        write_json(product_dir(data_dir, product_id) / "meta.json", meta)

    return Counter(read_meta(data_dir, pid).category for pid in selection["core"])
