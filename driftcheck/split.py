"""data/split.json: calibration and test halves, split by product, never by output."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from driftcheck.dataset import list_product_ids, read_meta
from driftcheck.labels import write_json_atomic


def make_split(data_dir: Path, seed: int = 42) -> dict:
    """Half the products go to calibration and half to test. Each category is
    shuffled and dealt out in turn, so both halves get a similar mix."""
    by_category: dict[str, list[str]] = defaultdict(list)
    for product_id in list_product_ids(data_dir):
        by_category[read_meta(data_dir, product_id).category or "unknown"].append(product_id)

    rng = random.Random(seed)
    halves: dict[str, list[str]] = {"calibration": [], "test": []}
    turn = 0
    for category in sorted(by_category):
        products = sorted(by_category[category])
        rng.shuffle(products)
        for product_id in products:
            halves["calibration" if turn % 2 == 0 else "test"].append(product_id)
            turn += 1

    return {
        "seed": seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration": sorted(halves["calibration"]),
        "test": sorted(halves["test"]),
    }


def write_split(data_dir: Path, seed: int = 42, force: bool = False) -> dict:
    path = data_dir / "split.json"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Thresholds may be tuned on it. Use --force only if you will retune.")
    split = make_split(data_dir, seed)
    write_json_atomic(path, split)
    return split


def read_split(data_dir: Path) -> dict:
    return json.loads((data_dir / "split.json").read_text(encoding="utf-8"))
