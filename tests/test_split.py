from __future__ import annotations

from pathlib import Path

import pytest

from driftcheck.dataset import ProductMeta, product_dir, write_json
from driftcheck.split import make_split, read_split, write_split


def _products(tmp_path: Path, categories: list[str]) -> Path:
    for i, category in enumerate(categories, 1):
        pid = f"p{i:03d}"
        write_json(product_dir(tmp_path, pid) / "meta.json", ProductMeta(
            product_id=pid, category=category, source_file="x.jpg", sha256=str(i), ingested_at="2026-10-24",
        ))
    return tmp_path


def test_split_is_by_product_half_and_half(tmp_path: Path) -> None:
    split = make_split(_products(tmp_path, ["clothing"] * 6 + ["packaged_food"] * 4))
    calibration, test = set(split["calibration"]), set(split["test"])
    assert calibration.isdisjoint(test)
    assert len(calibration) == 5 and len(test) == 5
    assert calibration | test == {f"p{i:03d}" for i in range(1, 11)}


def test_each_half_gets_each_category(tmp_path: Path) -> None:
    split = make_split(_products(tmp_path, ["clothing"] * 6 + ["packaged_food"] * 4))
    food = {"p007", "p008", "p009", "p010"}
    assert len(food & set(split["calibration"])) == 2
    assert len(food & set(split["test"])) == 2


def test_same_seed_same_split(tmp_path: Path) -> None:
    data = _products(tmp_path, ["clothing"] * 8)
    assert make_split(data, 7)["test"] == make_split(data, 7)["test"]


def test_existing_split_is_never_overwritten_without_force(tmp_path: Path) -> None:
    data = _products(tmp_path, ["clothing"] * 4)
    first = write_split(data)
    with pytest.raises(FileExistsError):
        write_split(data, seed=1)
    assert read_split(data)["test"] == first["test"]
    write_split(data, seed=1, force=True)
    assert read_split(data)["seed"] == 1
