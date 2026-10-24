from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftcheck.config import PROJECT_ROOT
from driftcheck.dataset import parse_output_id, read_meta
from driftcheck.ingest import ingest_inbox
from driftcheck.prompts import KEEP_SHOP, build_prompt
from driftcheck.register import make_jobs
from driftcheck.selection import apply_selection, load_selection
from driftcheck.split import make_split
from tests.helpers import FakeVisionClient, save_image

TOP = {"category": "clothing", "product_type": "sleeveless top", "worn": True}
SHOP = {"category": "jewellery", "product_type": "jewels", "worn": False}


def _three_products(tmp_path: Path) -> Path:
    for i in range(3):
        save_image(tmp_path / "inbox" / f"{i}.jpg", (40 * i, 90, 160))
    ingest_inbox(tmp_path, FakeVisionClient([TOP, SHOP, TOP]))
    return tmp_path


SELECTION = {
    "overrides": {"p002": {"category": "storefront", "product_type": "jewellery shop sign"}, "p001": {"worn": False}},
    "core": {"p001": "sleeveless top", "p002": "shop sign with Devanagari name"},
}


def test_apply_sets_overrides_and_tags(tmp_path: Path) -> None:
    data = _three_products(tmp_path)
    core = apply_selection(data, SELECTION)
    assert core == {"clothing": 1, "storefront": 1}
    shop = read_meta(data, "p002")
    assert shop.category == "storefront" and shop.subset == "core"
    assert shop.manual_fields == ["category", "product_type"]
    assert read_meta(data, "p001").worn is False
    extra = read_meta(data, "p003")
    assert extra.subset == "extra" and extra.subset_reason == ""


def test_refresh_never_overwrites_hand_fixes(tmp_path: Path) -> None:
    data = _three_products(tmp_path)
    apply_selection(data, SELECTION)
    ingest_inbox(data, FakeVisionClient([TOP, SHOP, TOP]), refresh=True)
    assert read_meta(data, "p002").category == "storefront"
    assert read_meta(data, "p001").worn is False
    assert read_meta(data, "p003").worn is True  # not fixed by hand, so refreshed


def test_unknown_product_or_field_is_rejected(tmp_path: Path) -> None:
    data = _three_products(tmp_path)
    with pytest.raises(ValueError, match="p099"):
        apply_selection(data, {"overrides": {}, "core": {"p099": "x"}})
    with pytest.raises(ValueError, match="cannot be set"):
        apply_selection(data, {"overrides": {"p001": {"sha256": "x"}}, "core": {}})


def test_jobs_and_split_use_core_by_default(tmp_path: Path) -> None:
    data = _three_products(tmp_path)
    apply_selection(data, SELECTION)
    jobs = make_jobs(data, "qwen_edit")
    assert {job.product_id for job in jobs} == {"p001", "p002"}
    assert len(make_jobs(data, "qwen_edit", include_extra=True)) == 9
    split = make_split(data)
    assert sorted(split["calibration"] + split["test"]) == ["p001", "p002"]


def test_storefront_gets_its_own_styles_and_prompts(tmp_path: Path) -> None:
    data = _three_products(tmp_path)
    apply_selection(data, SELECTION)
    shop_jobs = [job for job in make_jobs(data, "gemini_app") if job.product_id == "p002"]
    assert [job.style for job in shop_jobs] == ["festival_poster", "instagram_post", "reel_cover"]
    assert all(KEEP_SHOP in job.prompt for job in shop_jobs)
    assert "Diwali" in build_prompt("storefront", "shop", "festival_poster")
    assert parse_output_id("reel_cover__gemini_app") == ("reel_cover", "gemini_app")


def test_duplicate_product_in_selection_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "selection.json").write_text('{"overrides": {"p001": {"worn": false}, "p001": {"text_scripts": ["hi"]}}, "core": {}}')
    with pytest.raises(ValueError, match="p001"):
        load_selection(tmp_path)


def test_text_scripts_can_be_set_by_hand(tmp_path: Path) -> None:
    from driftcheck.scripts import detect_product_scripts

    data = _three_products(tmp_path)
    apply_selection(data, {"overrides": {"p001": {"text_scripts": ["en", "hi"]}}, "core": {"p001": "x", "p002": "y"}})
    found = detect_product_scripts(data, lambda image: ["en", "bn"], refresh=True)
    assert found == {"p001": ["en", "hi"], "p002": ["en", "bn"]}


def test_project_selection_has_the_agreed_mix() -> None:
    selection = load_selection(PROJECT_ROOT / "config")
    core = selection["core"]
    assert len(core) == 40
    groups = {}
    for reason in core.values():
        group = reason.split(":")[0]
        groups[group] = groups.get(group, 0) + 1
    assert groups == {"clothing": 22, "packaged_food": 8, "beauty": 4, "accessory": 2, "storefront": 4}
    storefronts = [pid for pid, o in selection["overrides"].items() if o.get("category") == "storefront"]
    assert sorted(storefronts) == ["p004", "p029", "p055", "p062", "p068", "p074"]
    assert json.dumps(selection)  # stays plain JSON
