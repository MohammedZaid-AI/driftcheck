from __future__ import annotations

import json
import zipfile
from pathlib import Path

from driftcheck.dataset import OutputMeta, parse_output_id
from driftcheck.ingest import ingest_inbox
from driftcheck.prompts import KEEP_PRODUCT, build_prompt
from driftcheck.register import export_jobs, make_jobs, register_outputs
from tests.helpers import FakeVisionClient, save_image


def _two_products(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "a_top.jpg", (200, 0, 0))
    save_image(tmp_path / "inbox" / "b_spice.jpg", (0, 150, 0))
    ingest_inbox(
        tmp_path,
        FakeVisionClient([
            {"category": "clothing", "product_type": "sleeveless top", "worn": False},
            {"category": "packaged_food", "product_type": "spice pack", "worn": False},
        ]),
    )


def _read_output(tmp_path: Path, product_id: str, oid: str) -> OutputMeta:
    path = tmp_path / "products" / product_id / "outputs" / f"{oid}.json"
    return OutputMeta.model_validate_json(path.read_text(encoding="utf-8"))


def test_prompts_fit_the_category() -> None:
    top = build_prompt("clothing", "sleeveless top", "model_shot")
    spice = build_prompt("packaged_food", "spice pack", "lifestyle")
    assert "worn by" in top and "sleeveless top" in top
    assert "table" in spice and "worn" not in spice
    assert KEEP_PRODUCT in top and KEEP_PRODUCT in spice


def test_output_id_rule() -> None:
    assert parse_output_id("studio__gemini_app") == ("studio", "gemini_app")
    assert parse_output_id("model_shot__qwen_edit_v2") == ("model_shot", "qwen_edit_v2")
    assert parse_output_id("studio") is None
    assert parse_output_id("closeup__gemini_app") is None
    assert parse_output_id("Studio__Gemini") is None


def test_two_styles_per_product_by_default(tmp_path: Path) -> None:
    _two_products(tmp_path)
    jobs = make_jobs(tmp_path, "gemini_app")
    assert [(job.product_id, job.style) for job in jobs] == [
        ("p001", "model_shot"), ("p001", "studio"),  # clothing goes on a model
        ("p002", "lifestyle"), ("p002", "studio"),  # a spice pack goes in a scene
    ]
    assert len(make_jobs(tmp_path, "gemini_app", all_styles=True)) == 6


def test_storefront_gets_poster_and_post() -> None:
    from driftcheck.dataset import styles_for

    assert styles_for("storefront") == ("festival_poster", "instagram_post")
    assert styles_for("storefront", all_styles=True) == ("festival_poster", "instagram_post", "reel_cover")


def test_checklist_has_one_line_per_image(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _two_products(data)
    out = export_jobs(data, "gemini_app")
    lines = (out / "manual_checklist.md").read_text(encoding="utf-8").splitlines()
    items = [line for line in lines if line[:1].isdigit()]
    assert len(items) == 4 and lines[0].endswith("4 images")
    first = items[0]
    assert first.startswith("1. [ ] Photo: `data")
    assert "original.jpg" in first and "model_shot__gemini_app.jpg" in first
    assert build_prompt("clothing", "sleeveless top", "model_shot") in first

    with zipfile.ZipFile(out / "kaggle_input.zip") as archive:
        assert len(json.loads(archive.read("jobs.json"))) == 4
        assert set(archive.namelist()) == {"jobs.json", "products/p001/original.jpg", "products/p002/original.jpg"}


def test_done_outputs_drop_out_of_jobs(tmp_path: Path) -> None:
    _two_products(tmp_path)
    save_image(tmp_path / "products" / "p001" / "outputs" / "studio__gemini_app.jpg")
    assert len(make_jobs(tmp_path, "gemini_app")) == 3
    assert len(make_jobs(tmp_path, "qwen_edit")) == 4


def test_register_manual_outputs(tmp_path: Path) -> None:
    _two_products(tmp_path)
    outputs = tmp_path / "products" / "p001" / "outputs"
    save_image(outputs / "studio__gemini_app.png")
    save_image(outputs / "my photo.jpg")

    result = register_outputs(tmp_path)
    assert result.registered == ["p001/studio__gemini_app"]
    assert result.bad_names == ["p001/my photo.jpg"]
    assert (outputs / "studio__gemini_app.jpg").exists()
    assert not (outputs / "studio__gemini_app.png").exists()
    record = _read_output(tmp_path, "p001", "studio__gemini_app")
    assert record.generator == "gemini_app"
    assert record.model == "Gemini app, free tier"
    assert record.prompt == build_prompt("clothing", "sleeveless top", "studio")

    again = register_outputs(tmp_path)
    assert again.registered == [] and again.already == ["p001/studio__gemini_app"]


def test_register_imports_kaggle_folder_and_keeps_its_json(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _two_products(data)
    kaggle = tmp_path / "kaggle_out" / "products" / "p002" / "outputs"
    save_image(kaggle / "lifestyle__qwen_edit.jpg")
    sidecar = {
        "product_id": "p002",
        "output_id": "lifestyle__qwen_edit",
        "style": "lifestyle",
        "prompt": "from the notebook",
        "generator": "qwen_edit",
        "model": "ovedrive/Qwen-Image-Edit-2511-4bit",
        "date": "2026-10-25",
    }
    (kaggle / "lifestyle__qwen_edit.json").write_text(json.dumps(sidecar))
    save_image(tmp_path / "kaggle_out" / "products" / "p999" / "outputs" / "studio__qwen_edit.jpg")

    result = register_outputs(data, source=tmp_path / "kaggle_out")
    assert result.imported == ["p002/lifestyle__qwen_edit"]
    assert result.already == ["p002/lifestyle__qwen_edit"]
    assert _read_output(data, "p002", "lifestyle__qwen_edit").model == "ovedrive/Qwen-Image-Edit-2511-4bit"
