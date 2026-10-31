from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftcheck.eval_report import build_report, render_markdown
from driftcheck.evaluate import (
    Item,
    baseline_scores,
    collect_items,
    decide_all,
    decide_check,
    driftcheck_scores,
    overall_label,
    recording_thresholds,
    run_all,
    score,
    tune,
    tune_baseline_1,
    verdict,
)
from driftcheck.labels import ATTRIBUTES
from tests.checks_helpers import thresholds

T = thresholds()
CLEAN = {name: False for name in ATTRIBUTES} | {"logo_changed": None}


def raw(colour: float = 2.0, grid_value: float = 0.98, output_share: float = 0.4, shape_steps: int = 0,
        text_score: float | None = None, b1: float = 0.95, b2: bool | None = False, seconds: float = 10.0) -> dict:
    """A saved run with chosen raw scores, as eval-run would store it."""
    text = ({"name": "text", "status": "pass", "score": None, "reason": "", "applicable": True,
             "details": {"items": [{"text": "SALE", "box": [0, 0, 1, 1], "score": text_score, "best_match": "", "found_by": "ocr"}]}}
            if text_score is not None else
            {"name": "text", "status": "unknown", "score": None, "reason": "no text", "applicable": False, "details": {}})
    checks = [
        {"name": "product_missing", "status": "pass", "score": output_share, "reason": "", "applicable": True,
         "details": {"original_share": 0.4, "output_share": output_share}},
        {"name": "colour", "status": "pass", "score": colour, "reason": "", "applicable": True, "details": {}},
        text,
        {"name": "pattern", "status": "pass", "score": 0, "reason": "", "applicable": True,
         "details": {"similarity_grid": [[grid_value] * 10 for _ in range(10)]}},
        {"name": "shape", "status": "pass", "score": 0, "reason": "", "applicable": True,
         "details": {"sleeves": {"changed": shape_steps > 0, "original": "full", "output": "short", "steps": shape_steps, "reason": ""},
                     "silhouette": {"changed": False, "original": "fitted", "output": "fitted", "steps": None, "reason": ""}}},
        {"name": "logo", "status": "unknown", "score": None, "reason": "no logo", "applicable": False, "details": {}},
    ]
    return {"report": {"checks": checks, "verdict": "faithful", "seconds": seconds, "vision_calls": 2},
            "b1": {"similarity": b1, "seconds": 0.5}, "b2": {"changed": b2, "reason": "", "seconds": 3.0, "vision_calls": 1}}


def item(key: str, split: str = "calibration", set_name: str = "synthetic", **changed) -> Item:
    label = dict(CLEAN) | changed
    return Item(set_name, split, key, "x", Path("o.jpg"), Path("u.jpg"), label, None)


def test_score_rates_and_not_measurable() -> None:
    s = score([(True, "flag"), (True, "unknown"), (False, "clear"), (False, "flag"), (None, "flag")])
    assert (s["tp"], s["fn"], s["unknown_positive"], s["fp"], s["tn"]) == (1, 0, 1, 1, 1)
    assert s["recall"] == 0.5  # unknown on a drifted image counts as missed
    assert s["wrong_flag_rate"] == 0.5 and s["n"] == 4
    only_good = score([(False, "clear"), (False, "clear")])
    assert only_good["recall"] is None and only_good["precision"] is None and only_good["wrong_flag_rate"] == 0


def test_decisions_follow_thresholds() -> None:
    r = raw(colour=9.0, grid_value=0.80, shape_steps=2)["report"]
    assert decide_check(r, "colour", T)[0] == "fail"
    loose = T.model_copy(deep=True)
    loose.colour.max_weighted_delta_e = 10
    assert decide_check(r, "colour", loose)[0] == "pass"
    assert decide_check(r, "pattern", T)[0] == "fail"  # 0.80 is under the 0.85 patch threshold everywhere
    assert decide_check(r, "shape", T)[0] == "fail"  # 2 steps meets min_ordinal_steps 2
    assert decide_check(r, "text", T) == ("unknown", False)
    assert verdict(decide_all(r, T)) == "drifted"


def test_missing_product_makes_other_checks_unknown() -> None:
    r = raw(output_share=0.0005)["report"]
    decisions = decide_all(r, T)
    assert decisions["product_missing"][0] == "fail"
    assert decisions["colour"] == ("unknown", True)


def test_recording_thresholds_keep_details() -> None:
    t = recording_thresholds(T)
    assert t.text.min_match == 95 and t.judge.use_silhouette is True and t.segmentation.missing_share < 0.01
    assert T.text.min_match != 95  # the original is not changed


def test_tune_picks_best_recall_within_wrong_flag_cap() -> None:
    items = [item(f"g{i}") for i in range(20)] + [item(f"d{i}", colour_changed=True) for i in range(10)]
    raws = {i.key: raw(colour=1.0 + 0.2 * n) for n, i in enumerate(items[:20])}  # good: 1.0 to 4.8
    raws |= {i.key: raw(colour=4.0 + n) for n, i in enumerate(items[20:])}  # drifted: 4.0 to 13.0
    tuned, notes = tune(items, raws, T)
    assert tuned.tuned is True and notes["colour"]["tuned"] is True
    assert notes["colour"]["calibration"]["wrong_flag_rate"] <= 0.10
    # 4.4 and 4.6 both catch 9 of 10 drifted images. 4.4 flags 2 good images, 4.6 flags 1, so 4.6 wins.
    assert tuned.colour.max_weighted_delta_e == 4.6
    assert notes["colour"]["calibration"]["recall"] == 0.9
    assert notes["shape"]["tuned"] is False and "no shape_changed positives" in notes["shape"]["reason"]


def test_cap_on_real_images_only() -> None:
    # Real good images score 5.0 to 8.8, synthetic good images 1 to 2, drifted synthetic 7.0 to 8.8.
    real = [item(f"r{i}", set_name="real") for i in range(20)]
    synthetic_good = [item(f"g{i}") for i in range(20)]
    drifted = [item(f"d{i}", colour_changed=True) for i in range(10)]
    raws = {i.key: raw(colour=round(5.0 + 0.2 * n, 2)) for n, i in enumerate(real)}
    raws |= {i.key: raw(colour=1.0 + 0.05 * n) for n, i in enumerate(synthetic_good)}
    raws |= {i.key: raw(colour=round(7.0 + 0.2 * n, 2)) for n, i in enumerate(drifted)}
    items = real + synthetic_good + drifted

    all_negatives, _ = tune(items, raws, T)
    real_only, notes = tune(items, raws, T, cap_set="real")
    # Counting the 20 easy synthetic negatives, 4 of 40 may be flagged: 4 real images go over 8.0.
    assert all_negatives.colour.max_weighted_delta_e == 8.0
    # Measured on the 20 real images only, at most 2 may be flagged, so the line moves up to 8.4.
    assert real_only.colour.max_weighted_delta_e == 8.4
    assert notes["colour"]["calibration"]["capped_wrong_flag_rate"] <= 0.10


def test_recut_replaces_only_cutout_checks_and_fixes_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import numpy as np

    from driftcheck import evaluate
    from tests.synthetic import garment, simple_mask
    from tests.test_check_pattern import fake_features

    image = garment(stripes=True)
    monkeypatch.setattr(evaluate, "load_image", lambda path: image)
    saved = raw(colour=30.0)  # the old cut out saw a big colour change
    saved["report"]["card"] = {"vision": None, "vision_model": "m", "palette": [], "ocr": [], "product_box": None,
                               "product_share": 0.4}
    saved["report"]["check_seconds"] = {"masks": 2.0, "product_missing": 0.1, "colour": 0.5, "pattern": 1.4}
    saved["report"]["seconds"] = 10.0
    checker = SimpleNamespace(features_fn=fake_features,
                              masks=lambda o, u, c: (simple_mask(o), simple_mask(u), {"method": "clothes parsing"}))

    recut = evaluate.recut_item(item("a"), saved, checker, recording_thresholds(T))["report"]
    by_name = {c["name"]: c for c in recut["checks"]}
    assert by_name["colour"]["score"] < 2  # same image, so no colour change once cut out properly
    assert by_name["text"] == saved["report"]["checks"][2]  # text kept from the saved run
    assert recut["cutout"]["method"] == "clothes parsing"
    assert recut["recut"]["replaced_seconds"] == 4.0
    assert np.isclose(recut["seconds"], 10.0 - 4.0 + recut["recut"]["new_seconds"], atol=0.01)


def test_tune_says_when_no_setting_meets_the_cap() -> None:
    # Every good image has text that never matches, so any min_match flags all of them.
    items = [item(f"g{i}", text_changed=False) for i in range(10)] + [item("d", text_changed=True)]
    raws = {i.key: raw(text_score=10) for i in items}
    _, notes = tune(items, raws, T)
    assert notes["text"]["tuned"] is False
    assert "no setting kept the text_changed wrong flag rate" in notes["text"]["reason"]


def test_baselines_scored_on_overall_drift() -> None:
    items = [item("g", split="test"), item("d", split="test", colour_changed=True)]
    raws = {items[0].key: raw(b1=0.97, b2=False), items[1].key: raw(b1=0.80, b2=None)}
    scores = baseline_scores(items, raws, 0.9)
    assert scores["baseline_1"]["tp"] == 1 and scores["baseline_1"]["tn"] == 1
    assert scores["baseline_2"]["unknown_positive"] == 1
    threshold, note = tune_baseline_1(items, raws)
    assert 0.80 < threshold <= 0.97 and note["recall"] == 1.0


def test_overall_label() -> None:
    assert overall_label(CLEAN) is False
    assert overall_label(CLEAN | {"text_changed": True}) is True
    assert overall_label({name: None for name in ATTRIBUTES}) is None


def test_report_says_not_measurable_without_drift() -> None:
    items = [item(f"g{i}", split="test", set_name="real") for i in range(3)] + [item("s", split="test", colour_changed=True)]
    raws = {i.key: raw() for i in items}
    report = build_report(Path("."), items, {"m": raws}, "m", T, {}, 0.9, {}, {"vision": "m"}, "no cost", "none")
    real = report["results"]["real"]["driftcheck"]["overall"]
    assert real["recall"] is None and real["wrong_flag_rate"] == 0.0
    text = render_markdown(report)
    assert "not measurable" in text and "Real test set" in text and "Synthetic test set" in text


def test_gallery_draws_with_tuned_thresholds() -> None:
    from driftcheck.eval_report import show_tuned_thresholds
    from driftcheck.pipeline import Report

    r = raw(colour=9.0, text_score=85)["report"]
    r["checks"][1]["threshold"] = 7.0
    r["checks"][3]["details"]["patch_similarity"] = 0.85
    r["checks"][2]["details"]["items"][0]["found_by"] = None  # missed at the recording run's 95
    report = Report.model_validate({**r, "original": "o", "output": "u", "card": {
        "vision": None, "vision_model": "m", "palette": [], "ocr": [], "product_box": None, "product_share": 0.4},
        "check_seconds": {}, "models": {}, "thresholds_tuned": True, "text_scripts": ["en"], "created": "x"})
    tuned = T.model_copy(deep=True)
    tuned.colour.max_weighted_delta_e = 11.75
    tuned.pattern.patch_similarity = 0.78
    show_tuned_thresholds(report, tuned)
    assert report.check("colour").threshold == 11.75
    assert report.check("pattern").details["patch_similarity"] == 0.78
    assert report.check("text").details["items"][0]["found_by"] is None  # stays: never found by any reader
    report.check("text").details["items"][0]["found_by"] = "ocr"
    show_tuned_thresholds(report, tuned)
    assert report.check("text").details["items"][0]["found_by"] == "ocr"  # 85 is over the tuned 80


def test_comparison_table_shows_all_three_columns() -> None:
    from driftcheck.eval_report import comparison, render_comparison

    items = [item(f"g{i}", split="test", set_name="real") for i in range(4)] + [item("s", split="test", colour_changed=True)]
    noisy = {i.key: raw(colour=20) for i in items}  # everything flagged
    clean = {i.key: raw(colour=1) for i in items[:4]} | {items[4].key: raw(colour=20)}
    build = lambda raws: build_report(Path("."), items, {"m": raws}, "m", T, {}, 0.9, {}, {}, "", "")  # noqa: E731
    data = comparison(build(noisy), build(clean), build(clean))
    assert data["real_test_flagged"] == {"before": (4, 4), "middle": (0, 4), "after": (0, 4)}
    text = render_comparison(data)
    assert "| Real test images wrongly flagged | 4 of 4 | 0 of 4 | 0 of 4 |" in text
    assert "changed after the first test results were seen" in text
    assert render_comparison(None) == ""


def test_driftcheck_scores_count_per_attribute() -> None:
    items = [item("a", split="test", colour_changed=True), item("b", split="test")]
    raws = {items[0].key: raw(colour=20), items[1].key: raw(colour=1)}
    s = driftcheck_scores(items, raws, T)
    assert s["attributes"]["colour_changed"]["tp"] == 1 and s["attributes"]["colour_changed"]["tn"] == 1
    assert s["attributes"]["logo_changed"]["n"] == 0  # logo is null on every item
    assert s["overall"]["tp"] == 1


def test_collect_items_reads_labels_and_synthetic(tmp_path: Path) -> None:
    from driftcheck.dataset import OutputMeta, ProductMeta, product_dir, write_json
    from tests.helpers import save_image

    for pid in ("p001", "p002"):
        folder = product_dir(tmp_path, pid)
        save_image(folder / "original.jpg")
        write_json(folder / "meta.json", ProductMeta(product_id=pid, category="clothing", source_file="x",
                                                     sha256=pid, ingested_at="2026-10-30", subset="core"))
        save_image(folder / "outputs" / "studio__gemini_app.jpg")
        write_json(folder / "outputs" / "studio__gemini_app.json", OutputMeta(
            product_id=pid, output_id="studio__gemini_app", style="studio", prompt="p", generator="gemini_app",
            model="m", date="2026-10-30"))
    (tmp_path / "split.json").write_text(json.dumps({"calibration": ["p001"], "test": ["p002"]}))
    labels = [{"product_id": "p001", "output_id": "studio__gemini_app", **CLEAN},
              {"product_id": "p002", "output_id": "studio__gemini_app", "colour_changed": False}]  # incomplete
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "synthetic" / "labels.json").write_text(json.dumps(
        [{"product_id": "p002", "output_id": "colour_shift", **CLEAN, "colour_changed": True, "notes": "synthetic"}]))

    items = collect_items(tmp_path)
    assert [(i.set_name, i.split, i.product_id, i.output_id) for i in items] == [
        ("real", "calibration", "p001", "studio__gemini_app"),
        ("synthetic", "test", "p002", "colour_shift"),
    ]
    assert items[1].label["colour_changed"] is True


def test_baseline_2_runs_on_test_split_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from driftcheck import evaluate
    from tests.synthetic import garment

    class FakeReport:
        seconds = 4.0

        def model_dump(self, mode: str = "json") -> dict:
            return {"checks": [], "verdict": "faithful", "seconds": 4.0, "vision_calls": 1}

    asked = []
    monkeypatch.setattr(evaluate, "ask_changed", lambda o, u, client: asked.append(1) or (False, "same"))
    monkeypatch.setattr(evaluate, "whole_image_similarity", lambda o, u, f: 0.9)
    monkeypatch.setattr(evaluate, "load_image", lambda path: garment())
    checker = SimpleNamespace(check=lambda *a: FakeReport(), client=SimpleNamespace(calls=[], provider=SimpleNamespace()))

    calibration = evaluate.run_item(item("c", split="calibration"), checker, None)
    test = evaluate.run_item(item("t", split="test"), checker, None)
    assert calibration["b2"]["skipped"] is True and calibration["b2"]["changed"] is None
    assert test["b2"]["changed"] is False and len(asked) == 1
    assert test["report"]["rate_limit_wait_seconds"] == 0


def test_run_all_saves_and_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from driftcheck import evaluate

    calls = []
    monkeypatch.setattr(evaluate, "run_item", lambda i, checker, f: calls.append(i.key) or raw())
    monkeypatch.setattr(evaluate, "warm_up", lambda checker, f: calls.append("warm"))
    items = [item("a"), item("b")]
    run_all(items, None, None, tmp_path, "qwen2.5vl:3b")
    again = run_all(items, None, None, tmp_path, "qwen2.5vl:3b")
    assert calls == ["warm", "synthetic/a__x", "synthetic/b__x"]  # no warm up on the second run, all saved
    assert set(again) == {"synthetic/a__x", "synthetic/b__x"}
    assert (tmp_path / "eval" / "qwen2.5vl_3b" / "synthetic" / "a__x.json").exists()
