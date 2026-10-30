"""Writes reports/eval_report.json, reports/eval_report.md and the failure gallery.

Every number comes from the saved raw runs. Rates that cannot be computed, such as
recall on a set with no drift, are written as not measurable, never as 0.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from driftcheck.checks.base import Thresholds
from driftcheck.checks.verdict import CHECK_ATTRIBUTE
from driftcheck.evaluate import (
    SETS,
    Item,
    baseline_scores,
    decide_all,
    driftcheck_scores,
    overall_label,
    prediction,
    verdict,
)
from driftcheck.labels import ATTRIBUTES

GALLERY_SIZE = 10


def git_commit(root: Path) -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "driftcheck", "config"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return commit + (" with uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def timing(raws: list[dict]) -> dict:
    def stats(values: list[float]) -> dict:
        if not values:
            return {"median_seconds": None, "p90_seconds": None, "n": 0}
        return {"median_seconds": round(float(np.median(values)), 2),
                "p90_seconds": round(float(np.percentile(values, 90)), 2), "n": len(values)}

    return {
        "driftcheck": {**stats([r["report"]["seconds"] for r in raws]),
                       "vision_calls_per_image": round(float(np.mean([r["report"]["vision_calls"] for r in raws])), 2) if raws else None},
        "baseline_1": stats([r["b1"]["seconds"] for r in raws]),
        "baseline_2": {**stats([r["b2"]["seconds"] for r in raws if r["b2"]["seconds"] is not None]),
                       "vision_calls_per_image": round(float(np.mean([r["b2"]["vision_calls"] for r in raws])), 2) if raws else None},
        "rate_limit_wait_seconds_total": round(sum(r["report"].get("rate_limit_wait_seconds", 0)
                                                   + (r["b2"].get("rate_limit_wait_seconds") or 0) for r in raws), 1),
    }


def dataset_sizes(items: list[Item]) -> dict:
    sizes = {}
    for set_name in SETS:
        for split in ("calibration", "test"):
            chosen = [i for i in items if i.set_name == set_name and i.split == split]
            sizes[f"{set_name}_{split}"] = {
                "images": len(chosen),
                "products": len({i.product_id for i in chosen}),
                "drifted_images": sum(overall_label(i.label) is True for i in chosen),
                "positives": {name: sum(i.label[name] is True for i in chosen) for name in ATTRIBUTES},
            }
    return sizes


def evaluate_set(items: list[Item], raws: dict, t: Thresholds, b1_threshold: float) -> dict:
    present = [raws[i.key] for i in items if i.key in raws]
    return {
        "images": len(items),
        "evaluated": len(present),
        "driftcheck": driftcheck_scores(items, raws, t),
        **baseline_scores(items, raws, b1_threshold),
        "timing": timing(present),
    }


def build_report(root: Path, items: list[Item], raws_by_model: dict[str, dict], primary: str, t: Thresholds,
                 tune_notes: dict, b1_threshold: float, b1_note: dict, models: dict, cost_note: str,
                 scalio_note: str) -> dict:
    test = {name: [i for i in items if i.set_name == name and i.split == "test"] for name in SETS}
    report = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "models": models,
        "thresholds": t.model_dump(),
        "tuning": {"split": "calibration", "max_wrong_flag_rate": 0.10, "checks": tune_notes,
                   "baseline_1": {"threshold": b1_threshold, "calibration": b1_note}},
        "dataset_sizes": dataset_sizes(items),
        "results": {name: evaluate_set(test[name], raws_by_model[primary], t, b1_threshold) for name in SETS},
        "model_comparison": {},
        "api_cost": cost_note,
        "scalio_examples": scalio_note,
    }
    for model, raws in raws_by_model.items():
        report["model_comparison"][model] = {name: evaluate_set(test[name], raws, t, b1_threshold) for name in SETS}
    return report


# Markdown ----------------------------------------------------------------------------

def pct(value: float | None) -> str:
    return "not measurable" if value is None else f"{value * 100:.1f}%"


def _score_row(name: str, s: dict) -> str:
    return (f"| {name} | {pct(s['precision'])} | {pct(s['recall'])} | {pct(s['f1'])} | {pct(s['wrong_flag_rate'])} | "
            f"{s['tp']} | {s['fp']} | {s['tn']} | {s['fn']} | {s['unknown_positive'] + s['unknown_negative']} | {pct(s['unknown_rate'])} |")


HEADER = ("| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def render_markdown(report: dict) -> str:
    lines = [
        "# DriftCheck evaluation report",
        "",
        f"Date: {report['date']}",
        f"Git commit: {report['git_commit']}",
        "",
        "Models:",
        *[f"1. {key}: {value}" for key, value in report["models"].items()],
        "",
        "Thresholds were tuned on the calibration split only and then frozen. Results below are on the test split.",
        "An unknown on a drifted image counts as missed. Rates with nothing to divide by are shown as not measurable.",
        "",
        "## Dataset sizes",
        "",
        "| Set | Images | Products | Drifted images |",
        "|---|---|---|---|",
    ]
    for name, size in report["dataset_sizes"].items():
        lines.append(f"| {name.replace('_', ' ')} | {size['images']} | {size['products']} | {size['drifted_images']} |")

    for set_name, result in report["results"].items():
        dc = result["driftcheck"]
        lines += ["", f"## {set_name.capitalize()} test set", "",
                  f"{result['evaluated']} of {result['images']} images evaluated.", "", "### Overall drift detection", "", HEADER,
                  _score_row("DriftCheck", dc["overall"]),
                  _score_row("Baseline 1, whole image DINOv2", result["baseline_1"]),
                  _score_row("Baseline 2, one vision call", result["baseline_2"]),
                  "", "### DriftCheck per attribute", "", HEADER]
        lines += [_score_row(name, s) for name, s in dc["attributes"].items()]
        tm = result["timing"]
        lines += ["", "### Time and calls per image", "",
                  "| | Median seconds | P90 seconds | Vision calls per image |", "|---|---|---|---|",
                  f"| DriftCheck | {tm['driftcheck']['median_seconds']} | {tm['driftcheck']['p90_seconds']} | {tm['driftcheck']['vision_calls_per_image']} |",
                  f"| Baseline 1 | {tm['baseline_1']['median_seconds']} | {tm['baseline_1']['p90_seconds']} | 0 |",
                  f"| Baseline 2 | {tm['baseline_2']['median_seconds']} | {tm['baseline_2']['p90_seconds']} | {tm['baseline_2']['vision_calls_per_image']} |"]

    if len(report["model_comparison"]) > 1:
        lines += ["", "## Vision model comparison", "",
                  "Same frozen thresholds. Only the vision parts differ: product card, shape and logo judge, text backup, baseline 2.", ""]
        for set_name in SETS:
            lines += [f"### {set_name.capitalize()} test set", "",
                      "| Model | Images | Overall wrong flag rate | Overall recall | Shape recall | Shape wrong flag rate | Median seconds | P90 seconds |",
                      "|---|---|---|---|---|---|---|---|"]
            for model, by_set in report["model_comparison"].items():
                r = by_set[set_name]
                overall, shape = r["driftcheck"]["overall"], r["driftcheck"]["attributes"]["shape_changed"]
                lines.append(f"| {model} | {r['evaluated']} | {pct(overall['wrong_flag_rate'])} | {pct(overall['recall'])} | "
                             f"{pct(shape['recall'])} | {pct(shape['wrong_flag_rate'])} | "
                             f"{r['timing']['driftcheck']['median_seconds']} | {r['timing']['driftcheck']['p90_seconds']} |")
            lines.append("")

    lines += ["", "## Tuning on the calibration split", ""]
    for name, note in report["tuning"]["checks"].items():
        state = "tuned" if note["tuned"] else note.get("reason", "not tuned")
        lines.append(f"1. {name}: {state}. Values: {note['values']}")
    lines.append(f"1. baseline 1 similarity threshold: {report['tuning']['baseline_1']['threshold']:.4f}")
    lines += ["", "## Cost", "", report["api_cost"], "", "## Scalio examples", "", report["scalio_examples"], ""]
    return "\n".join(lines)


# Failure gallery ----------------------------------------------------------------------

def failures(items: list[Item], raws: dict, t: Thresholds) -> tuple[list, list]:
    """Missed drifts and false alarms on the test split, worst first: most attributes wrong."""
    missed, false_alarms = [], []
    for item in items:
        if item.split != "test" or item.key not in raws:
            continue
        decisions = decide_all(raws[item.key]["report"], t)
        missed_attrs, flagged_attrs = [], []
        for check, attribute in CHECK_ATTRIBUTE.items():
            pred = prediction(*decisions[check])
            if item.label[attribute] is True and pred != "flag":
                missed_attrs.append(attribute)
            if item.label[attribute] is False and pred == "flag":
                flagged_attrs.append(attribute)
        if missed_attrs:
            missed.append((len(missed_attrs), item, missed_attrs, decisions))
        if flagged_attrs:
            false_alarms.append((len(flagged_attrs), item, flagged_attrs, decisions))
    order = lambda entry: (-entry[0], entry[1].key)  # noqa: E731
    return sorted(missed, key=order)[:GALLERY_SIZE], sorted(false_alarms, key=order)[:GALLERY_SIZE]


def show_tuned_thresholds(report, t: Thresholds) -> None:
    """The raw run used recording thresholds. Redraw limits, patch cut off and found text
    with the frozen ones, so the gallery pictures match the reported decisions."""
    colour = report.check("colour")
    if colour.threshold is not None:
        colour.threshold = t.colour.max_weighted_delta_e
    pattern = report.check("pattern")
    if "patch_similarity" in pattern.details:
        pattern.details["patch_similarity"] = t.pattern.patch_similarity
    for row in report.check("text").details.get("items", []):
        row["found_by"] = row["found_by"] if row["score"] >= t.text.min_match else None


def write_gallery(out_dir: Path, items: list[Item], raws: dict, t: Thresholds) -> list[dict]:
    from driftcheck.heatmap import draw_heatmap
    from driftcheck.pipeline import Report, load_image

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    missed, false_alarms = failures(items, raws, t)
    for kind, group in (("missed", missed), ("false_alarm", false_alarms)):
        for _, item, attributes, decisions in group:
            report = Report.model_validate(raws[item.key]["report"])
            for check in report.checks:  # show the tuned decision, not the recording run's
                check.status, check.applicable = decisions[check.name]
            show_tuned_thresholds(report, t)
            report.verdict = verdict(decisions)
            name = f"{kind}_{item.set_name}_{item.product_id}__{item.output_id}.png"
            # The same capped images the checks saw, so boxes and patches line up.
            draw_heatmap(load_image(item.original), load_image(item.output), report).save(out_dir / name)
            reasons = {c.name: c.reason for c in report.checks if CHECK_ATTRIBUTE[c.name] in attributes}
            entries.append({"kind": kind, "item": item.key, "attributes": attributes, "file": name,
                            "reasons_from_recorded_run": reasons})
    (out_dir / "gallery.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    return entries


def gallery_markdown(entries: list[dict]) -> str:
    if not entries:
        return "No missed drifts or false alarms on the test split.\n"
    counts = Counter(e["kind"] for e in entries)
    lines = [f"{counts['missed']} missed drifts and {counts['false_alarm']} false alarms shown, worst first.", ""]
    for e in entries:
        lines.append(f"1. {e['kind'].replace('_', ' ')}: {e['item']}, {', '.join(e['attributes'])}. Image: gallery/{e['file']}")
    return "\n".join(lines) + "\n"
