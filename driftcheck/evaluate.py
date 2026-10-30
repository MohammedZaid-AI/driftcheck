"""Evaluation: run every check once, then decide, tune and score from the saved results.

Each pair is run once per vision model with recording thresholds that keep every
detail: all checks run even for a small product, text items below 95 go to the
vision backup, and the judge records silhouette. The raw report is saved to
cache/eval, so tuning and scoring never call a model again.

Tuning uses the calibration half only. It picks, per check, the setting with the
best recall while good images are wrongly flagged at most 10 percent of the time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from driftcheck.baselines import ask_changed, whole_image_similarity
from driftcheck.checks.base import Thresholds
from driftcheck.checks.verdict import CHECK_ATTRIBUTE
from driftcheck.dataset import list_outputs, output_image, product_dir, read_meta
from driftcheck.labels import ATTRIBUTES, LabelStore
from driftcheck.log import get_logger
from driftcheck.pipeline import load_image
from driftcheck.split import read_split

log = get_logger(__name__)

MAX_WRONG_FLAG_RATE = 0.10
SETS = ("real", "synthetic")


@dataclass
class Item:
    set_name: str  # real or synthetic
    split: str  # calibration or test
    product_id: str
    output_id: str
    original: Path
    output: Path
    label: dict[str, bool | None]
    text_scripts: list[str] | None

    @property
    def key(self) -> str:
        return f"{self.set_name}/{self.product_id}__{self.output_id}"


def collect_items(data_dir: Path) -> list[Item]:
    """Every labelled real output and every synthetic output of a product in the split."""
    split = read_split(data_dir)
    half = {pid: name for name in ("calibration", "test") for pid in split[name]}
    items: list[Item] = []

    store = LabelStore(data_dir / "labels.json")
    for product_id, output_id in list_outputs(data_dir):
        if product_id not in half or not store.is_complete(product_id, output_id):
            continue
        row = store.get(product_id, output_id)
        items.append(Item("real", half[product_id], product_id, output_id,
                          product_dir(data_dir, product_id) / "original.jpg",
                          output_image(data_dir, product_id, output_id),
                          {name: row[name] for name in ATTRIBUTES}, read_meta(data_dir, product_id).text_scripts))

    synthetic = data_dir / "synthetic" / "labels.json"
    if synthetic.exists():
        for row in json.loads(synthetic.read_text(encoding="utf-8")):
            product_id = row["product_id"]
            if product_id not in half:
                continue
            items.append(Item("synthetic", half[product_id], product_id, row["output_id"],
                              product_dir(data_dir, product_id) / "original.jpg",
                              data_dir / "synthetic" / product_id / f"{row['output_id']}.jpg",
                              {name: row[name] for name in ATTRIBUTES}, read_meta(data_dir, product_id).text_scripts))
    return items


def recording_thresholds(base: Thresholds) -> Thresholds:
    """Settings for the recorded run that keep every detail the tuning needs."""
    t = base.model_copy(deep=True)
    t.segmentation.missing_share = 0.001
    t.segmentation.small_share = 0.001
    t.text.min_match = 95
    t.judge.use_silhouette = True
    return t


def model_tag(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def raw_path(cache_dir: Path, model: str, item: Item) -> Path:
    return cache_dir / "eval" / model_tag(model) / item.set_name / f"{item.product_id}__{item.output_id}.json"


def _waited(checker) -> float:
    """Seconds the provider has spent waiting on rate limits so far. Kept out of speed figures."""
    return getattr(checker.client.provider, "waited_seconds", 0.0)


def run_item(item: Item, checker, features_fn: Callable) -> dict:
    """One recorded run of DriftCheck and both baselines on one pair.
    Baseline 2 runs on the test split only: it is reported there and never tuned."""
    waited_before = _waited(checker)
    report = checker.check(item.original, item.output, item.text_scripts)
    report_wait = _waited(checker) - waited_before
    original, output = load_image(item.original), load_image(item.output)

    began = time.perf_counter()
    similarity = whole_image_similarity(original, output, features_fn)
    b1_seconds = time.perf_counter() - began

    b2 = {"changed": None, "reason": "not run on the calibration split", "seconds": None, "vision_calls": 0, "skipped": True}
    if item.split == "test":
        calls_before, waited_before = len(checker.client.calls), _waited(checker)
        began = time.perf_counter()
        changed, reason = ask_changed(original, output, checker.client)
        b2_wait = _waited(checker) - waited_before
        b2 = {"changed": changed, "reason": reason,
              "seconds": round(time.perf_counter() - began - b2_wait, 3),
              "rate_limit_wait_seconds": round(b2_wait, 3),
              "vision_calls": sum(not call.cached for call in checker.client.calls[calls_before:])}

    report_json = report.model_dump(mode="json")
    report_json["rate_limit_wait_seconds"] = round(report_wait, 3)
    report_json["seconds"] = round(report.seconds - report_wait, 3)  # DriftCheck's own time, without waiting
    return {
        "report": report_json,
        "b1": {"similarity": similarity, "seconds": round(b1_seconds, 3)},
        "b2": b2,
    }


def run_all(items: Iterable[Item], checker, features_fn: Callable, cache_dir: Path, model: str) -> dict[str, dict]:
    """Run every item that has no saved result yet. Returns item key to raw result."""
    items = list(items)
    results = {}
    for number, item in enumerate(items, 1):
        path = raw_path(cache_dir, model, item)
        if path.exists():
            results[item.key] = json.loads(path.read_text(encoding="utf-8"))
            continue
        # Logged before the work, so a native crash shows which pair it was on.
        log.info("starting %d/%d %s", number, len(items), item.key)
        raw = run_item(item, checker, features_fn)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw), encoding="utf-8")
        results[item.key] = raw
        log.info("%d/%d %s %s in %.1fs", number, len(items), item.key, raw["report"]["verdict"], raw["report"]["seconds"])
    return results


def load_raw(items: Iterable[Item], cache_dir: Path, model: str) -> dict[str, dict]:
    found = {}
    for item in items:
        path = raw_path(cache_dir, model, item)
        if path.exists():
            found[item.key] = json.loads(path.read_text(encoding="utf-8"))
    return found


# Deciding again from saved details --------------------------------------------------------

Decision = tuple[str, bool]  # status, applicable


def _check(report: dict, name: str) -> dict:
    return next(c for c in report["checks"] if c["name"] == name)


def decide_segmentation(report: dict, t: Thresholds) -> Decision:
    details = _check(report, "product_missing")["details"]
    if "original_share" not in details:
        return "unknown", True
    if details["original_share"] < t.segmentation.missing_share:
        return "unknown", True
    if details["output_share"] < t.segmentation.missing_share:
        return "fail", True
    if details["output_share"] < t.segmentation.small_share:
        return "unknown", True
    return "pass", True


def decide_check(report: dict, name: str, t: Thresholds) -> Decision:
    """The status a check would have had with thresholds t."""
    if name == "product_missing":
        return decide_segmentation(report, t)
    if decide_segmentation(report, t)[0] != "pass":
        return "unknown", True
    check = _check(report, name)
    details = check["details"]
    if not check.get("applicable", True):
        return "unknown", False

    if name == "colour":
        if check["score"] is None:
            return "unknown", True
        return ("fail" if check["score"] > t.colour.max_weighted_delta_e else "pass"), True
    if name == "text":
        rows = details.get("items")
        if not rows:
            return "unknown", True
        return ("pass" if all(r["score"] >= t.text.min_match for r in rows) else "fail"), True
    if name == "pattern":
        grid = details.get("similarity_grid")
        if not grid:
            return "unknown", True
        values = np.array([v for row in grid for v in row if v is not None])
        share = float((values < t.pattern.patch_similarity).mean())
        return ("fail" if share > t.pattern.max_drift_share else "pass"), True
    if name == "shape":
        return decide_shape(details, t)
    if name == "logo":
        return check["status"], True
    raise ValueError(name)


def decide_shape(details: dict, t: Thresholds) -> Decision:
    parts = {name: part for name, part in details.items() if isinstance(part, dict) and "original" in part}
    if not parts:
        return "unknown", True
    if not t.judge.use_silhouette:
        parts.pop("silhouette", None)
    changed = []
    seen = 0
    for name, part in parts.items():
        if part["changed"] is None:
            continue
        seen += 1
        if part.get("steps") is not None:  # sleeves and length are ordered, so steps count
            changed.append(part["steps"] >= t.judge.min_ordinal_steps)
        else:
            changed.append(bool(part["changed"]))
    if not seen:
        return "unknown", False
    return ("fail" if sum(changed) >= t.judge.min_changed_parts else "pass"), True


def decide_all(report: dict, t: Thresholds) -> dict[str, Decision]:
    return {name: decide_check(report, name, t) for name in CHECK_ATTRIBUTE}


def verdict(decisions: dict[str, Decision]) -> str:
    statuses = [status for status, applicable in decisions.values() if applicable]
    if "fail" in statuses:
        return "drifted"
    if statuses and all(status == "pass" for status in statuses):
        return "faithful"
    return "unknown"


# Metrics ---------------------------------------------------------------------------------

def prediction(status: str, applicable: bool) -> str:
    """flag, clear or unknown. A check that does not apply is a clear: the attribute is not there."""
    if status == "fail":
        return "flag"
    if status == "pass" or not applicable:
        return "clear"
    return "unknown"


def overall_label(label: dict[str, bool | None]) -> bool | None:
    values = [value for value in label.values() if value is not None]
    if any(values):
        return True
    return False if values else None


def score(pairs: list[tuple[bool | None, str]]) -> dict:
    """Counts and rates for (true label, prediction) pairs. Rates with nothing to divide by are None.
    An unknown on a drifted image counts as missed. An unknown on a good image is not a wrong flag."""
    pairs = [(label, pred) for label, pred in pairs if label is not None]
    tp = sum(label and pred == "flag" for label, pred in pairs)
    fn = sum(label and pred == "clear" for label, pred in pairs)
    unknown_pos = sum(label and pred == "unknown" for label, pred in pairs)
    fp = sum(not label and pred == "flag" for label, pred in pairs)
    tn = sum(not label and pred == "clear" for label, pred in pairs)
    unknown_neg = sum(not label and pred == "unknown" for label, pred in pairs)
    positives, negatives = tp + fn + unknown_pos, fp + tn + unknown_neg

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    return {
        "n": len(pairs), "positives": positives, "negatives": negatives,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "unknown_positive": unknown_pos, "unknown_negative": unknown_neg,
        "precision": precision, "recall": recall, "f1": f1,
        "wrong_flag_rate": fp / negatives if negatives else None,
        "unknown_rate": (unknown_pos + unknown_neg) / len(pairs) if pairs else None,
    }


def driftcheck_scores(items: list[Item], raws: dict[str, dict], t: Thresholds) -> dict:
    """Per attribute scores and overall drift detection for DriftCheck."""
    per_attribute: dict[str, list] = {name: [] for name in ATTRIBUTES}
    overall = []
    for item in items:
        raw = raws.get(item.key)
        if raw is None:
            continue
        decisions = decide_all(raw["report"], t)
        for check, attribute in CHECK_ATTRIBUTE.items():
            per_attribute[attribute].append((item.label[attribute], prediction(*decisions[check])))
        verdict_pred = {"drifted": "flag", "faithful": "clear", "unknown": "unknown"}[verdict(decisions)]
        overall.append((overall_label(item.label), verdict_pred))
    return {"attributes": {name: score(pairs) for name, pairs in per_attribute.items()}, "overall": score(overall)}


def baseline_scores(items: list[Item], raws: dict[str, dict], b1_threshold: float) -> dict:
    b1, b2 = [], []
    for item in items:
        raw = raws.get(item.key)
        if raw is None:
            continue
        truth = overall_label(item.label)
        b1.append((truth, "flag" if raw["b1"]["similarity"] < b1_threshold else "clear"))
        changed = raw["b2"]["changed"]
        b2.append((truth, "unknown" if changed is None else ("flag" if changed else "clear")))
    return {"baseline_1": score(b1), "baseline_2": score(b2)}


# Tuning ----------------------------------------------------------------------------------

def _better(candidate: dict, best: dict | None) -> bool:
    """More recall wins, then fewer wrong flags. Equal scores keep the earlier candidate."""
    if best is None:
        return True
    if (candidate["recall"] or 0) != (best["recall"] or 0):
        return (candidate["recall"] or 0) > (best["recall"] or 0)
    return (candidate["wrong_flag_rate"] or 0) < (best["wrong_flag_rate"] or 0)


def _choose(items: list[Item], raws: dict, attribute: str, check: str, candidates: list[Thresholds]) -> tuple[Thresholds, dict]:
    best, best_score = candidates[0], None
    for t in candidates:
        pairs = [(i.label[attribute], prediction(*decide_check(raws[i.key]["report"], check, t)))
                 for i in items if i.key in raws]
        result = score(pairs)
        if result["wrong_flag_rate"] is not None and result["wrong_flag_rate"] > MAX_WRONG_FLAG_RATE:
            continue
        if _better(result, best_score):
            best, best_score = t, result
    return best, best_score or {}


def _with(t: Thresholds, section: str, **values) -> Thresholds:
    copy = t.model_copy(deep=True)
    for key, value in values.items():
        setattr(getattr(copy, section), key, value)
    return copy


def tune(items: list[Item], raws: dict[str, dict], start: Thresholds) -> tuple[Thresholds, dict]:
    """Tune every threshold on the given calibration items. Returns the thresholds and notes per check."""
    notes: dict[str, dict] = {}
    t = start.model_copy(deep=True)

    def record(name: str, attribute: str, chosen: Thresholds, result: dict, section: str) -> None:
        positives = result.get("positives", 0)
        notes[name] = {"tuned": bool(positives), "calibration": result,
                       "values": getattr(chosen, section).model_dump()}
        if not positives:
            notes[name]["reason"] = f"no {attribute} positives in calibration, starting values kept"

    # Segmentation first, because every other check depends on it.
    candidates = [_with(t, "segmentation", missing_share=m, small_share=3 * m) for m in (0.002, 0.005, 0.01, 0.02, 0.03)]
    chosen, result = _choose(items, raws, "product_missing", "product_missing", candidates)
    t = chosen if result.get("positives") else t
    record("product_missing", "product_missing", t, result, "segmentation")

    colour_scores = {round(_check(raws[i.key]["report"], "colour")["score"], 2) for i in items
                     if i.key in raws and _check(raws[i.key]["report"], "colour")["score"] is not None}
    candidates = [_with(t, "colour", max_weighted_delta_e=s) for s in sorted(colour_scores | {t.colour.max_weighted_delta_e})]
    chosen, result = _choose(items, raws, "colour_changed", "colour", candidates)
    t = chosen if result.get("positives") else t
    record("colour", "colour_changed", t, result, "colour")

    candidates = [_with(t, "text", min_match=m) for m in range(60, 96, 5)]
    chosen, result = _choose(items, raws, "text_changed", "text", candidates)
    t = chosen if result.get("positives") else t
    record("text", "text_changed", t, result, "text")

    candidates = [_with(t, "pattern", patch_similarity=round(p, 2), max_drift_share=round(s, 2))
                  for p in np.arange(0.70, 0.96, 0.01) for s in np.arange(0.05, 0.55, 0.05)]
    chosen, result = _choose(items, raws, "pattern_changed", "pattern", candidates)
    t = chosen if result.get("positives") else t
    record("pattern", "pattern_changed", t, result, "pattern")

    candidates = [_with(t, "judge", min_ordinal_steps=k, min_changed_parts=c, use_silhouette=s)
                  for k in (1, 2, 3) for c in (1, 2) for s in (False, True)]
    chosen, result = _choose(items, raws, "shape_changed", "shape", candidates)
    t = chosen if result.get("positives") else t
    record("shape", "shape_changed", t, result, "judge")

    t.tuned = True
    return t, notes


def tune_baseline_1(items: list[Item], raws: dict[str, dict]) -> tuple[float, dict]:
    """One similarity threshold for baseline 1, chosen the same way on overall drift."""
    sims = sorted({raws[i.key]["b1"]["similarity"] for i in items if i.key in raws})
    best, best_score = 0.0, None
    for threshold in [0.0] + [s + 1e-9 for s in sims]:
        pairs = [(overall_label(i.label), "flag" if raws[i.key]["b1"]["similarity"] < threshold else "clear")
                 for i in items if i.key in raws]
        result = score(pairs)
        if result["wrong_flag_rate"] is not None and result["wrong_flag_rate"] > MAX_WRONG_FLAG_RATE:
            continue
        if _better(result, best_score):
            best, best_score = threshold, result
    return best, best_score or {}
