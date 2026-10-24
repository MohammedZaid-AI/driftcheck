---
name: run-eval
description: Use when running the evaluation or writing any results.
---
1. Tune thresholds on the calibration split only, then freeze them in config/thresholds.json.
2. Evaluate on the test split, the synthetic set and the Scalio examples, each reported separately.
3. Always compare DriftCheck with both baselines.
4. Report per attribute precision, recall, F1, wrong flag rate, raw counts, unknown rate, median and p90 time, and API calls per image.
5. Write reports/eval_report.md and reports/eval_report.json with the date, git commit hash, model names and dataset sizes.
6. Save a failure gallery of missed drifts and false alarms.
7. If DriftCheck loses to a baseline on something, say so plainly. Do not hide it.
