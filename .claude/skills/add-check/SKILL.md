---
name: add-check
description: Use when creating or changing any check in driftcheck/checks.
---
1. Each check lives in its own file in driftcheck/checks.
2. It returns a CheckResult with name, status pass or fail or unknown, score, threshold, reason and details.
3. Return unknown, never pass, when the check cannot decide, for example no text in the original or the product is not visible.
4. Read thresholds from config/thresholds.json. Never hardcode them in the check.
5. Write at least three tests with small synthetic images: one clear pass, one clear fail, one unknown.
6. Register the check in the overall verdict and in the report.
7. Use the commit-step skill.
