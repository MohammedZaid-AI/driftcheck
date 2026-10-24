---
name: data-rules
description: Use when touching data, labels, the split or synthetic drift.
---
1. Split by product, never by output, so one product is never in both calibration and test.
2. Never change data/split.json after thresholds are tuned. If it must change, retune and say so in PROGRESS.md.
3. Labels are true, false or null. Null means not applicable.
4. Synthetic drift gets automatic labels and is reported as its own set, never mixed with real outputs.
5. Scalio examples are reported on their own and marked as marketing picks.
6. Keep images out of git except data/samples.
