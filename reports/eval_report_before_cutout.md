# DriftCheck evaluation report

Date: 2026-10-30T22:43:45+00:00
Git commit: 96cf374eef69a6c1f30b1920831044bd730ab951

Models:
1. vision: ollama/qwen2.5vl:7b
1. compared vision models: qwen2.5vl:7b
1. pattern and baseline 1: facebook/dinov2-small
1. segmentation: rembg/isnet-general-use
1. ocr: easyocr readers en, en_hi, en_kn, en_bn
1. outputs: Gemini app, model 3.6 Flash, made by hand

Thresholds were tuned on the calibration split only and then frozen. Results below are on the test split.
An unknown on a drifted image counts as missed. Rates with nothing to divide by are shown as not measurable.

## Dataset sizes

| Set | Images | Products | Drifted images |
|---|---|---|---|
| real calibration | 40 | 20 | 0 |
| real test | 40 | 20 | 0 |
| synthetic calibration | 167 | 20 | 107 |
| synthetic test | 159 | 20 | 99 |

## Real test set

40 of 40 images evaluated.

### Overall drift detection

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| DriftCheck | 0.0% | not measurable | not measurable | 67.5% | 0 | 27 | 13 | 0 | 0 | 0.0% |
| Baseline 1, whole image DINOv2 | 0.0% | not measurable | not measurable | 25.0% | 0 | 10 | 30 | 0 | 0 | 0.0% |
| Baseline 2, one vision call | not measurable | not measurable | not measurable | 0.0% | 0 | 0 | 40 | 0 | 0 | 0.0% |

### DriftCheck per attribute

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_changed | 0.0% | not measurable | not measurable | 10.0% | 0 | 4 | 35 | 0 | 1 | 2.5% |
| colour_changed | 0.0% | not measurable | not measurable | 12.5% | 0 | 5 | 34 | 0 | 1 | 2.5% |
| pattern_changed | 0.0% | not measurable | not measurable | 42.5% | 0 | 17 | 22 | 0 | 1 | 2.5% |
| logo_changed | 0.0% | not measurable | not measurable | 14.3% | 0 | 5 | 29 | 0 | 1 | 2.9% |
| text_changed | 0.0% | not measurable | not measurable | 22.9% | 0 | 8 | 26 | 0 | 1 | 2.9% |
| product_missing | 0.0% | not measurable | not measurable | 2.5% | 0 | 1 | 39 | 0 | 0 | 0.0% |

### Time and calls per image

| | Median seconds | P90 seconds | Vision calls per image |
|---|---|---|---|
| DriftCheck | 51.21 | 145.82 | 2.42 |
| Baseline 1 | 0.56 | 1.03 | 0 |
| Baseline 2 | 7.31 | 10.15 | 1.0 |

## Synthetic test set

159 of 159 images evaluated.

### Overall drift detection

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| DriftCheck | 74.8% | 80.8% | 77.7% | 45.0% | 80 | 27 | 33 | 14 | 5 | 3.1% |
| Baseline 1, whole image DINOv2 | 100.0% | 2.0% | 4.0% | 0.0% | 2 | 0 | 60 | 97 | 0 | 0.0% |
| Baseline 2, one vision call | 100.0% | 5.1% | 9.6% | 0.0% | 5 | 0 | 60 | 94 | 0 | 0.0% |

### DriftCheck per attribute

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_changed | 14.3% | 9.1% | 11.1% | 4.7% | 1 | 6 | 114 | 10 | 8 | 5.8% |
| colour_changed | 86.7% | 65.0% | 74.3% | 1.7% | 13 | 2 | 109 | 7 | 8 | 5.8% |
| pattern_changed | 100.0% | 73.3% | 84.6% | 0.0% | 22 | 0 | 109 | 0 | 8 | 5.8% |
| logo_changed | not measurable | not measurable | not measurable | not measurable | 0 | 0 | 0 | 0 | 0 | not measurable |
| text_changed | 61.4% | 84.4% | 71.1% | 36.2% | 27 | 17 | 30 | 0 | 5 | 6.3% |
| product_missing | 33.3% | 10.0% | 15.4% | 2.9% | 2 | 4 | 131 | 17 | 5 | 3.1% |

### Time and calls per image

| | Median seconds | P90 seconds | Vision calls per image |
|---|---|---|---|
| DriftCheck | 17.74 | 33.59 | 1.45 |
| Baseline 1 | 0.51 | 0.55 | 0 |
| Baseline 2 | 6.52 | 8.8 | 1.0 |

## Tuning on the calibration split

1. product_missing: tuned. Values: {'missing_share': 0.02, 'small_share': 0.06}
1. colour: tuned. Values: {'max_weighted_delta_e': 11.75, 'white_balance': True}
1. text: no setting kept the text_changed wrong flag rate at or under 10% on calibration, starting values kept. Values: {'min_match': 80.0, 'min_item_length': 2, 'vision_backup': True}
1. pattern: tuned. Values: {'patch_similarity': 0.78, 'max_drift_share': 0.45, 'min_patches': 20}
1. shape: tuned. Values: {'min_ordinal_steps': 3, 'min_changed_parts': 1, 'use_silhouette': False}
1. baseline 1 similarity threshold: 0.5783

## Cost

Vision calls run on a local Ollama model, so there is no API charge. Calls per image are in the tables.

## Scalio examples

No images in data/scalio_examples, so this part was not run.

## Failure gallery

10 missed drifts and 10 false alarms shown, worst first.

1. missed: synthetic/p015__pattern_blur, text_changed, pattern_changed. Image: gallery/missed_synthetic_p015__pattern_blur.png
1. missed: synthetic/p029__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p029__pattern_swap.png
1. missed: synthetic/p043__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p043__pattern_swap.png
1. missed: synthetic/p068__pattern_blur, text_changed, pattern_changed. Image: gallery/missed_synthetic_p068__pattern_blur.png
1. missed: synthetic/p068__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p068__pattern_swap.png
1. missed: synthetic/p010__product_erase, product_missing. Image: gallery/missed_synthetic_p010__product_erase.png
1. missed: synthetic/p010__sleeve_erase, shape_changed. Image: gallery/missed_synthetic_p010__sleeve_erase.png
1. missed: synthetic/p015__colour_shift, colour_changed. Image: gallery/missed_synthetic_p015__colour_shift.png
1. missed: synthetic/p018__pattern_blur, pattern_changed. Image: gallery/missed_synthetic_p018__pattern_blur.png
1. missed: synthetic/p018__sleeve_erase, shape_changed. Image: gallery/missed_synthetic_p018__sleeve_erase.png
1. false alarm: real/p029__festival_poster__gemini_app, colour_changed, pattern_changed, logo_changed. Image: gallery/false_alarm_real_p029__festival_poster__gemini_app.png
1. false alarm: real/p018__model_shot__gemini_app, pattern_changed, shape_changed. Image: gallery/false_alarm_real_p018__model_shot__gemini_app.png
1. false alarm: real/p018__studio__gemini_app, pattern_changed, shape_changed. Image: gallery/false_alarm_real_p018__studio__gemini_app.png
1. false alarm: real/p021__lifestyle__gemini_app, colour_changed, pattern_changed. Image: gallery/false_alarm_real_p021__lifestyle__gemini_app.png
1. false alarm: real/p027__model_shot__gemini_app, text_changed, pattern_changed. Image: gallery/false_alarm_real_p027__model_shot__gemini_app.png
1. false alarm: real/p031__model_shot__gemini_app, colour_changed, pattern_changed. Image: gallery/false_alarm_real_p031__model_shot__gemini_app.png
1. false alarm: real/p035__lifestyle__gemini_app, colour_changed, pattern_changed. Image: gallery/false_alarm_real_p035__lifestyle__gemini_app.png
1. false alarm: real/p043__lifestyle__gemini_app, colour_changed, text_changed. Image: gallery/false_alarm_real_p043__lifestyle__gemini_app.png
1. false alarm: real/p067__model_shot__gemini_app, text_changed, pattern_changed. Image: gallery/false_alarm_real_p067__model_shot__gemini_app.png
1. false alarm: real/p068__instagram_post__gemini_app, pattern_changed, logo_changed. Image: gallery/false_alarm_real_p068__instagram_post__gemini_app.png
