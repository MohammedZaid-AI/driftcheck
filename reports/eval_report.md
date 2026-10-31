# What changed since the first report

Every change after the first report is listed here, and each column shows the numbers after it, so nothing is hidden.

1. Cut out. For worn clothing and accessories, a clothes parsing model keeps only the garment, not the model's face, arms, legs or other clothes. Flat lays and studio shots keep the rembg cut out. Product missing, colour and pattern were run again with the new cut out on the saved run. Text, shape and logo come from the saved run unchanged.
2. Tuning. The first tuning kept wrong flags at or under 10 percent across all calibration negatives, which were mostly synthetic. The retuning measures that 10 percent on the user's real calibration images only. Catches are still measured on the synthetic changes. This rule was changed after the first test results were seen.
3. Search ranges. With the cap on real images, no setting in the first ranges met it for pattern and text. The ranges were widened: pattern patch similarity 0.40 to 0.94 and share up to 0.95, text match 30 to 95. Also changed after results were seen.
4. Routing by image type. Studio outputs are flat images. Model shots, lifestyle scenes, posters and Instagram posts are scene images. A synthetic copy takes the type of its original: a scene if the original shows the product worn. On scene images colour and pattern are skipped. A skipped check is neither a pass nor a fail: it is left out of the verdict and out of that check's counts. A deliberate colour or pattern change on a scene image can then only be caught by the checks that still run, and counts as missed in the overall catch rate if they do not catch it. Thresholds are the ones frozen at the retuning, not tuned again. Also decided after results were seen.
5. Text decision fix in the evaluation. Found while building the demo. Live, a text item that OCR misses goes to the vision model, and counts as found if its reading matches at the text threshold. The evaluation only used vision readings that matched at 95, the recording threshold, so at the tuned threshold of 40 it marked items missing that the live check finds. The evaluation now uses the saved vision readings the same way as the live check. The earlier columns used the old decision. Thresholds are unchanged.

| | First report | Cut out fix and retune | Routing by image type | Text decision fix |
|---|---|---|---|---|
| Real test images wrongly flagged | 27 of 40 | 20 of 40 | 15 of 40 | 11 of 40 |
| Synthetic test changes caught | 80 of 99 | 60 of 99 | 58 of 99 | 57 of 99 |
| Real test, overall wrong flag rate | 67.5% | 50.0% | 37.5% | 27.5% |
| Real test, shape_changed wrong flag rate | 10.0% | 10.0% | 10.0% | 10.0% |
| Real test, colour_changed wrong flag rate | 12.5% | 7.5% | 0.0% | 0.0% |
| Real test, pattern_changed wrong flag rate | 42.5% | 12.5% | 0.0% | 0.0% |
| Real test, logo_changed wrong flag rate | 14.3% | 14.3% | 14.3% | 14.3% |
| Real test, text_changed wrong flag rate | 22.9% | 14.3% | 14.3% | 2.9% |
| Real test, product_missing wrong flag rate | 2.5% | 2.5% | 2.5% | 2.5% |
| Synthetic test, overall recall | 80.8% | 60.6% | 58.6% | 57.6% |
| Synthetic test, overall precision | 74.8% | 76.9% | 76.3% | 81.4% |
| Synthetic test, overall wrong flag rate | 45.0% | 30.0% | 30.0% | 21.7% |
| Synthetic test, shape_changed recall | 9.1% | 9.1% | 9.1% | 9.1% |
| Synthetic test, colour_changed recall | 65.0% | 35.0% | 33.3% | 33.3% |
| Synthetic test, pattern_changed recall | 73.3% | 43.3% | 46.2% | 46.2% |
| Synthetic test, logo_changed recall | not measurable | not measurable | not measurable | not measurable |
| Synthetic test, text_changed recall | 84.4% | 59.4% | 59.4% | 56.2% |
| Synthetic test, product_missing recall | 10.0% | 10.0% | 10.0% | 10.0% |
| Baseline 1, real test wrong flag rate | 25.0% | 0.0% | 0.0% | 0.0% |
| Baseline 1, synthetic test recall | 2.0% | 0.0% | 0.0% | 0.0% |

Thresholds in each column:

1. First report: colour 11.75, pattern patch 0.78 and share 0.45, text 80.0, product missing 0.02, shape steps 3
1. Cut out fix and retune: colour 15.01, pattern patch 0.82 and share 0.9, text 40.0, product missing 0.02, shape steps 3
1. Routing by image type: colour 15.01, pattern patch 0.82 and share 0.9, text 40.0, product missing 0.02, shape steps 3
1. Text decision fix: colour 15.01, pattern patch 0.82 and share 0.9, text 40.0, product missing 0.02, shape steps 3

# DriftCheck evaluation report

Date: 2026-10-31T14:53:13+00:00
Git commit: 50b77b965342c8d01ebf5a0281ebf30dfed3d3ed

Models:
1. vision: ollama/qwen2.5vl:7b
1. result set: qwen2.5vl_7b_garmentcut
1. compared vision models: qwen2.5vl_7b_garmentcut, qwen2.5vl:7b
1. pattern and baseline 1: facebook/dinov2-small
1. segmentation: rembg/isnet-general-use, plus mattmdjaga/segformer_b2_clothes for worn clothing and accessories, licence listed as other
1. ocr: easyocr readers en, en_hi, en_kn, en_bn
1. outputs: Gemini app, model 3.6 Flash, made by hand
1. tuning: calibration split, 207 pairs, results qwen2.5vl_7b_garmentcut, wrong flag cap on real calibration images only

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
| DriftCheck | 0.0% | not measurable | not measurable | 27.5% | 0 | 11 | 29 | 0 | 0 | 0.0% |
| Baseline 1, whole image DINOv2 | not measurable | not measurable | not measurable | 0.0% | 0 | 0 | 40 | 0 | 0 | 0.0% |
| Baseline 2, one vision call | not measurable | not measurable | not measurable | 0.0% | 0 | 0 | 40 | 0 | 0 | 0.0% |

### DriftCheck per attribute

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_changed | 0.0% | not measurable | not measurable | 10.0% | 0 | 4 | 35 | 0 | 1 | 2.5% |
| colour_changed | not measurable | not measurable | not measurable | 0.0% | 0 | 0 | 18 | 0 | 0 | 0.0% |
| pattern_changed | not measurable | not measurable | not measurable | 0.0% | 0 | 0 | 18 | 0 | 0 | 0.0% |
| logo_changed | 0.0% | not measurable | not measurable | 14.3% | 0 | 5 | 29 | 0 | 1 | 2.9% |
| text_changed | 0.0% | not measurable | not measurable | 2.9% | 0 | 1 | 33 | 0 | 1 | 2.9% |
| product_missing | 0.0% | not measurable | not measurable | 2.5% | 0 | 1 | 39 | 0 | 0 | 0.0% |

### Time and calls per image

| | Median seconds | P90 seconds | Vision calls per image |
|---|---|---|---|
| DriftCheck | 51.52 | 143.56 | 2.42 |
| Baseline 1 | 0.56 | 1.03 | 0 |
| Baseline 2 | 7.31 | 10.15 | 1.0 |

## Synthetic test set

159 of 159 images evaluated.

### Overall drift detection

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| DriftCheck | 81.4% | 57.6% | 67.5% | 21.7% | 57 | 13 | 47 | 38 | 4 | 2.5% |
| Baseline 1, whole image DINOv2 | not measurable | 0.0% | not measurable | 0.0% | 0 | 0 | 60 | 99 | 0 | 0.0% |
| Baseline 2, one vision call | 100.0% | 5.1% | 9.6% | 0.0% | 5 | 0 | 60 | 94 | 0 | 0.0% |

### DriftCheck per attribute

| | Precision | Recall | F1 | Wrong flag rate | TP | FP | TN | FN | Unknown | Unknown rate |
|---|---|---|---|---|---|---|---|---|---|---|
| shape_changed | 14.3% | 9.1% | 11.1% | 4.7% | 1 | 6 | 115 | 10 | 7 | 5.0% |
| colour_changed | 85.7% | 33.3% | 48.0% | 0.9% | 6 | 1 | 99 | 12 | 7 | 5.6% |
| pattern_changed | 100.0% | 46.2% | 63.2% | 0.0% | 12 | 0 | 99 | 7 | 7 | 5.6% |
| logo_changed | not measurable | not measurable | not measurable | not measurable | 0 | 0 | 0 | 0 | 0 | not measurable |
| text_changed | 94.7% | 56.2% | 70.6% | 2.1% | 18 | 1 | 46 | 9 | 5 | 6.3% |
| product_missing | 33.3% | 10.0% | 15.4% | 2.9% | 2 | 4 | 132 | 17 | 4 | 2.5% |

### Time and calls per image

| | Median seconds | P90 seconds | Vision calls per image |
|---|---|---|---|
| DriftCheck | 16.45 | 33.54 | 1.45 |
| Baseline 1 | 0.51 | 0.55 | 0 |
| Baseline 2 | 6.52 | 8.8 | 1.0 |

## Vision model comparison

Same frozen thresholds. Only the vision parts differ: product card, shape and logo judge, text backup, baseline 2.

### Real test set

| Model | Images | Overall wrong flag rate | Overall recall | Shape recall | Shape wrong flag rate | Median seconds | P90 seconds |
|---|---|---|---|---|---|---|---|
| qwen2.5vl_7b_garmentcut | 40 | 27.5% | not measurable | not measurable | 10.0% | 51.52 | 143.56 |
| qwen2.5vl:7b | 40 | 27.5% | not measurable | not measurable | 10.0% | 51.21 | 145.82 |

### Synthetic test set

| Model | Images | Overall wrong flag rate | Overall recall | Shape recall | Shape wrong flag rate | Median seconds | P90 seconds |
|---|---|---|---|---|---|---|---|
| qwen2.5vl_7b_garmentcut | 159 | 21.7% | 57.6% | 9.1% | 4.7% | 16.45 | 33.54 |
| qwen2.5vl:7b | 159 | 21.7% | 57.6% | 9.1% | 4.7% | 17.74 | 33.59 |


## Tuning on the calibration split

1. product_missing: tuned. Values: {'missing_share': 0.02, 'small_share': 0.06}
1. colour: tuned. Values: {'max_weighted_delta_e': 15.01, 'white_balance': True}
1. text: tuned. Values: {'min_match': 40, 'min_item_length': 2, 'vision_backup': True}
1. pattern: tuned. Values: {'patch_similarity': 0.82, 'max_drift_share': 0.9, 'min_patches': 20}
1. shape: tuned. Values: {'min_ordinal_steps': 3, 'min_changed_parts': 1, 'use_silhouette': False}
1. baseline 1 similarity threshold: 0.0000

## Cost

Vision calls run on a local Ollama model, so there is no API charge. Calls per image are in the tables.

## Scalio examples

No images in data/scalio_examples, so this part was not run.

## Failure gallery

10 missed drifts and 10 false alarms shown, worst first.

1. missed: synthetic/p015__pattern_blur, text_changed, pattern_changed. Image: gallery/missed_synthetic_p015__pattern_blur.png
1. missed: synthetic/p028__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p028__pattern_swap.png
1. missed: synthetic/p029__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p029__pattern_swap.png
1. missed: synthetic/p043__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p043__pattern_swap.png
1. missed: synthetic/p068__pattern_blur, text_changed, pattern_changed. Image: gallery/missed_synthetic_p068__pattern_blur.png
1. missed: synthetic/p068__pattern_swap, text_changed, pattern_changed. Image: gallery/missed_synthetic_p068__pattern_swap.png
1. missed: synthetic/p010__colour_shift, colour_changed. Image: gallery/missed_synthetic_p010__colour_shift.png
1. missed: synthetic/p010__pattern_blur, pattern_changed. Image: gallery/missed_synthetic_p010__pattern_blur.png
1. missed: synthetic/p010__pattern_swap, pattern_changed. Image: gallery/missed_synthetic_p010__pattern_swap.png
1. missed: synthetic/p010__product_erase, product_missing. Image: gallery/missed_synthetic_p010__product_erase.png
1. false alarm: real/p018__model_shot__gemini_app, shape_changed. Image: gallery/false_alarm_real_p018__model_shot__gemini_app.png
1. false alarm: real/p018__studio__gemini_app, shape_changed. Image: gallery/false_alarm_real_p018__studio__gemini_app.png
1. false alarm: real/p027__model_shot__gemini_app, text_changed. Image: gallery/false_alarm_real_p027__model_shot__gemini_app.png
1. false alarm: real/p029__festival_poster__gemini_app, logo_changed. Image: gallery/false_alarm_real_p029__festival_poster__gemini_app.png
1. false alarm: real/p029__instagram_post__gemini_app, logo_changed. Image: gallery/false_alarm_real_p029__instagram_post__gemini_app.png
1. false alarm: real/p068__festival_poster__gemini_app, product_missing. Image: gallery/false_alarm_real_p068__festival_poster__gemini_app.png
1. false alarm: real/p068__instagram_post__gemini_app, logo_changed. Image: gallery/false_alarm_real_p068__instagram_post__gemini_app.png
1. false alarm: real/p071__model_shot__gemini_app, shape_changed. Image: gallery/false_alarm_real_p071__model_shot__gemini_app.png
1. false alarm: real/p071__studio__gemini_app, shape_changed. Image: gallery/false_alarm_real_p071__studio__gemini_app.png
1. false alarm: real/p076__model_shot__gemini_app, logo_changed. Image: gallery/false_alarm_real_p076__model_shot__gemini_app.png
