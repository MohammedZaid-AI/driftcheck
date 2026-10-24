# DriftCheck progress

Goal: check if an AI generated product image changed the original product.

Status: phase 5 done. Stopped before phase 6, which needs labels. Phases 3 to 5 do not need real photos. Phase 6 needs labels.

## Plan changes from 2026-10-24

These replace the matching parts of the original plan. CLAUDE.md has the short version.

1. Vision model is local Ollama by default. qwen2.5vl:3b is the default. qwen2.5vl:7b is the option for the final overnight eval.
2. All vision calls go through one interface in driftcheck/vision with three providers picked by VISION_PROVIDER: ollama, gemini and hf. Disk cache, retries and call logging apply to every provider.
3. Images are resized to a max side of 1024. For text and logos, crop to the product with rembg and send zoomed crops. easyocr is the main text source, the vision model is the backup. Build this in phases 3 and 4.
4. No API image generation. Phase 1 gives two routes: a Kaggle notebook with Qwen Image Edit, and a manual route with the free Gemini app. driftcheck register records both. The generate command is dropped.
5. Wherever the plan says Gemini, use the selected vision provider: ingest category, product card, shape and logo judge, baseline 2.
6. GEMINI_API_KEY is optional.
7. Windows and PowerShell commands. CPU only torch.
8. Phase 6 compares qwen2.5vl:3b and qwen2.5vl:7b on speed and accuracy.

## Done

Phase 0: setup
1. Package layout in driftcheck/ with pyproject.toml. Extras: vision, app, dev, all.
2. Settings from .env and config/vision.json in driftcheck/config.py.
3. Logging in driftcheck/log.py.
4. Disk cache in driftcheck/cache.py, under cache/.
5. Retry with backoff in driftcheck/retry.py. At most 4 tries on 408, 429, 5xx, timeouts and connection errors.
6. Vision interface in driftcheck/vision. VisionClient.ask(prompt, images, schema) resizes images, checks the provider once, uses the cache, retries, parses into the pydantic schema and logs every real call to cache/vision_calls.jsonl. Providers: ollama_provider.py, gemini_provider.py, hf_provider.py. A failed call raises VisionError so a check can return unknown.
7. CLI: info and ping work. ingest, register, synth, check and eval are stubs that name their phase.
8. Tests mock every provider. No test calls a real model.

Models in config/vision.json:
1. ollama: qwen2.5vl:3b, compare with qwen2.5vl:7b. Both are pulled on this machine.
2. gemini: gemini-3.8-flash, from https://ai.google.dev/gemini-api/docs/models on 2026-10-24.
3. hf: Qwen/Qwen2.5-VL-3B-Instruct, from the Hugging Face image text to text docs on 2026-10-24.

Phase 1: ingest and outputs
1. driftcheck ingest moves photos from data/inbox to data/products/pNNN/original.jpg, capped at 2048 px, and writes meta.json. It skips duplicate photos by SHA256. The vision provider fills category and product_type. If that fails, the category stays null and the next ingest retries it. --no-vision only moves photos.
2. driftcheck jobs --generator NAME writes data/jobs/NAME/ with jobs.json, kaggle_input.zip and manual_checklist.md. Prompts in driftcheck/prompts.py fit the category: wearables go on a model, other goods go on a table or in a hand.
3. notebooks/kaggle_qwen_image_edit.ipynb runs Qwen Image Edit 2511 in a 4 bit build (ovedrive/Qwen-Image-Edit-2511-4bit) with CPU offload. It has a smoke test, skips done jobs, rejects black images and writes json with model, steps, seed and time. NOT YET RUN ON KAGGLE. bf16 on a T4 is the main risk.
4. Manual route in docs/make_outputs.md.
5. driftcheck register records outputs named <style>__<generator>.jpg, converts png and webp, keeps the notebook json, writes json for manual outputs, and lists files with bad names. --from imports an unzipped Kaggle result.

Phase 2: labels, split and synthetic drift
1. data/samples/plain_shirt.jpg is the plain control example and a test fixture. It is the 333 by 500 photo from test.jpg, already under 800 px so not resized.
2. driftcheck label opens a Gradio app on port 7861: original and output side by side, yes, no or n/a for each attribute, notes, back and next, progress count. Every answer is saved to data/labels.json straight away. A missing key means not labelled yet, null means not applicable. Logic is in LabelSession so it is tested without the UI.
3. driftcheck split writes data/split.json, 50 50 by product, dealt out per category so both halves get a similar mix. It refuses to overwrite without --force.
4. driftcheck synth writes data/synthetic/<product_id>/<drift>.jpg and .json plus data/synthetic/labels.json. Drifts: colour_shift (Lab shift inside the mask), text_erase, text_replace (latin text only), pattern_blur and pattern_swap (only when texture score is at least 8), sleeve_erase (clothing only), product_erase, and three controls with no change: control_light, control_jpeg, control_shift. logo_changed is always null in this set.
5. driftcheck/segment.py: rembg isnet-general-use masks, cached in cache/masks. driftcheck/ocr.py: easyocr with an English and Hindi reader and an English and Kannada reader, merged, cached in cache/ocr.
6. Ran synth for real on the shirt sample: 6 outputs, checked by eye.

Known limits of synth:
1. rembg counts the hanger as product, so colour_shift recolours the hanger too.
2. sleeve_erase paints straight bands, so the edges are cruder than a real AI change and a thin outline of the old sleeve can remain.

Fixes made on the way:
1. normalise() used to drop Hindi and Kannada vowel signs. It now keeps combining marks.
2. Model downloads failed with CERTIFICATE_VERIFY_FAILED because Python did not trust the Windows certificate store. The CLI now calls truststore at start, as pip does.

Phase 3: product card
1. driftcheck/card.py builds a ProductCard: VisionCard from the vision provider (product type, category, views, sleeve length, neckline, silhouette, length, closures, colours, pattern, embellishments, logos with position, text items with language and position, key features), easyocr text with boxes, the pixel palette, the product box and product share.
2. The vision call gets the rembg product crop plus up to 3 zoomed crops of OCR text areas.
3. driftcheck/palette.py: k means with k=5 in Lab on the eroded product mask, clusters under 2 percent dropped. grey_world() is the optional white balance for the colour check.
4. Cards are cached in cache/cards. A card whose vision call failed is not cached, so the next run retries.
5. driftcheck card PATH prints the card as JSON.
6. Real run on the shirt sample with qwen2.5vl:3b: all fields right after the fix below, 8.7s with warm models.

Fix made on the way: with only Ollama's format setting, qwen2.5vl:3b said the shirt was footwear with a cotton neckline and a zip. Constrained decoding fixes the JSON shape but the model never sees field descriptions or allowed values. The Ollama and Hugging Face providers now also put the schema in the prompt. Providers carry a version that is part of the cache key, so this change did not reuse old answers.

Phase 4: checks
Each check is its own module in driftcheck/checks and returns a CheckResult: name, status pass, fail or unknown, score, threshold, reason, details, applicable. Thresholds come from config/thresholds.json, which says tuned: false. All values there are starting values, not tuned. Phase 6 tunes them on the calibration split only.
1. segmentation.py, check product_missing: rembg product share of the output. Fail below 1 percent, unknown below 3 percent. If it does not pass, every other check is skipped as unknown.
2. colour.py: k means Lab palettes of both products after grey world white balance, each original colour matched to its nearest output colour with CIEDE2000, weighted by share. Extra output colours such as skin do not count.
3. text.py: easyocr on the output, rapidfuzz match per original text item, partial match on joined text for long items. Items OCR misses go to the vision model as a backup reader with zoomed crops. The model only reads, the match is done in code. No text on the original gives unknown with applicable false.
4. pattern.py: DINOv2 small patch features on 448 px square crops, 32 by 32 patches. For each output product patch, best cosine similarity to any original patch. Score is the share of patches below 0.85. The similarity grid is saved for the heatmap. Product under min_patches times 196 real pixels gives unknown.
5. judge.py, checks shape and logo: the vision model describes the original and the output separately with fixed choices for sleeves, neckline, length and silhouette, plus whether a logo is visible. Code compares. Sleeves and length must move at least 2 steps. Silhouette is off. Logo fails when the original has one and the output shows none.
6. verdict.py: any fail means drifted, all applicable checks pass means faithful, anything else is unknown. Checks with applicable false, like logo on a product without a logo, are left out of the verdict, matching a null label.
7. pipeline.py: Checker runs everything on one pair and returns a Report with the card, every check, the verdict, time per step, real vision calls and model names.

Starting threshold values, set from synthetic checks on the shirt sample on 2026-10-24. These are not results:
1. colour 7.0: a Lab shift of about 35 scored 8 to 12 CIEDE2000, warm lighting about 5. 12 missed every synthetic colour shift.
2. pattern patch similarity 0.85: with DINOv2 small, controls on the shirt scored 0.97 to 0.99 median similarity, synthetic pattern blur, swap and dots 0.75 to 0.89. At 0.55 nothing was flagged. At 0.85 all 5 shirt variants passed and all 3 pattern drifts failed.
3. judge: on the shirt sample, qwen2.5vl:3b read full sleeves as three quarter every time and missed the synthetic full to short sleeve change, with no false alarms once silhouette was off. Silhouette flipped between fitted and straight on 2 of 3 controls. qwen2.5vl:7b read the sleeves right, caught the change and had no false alarms, at about 8 to 25 s per image. One sample only. Phase 6 measures this properly.

Design changes from the plan, with reasons:
1. The judge no longer asks the model to compare two images. Asked directly, qwen2.5vl:3b called a colour shift a change of all four shape parts, missed the sleeve change, and gave 0.9 confidence to every answer. Describing each image with the same prompt and comparing in code had no false alarms on the controls.
2. Confidence is not used for shape because the 3b model gave the same confidence to every answer. The thresholds are step counts instead.
3. Checks that do not apply are marked applicable false so the verdict can still be faithful for products with no logo or no text.
4. synth sleeve_erase now cuts long sleeves short, by painting out the outer bands below the upper arm. The first version cut out the upper sleeves and shoulders, which left floating cuffs.

Phase 5: heatmap and report
1. driftcheck check --original A --output B --out DIR writes report.json and heatmap.png. --model picks another vision model, for example qwen2.5vl:7b. If Ollama is down it still runs, with the vision parts unknown.
2. heatmap.png: verdict and every check status on top, the original with each original text item boxed green if found and red if missing, the output with patches that have no close pattern match painted red, and a strip pairing each original colour with its matched output colour and the CIEDE2000 distance. There is no pixel by pixel colour map because the photos are not aligned.
3. Real runs on the shirt sample, all real models, warm caches: colour_shift gave drifted, colour 19.2 CIEDE2000 against a limit of 7, 44.3s with 3 vision calls. control_light gave faithful, colour 1.8, 18.8s with 1 vision call. These are two smoke tests, not results.

Real data, 2026-10-24
1. 77 photos in data/inbox: 74 jpg, 1 avif, 1 png, 1 webp. Ingest now reads avif, flattens transparent backgrounds onto white, skips blank images, and moves each source to data/inbox/done/ instead of deleting it.
2. download.png became an all black p063. The first ingest had already deleted the source, so it could not be checked or redone. p063 was removed. 76 products remain, p001 to p077 without p063.
3. meta.json has worn: true when a person wears the product. The first prompt marked hands holding bottles and packs as worn. The fixed prompt plus a rule that only wearable categories can be worn left 7 worn, all clothing: p001, p010, p018, p024, p047, p060, p075. By eye, p001, p010, p024 and p060 are right, p075 is a flat lay, and p018 and p047 are close ups where no person is clearly visible.
4. Six photos are shop fronts or sign boards, not products: p004, p029, p055, p062, p068, p074. The model filed them under packaged_food, other, jewellery and accessory. There is no storefront category yet.

Core set, 2026-10-24
1. config/selection.json holds the hand fixes and the 40 core products with a reason each. driftcheck select applies it to meta.json. Fields set by hand are listed in manual_fields and ingest --refresh never overwrites them.
2. Worn means a person's body is visible wearing the item. p018, p047 and p075 set to not worn by hand. Worn now: p001, p010, p024, p060, all core.
3. New category storefront for p004, p029, p055, p062, p068, p074. Their styles are festival_poster, instagram_post and reel_cover, and the prompt tells the generator to keep the shop name exact in its own script.
4. Core mix: 22 clothing, 8 packaged food, 4 beauty, 2 accessory, 4 storefront. The other 36 are tagged extra.
5. The clothing rule asked for all sleeveless (6), all embroidered garments (7), all with a logo or number (12), 4 worn and 3 plain. Those add up to 28 distinct items, more than 22. Kept all sleeveless, all embroidered, all 4 worn and 3 plain (p031, p076, p077, which are also sleeveless), and 6 of the 12 logo or number items. Left out: p014, p022, p025, p050, p072, p073.
6. jobs and split use core products by default. --include-extra adds the rest. 120 jobs for 40 products.
7. download.png is skipped for good.
8. Swaps on 2026-10-24: p052 out because it shows two jerseys, p025 in. p047 out, p022 in, p018 kept. p062 out because it is not an Indian shop, p068 in, an Indian tea shop with a Devanagari name. Logo or number items in core are now p008, p022, p025, p027, p041, p059, p067.
9. easyocr now runs a third reader, English with Bengali, so every OCR call does three passes. On p045 it read রোস্ট, পাউডার and ইফাদ at 0.80 to 0.94 confidence, but not the large title রসুই. The Hindi reader also reads parts of the Bengali text as made up Devanagari words at 0.55 to 0.78 confidence, which become extra text items for that product.

## Next

Phase 6: baselines and evaluation. Needs labels.

## Commands

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[all]"
.venv\Scripts\driftcheck info
.venv\Scripts\driftcheck ping
.venv\Scripts\driftcheck ingest
.venv\Scripts\driftcheck select
.venv\Scripts\driftcheck jobs --generator qwen_edit
.venv\Scripts\driftcheck jobs --generator gemini_app
.venv\Scripts\driftcheck register --from data\jobs\qwen_edit\result
.venv\Scripts\driftcheck label
.venv\Scripts\driftcheck split
.venv\Scripts\driftcheck synth
.venv\Scripts\driftcheck card data\samples\plain_shirt.jpg
.venv\Scripts\driftcheck check --original data\samples\plain_shirt.jpg --output OUTPUT.jpg --out reports\check
.venv\Scripts\python -m pytest
```

Use Python 3.12. The default python on this machine is 3.14, which torch and easyocr may not support yet.

## Waiting on me

1. Done: photos ingested and the core set chosen.
2. Make outputs with the Kaggle notebook or the Gemini app, see docs/make_outputs.md, then run driftcheck register.
3. Run driftcheck split once all products are ingested, then driftcheck label and label every output. Phase 6 needs this.

## GitHub

Remote origin is https://github.com/MohammedZaid-AI/driftcheck.git. Push after every phase.
