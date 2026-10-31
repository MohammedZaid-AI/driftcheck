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
9. OCR by script, 2026-10-24. driftcheck scripts runs every reader once on each core original and saves text_scripts in meta.json. Checks then run English plus only that product's scripts, on both images, and drop readings in any other script. Readers are cached one by one, so detection and checks share results.
10. The rule from the plan was: keep a script if its reader finds text at 0.7 or more. Alone it kept Hindi on p045, because the Hindi reader read Bengali as a made up word at 0.78. Added: a script must also win at least half as many boxes as the strongest script. On p045 Bengali wins 4 boxes and Hindi 1, so Hindi is dropped.
11. Detection was still wrong on 5 of 40 core products: p017, p043, p049, p057, p068. easyocr reads real Hindi at 0.36 to 0.57 confidence, so it never reaches 0.7, and on p049 the Bengali reader read Hindi as made up Bengali at 0.74. A second rule tried, summing confidence per script, did not separate them either. These 5 are set by hand in config/selection.json, which wins over detection.
12. Unsure text boxes at least 3 percent of the image tall are read again from a 2x crop. The reading that overlaps the box most replaces the first one if it is more confident. The first version took the most confident reading in the crop, which was a neighbouring word. On p045 the title went from রস্ুই at 0.3 to বসুই at 0.43, still one letter wrong. Logged in docs/limitations.md and left there.
13. selection.json now refuses a product named twice. A duplicate p068 entry had silently dropped its storefront category.

Outputs, 2026-10-25
1. Kaggle failed. The 4 bit Qwen Image Edit 2511 loaded across both T4s, 11.8 GB on GPU 0 and 5.1 GB on GPU 1, and every generation ran out of memory on GPU 0, even at 768 by 768. No outputs made. The notebook is kept unchanged. Logged in docs/limitations.md.
2. Switched to the Gemini app route. driftcheck jobs now defaults to generator gemini_app and two styles per product: model_shot and studio for wearables, lifestyle and studio for other goods, festival_poster and instagram_post for storefronts, which have no studio shot. --all-styles gives three. 80 jobs for the 40 core products.
3. data/jobs/gemini_app/manual_checklist.md has one line per image with a tick box, the photo path, the exact prompt and the exact file name to save, all relative to the project folder.
4. Five product types from the vision model would have put wrong details into the prompts and could cause drift the test then blames on Gemini. Fixed by hand in config/selection.json: p001 sleeveless printed kurti, was embroidered; p060 elephant block print kurti, was embroidered; p071 sleeveless printed strappy dress, was embroidered silk; p025 collared sweatshirt, was hooded; p057 chatpati candy pack, was masala spice pack.

Gemini outputs placed, 2026-10-30
1. The user made the outputs in the Gemini app, model 3.6 Flash, and saved them into the product outputs folders under Gemini's download names.
2. 72 were matched to checklist targets by download time within their folder. Download times ran in checklist order with no breaks. p013, p060 and p067 had one file each, a model shot by eye. p068 had one Gemini file, the Diwali festival poster by eye.
3. Each was copied to its checklist name as jpg: 76 outputs. The Gemini pngs were then moved to data/gemini_downloads/<product>/. A spare copy of a p028 file in the project root went to data/gemini_downloads. Log: data/gemini_downloads/placement_log.json.
4. p068 also had a ChatGPT image. It was moved out and is not in the test set, so every output is from one generator.
5. p028 lifestyle came back on a plain white background, so it looks like a second studio shot.
6. Not registered yet. 4 targets are still to be made by the user in Gemini: studio for p013, p060 and p067, and instagram_post for p068.

Phase 6 so far, 2026-10-30
1. All 80 outputs registered and labelled. The user labelled every attribute on all 80 as not changed. p031 studio shape_changed was set to no by Claude at the user's request. So the real set can measure wrong flags, unknown rate and speed, but not recall. Recall comes from the synthetic set only.
2. driftcheck synth made 326 synthetic outputs for the 40 core products.
3. Code: driftcheck/baselines.py, driftcheck/evaluate.py, driftcheck/eval_report.py. driftcheck eval-run runs every pair once with recording thresholds and saves raw results to cache/eval/<model>/. driftcheck eval tunes on the calibration split, freezes config/thresholds.json and writes reports/eval_report.md, reports/eval_report.json, reports/tuning.json and the gallery in reports/gallery, which is gitignored because it shows the user's photos.
4. Ollama problems found while running qwen2.5vl:3b, all fixed in driftcheck/vision/ollama_provider.py: the default 4096 context cut off prompts with several images, now 16384; llama.cpp fails schema decoding with "empty grammar stack", sometimes as an error and sometimes as an empty reply, both now fall back to plain JSON mode; once the model generated until the context was full, so answers are capped at 1024 tokens. Even so, the model still failed on long card prompts and hit Ollama's token repeat limit: 8 failed calls in the first 4 pairs.
5. The python process also crashed twice with a Windows access violation, 0xC0000005, in native code, once at pair 23 and once at pair 50 of earlier attempts. eval-run now logs each pair before it starts, so a crash shows where it was.
6. The user stopped the qwen2.5vl:3b run and chose Groq. driftcheck/vision/groq_provider.py talks to Groq's OpenAI compatible API with httpx: JSON schema output with a fallback to JSON mode, waits on 429 as long as Groq asks, answers capped at 1024 tokens. The user picked qwen/qwen3.8-27b as the Groq model, set in config/vision.json. To be confirmed against Groq's /models list, and that it takes images, once the key is set.
7. A bug in a new CLI test ran driftcheck eval against the real config and overwrote config/thresholds.json with untuned values marked tuned. Restored from git, the test fixed, and eval now refuses to tune without calibration and test pairs.

8. Groq free tier: qwen/qwen3.8-27b answered correctly in 0.6s, but the free tier allows 200000 tokens per day for this model, shown only in the 429 error, not in headers. The full run needs about 2 million tokens. Abandoned for the evaluation. The Groq provider stays in the code.
9. The evaluation ran on local qwen2.5vl:7b through Ollama, chosen by the user, with VISION_PROVIDER=ollama and VISION_MODEL=qwen2.5vl:7b set for the run only. .env still says groq.
10. Problems fixed during the 7b run: Python ran out of RAM on the 12 GB laptop, so images are capped at 2048 px when checks load them and Ollama uses num_ctx 8192 with at most 3 images per call. easyocr's detector still crashed with a Windows access violation in a torch convolution, but only while the 7b model was loaded; alone it ran 15 times on the same image without a crash. The detector canvas is capped at 1600 px, and eval-run restarts after a crash and warms models up before timing. Final run: 406 pairs, 872 vision calls, 0 failed calls, 11 crashes restarted automatically, 00:25 to 04:09.

Phase 6: done, 2026-10-31
1. Tuned on the calibration split, 207 pairs, then frozen in config/thresholds.json: segmentation missing_share 0.02, colour 11.75, text min_match 80 kept, pattern patch_similarity 0.78 and max_drift_share 0.45, judge min_ordinal_steps 3, baseline 1 similarity 0.578. reports/tuning.json has the calibration scores.
2. reports/eval_report.md and .json, from commit 96cf374. Headline numbers on the test split, measured:
   a. Real Gemini outputs, 40 images, all labelled unchanged: DriftCheck flagged 27 as drifted, a wrong flag rate of 67.5%. Baseline 1 25.0%, baseline 2 0.0%. Per attribute wrong flag rates: pattern 42.5%, text 22.9%, logo 14.3%, colour 12.5%, shape 10.0%, product missing 2.5%.
   b. Synthetic set, 159 images, 99 drifted: DriftCheck recall 80.8%, precision 74.8%, wrong flag rate 45.0% on the 60 no change images. Baseline 1 recall 2.0%, baseline 2 recall 5.1%, both 0.0% wrong flags. Per attribute recall: text 84.4%, pattern 73.3%, colour 65.0%, product missing 10.0%, shape 9.1%.
   c. Time per image: DriftCheck median 51.2s and p90 145.8s on real outputs, median 17.7s on synthetic. Baseline 1 about 0.5s, baseline 2 about 7s. 2.42 vision calls per real image.
3. Not done from the plan: the qwen2.5vl:3b against 7b comparison. The 3b run was stopped after it failed many vision calls. Only 7b results exist.
4. Main reasons for false alarms, from the gallery: in model shots and scenes, rembg cuts out the whole person, so colour and pattern compare the garment with the model's face, skin and jeans. A blurred product can also fall below the product missing share and skip every other check.
5. A weakness in the tuning method: the 10% wrong flag cap was measured on all calibration negatives, mostly synthetic. It held there but not on real outputs. Measuring the cap on real calibration images only would fit the plan better. Not changed, because the test results are already seen; this is the user's call.
6. The failure gallery in reports/gallery is gitignored because it shows the user's photos.

Phase 6 follow up, cut out fix and retuning, 2026-10-31
1. driftcheck/garment.py: for clothing and accessories, mattmdjaga/segformer_b2_clothes, a local clothes parsing model, keeps only the garment in any image with a person in it. Classes are grouped, top or dress, bottom, scarf and so on, and the biggest group on the original is the product. Accessories keep scarf, hat, bag and belt classes. Images without a person keep rembg. If the garment cannot be found, rembg is kept and the reason is recorded. The model's licence is listed as other. Before and after pictures were checked by eye for p001, p010, p018, p022, p031, p038, p060, p067.
2. driftcheck eval-recut ran product missing, colour and pattern again with the new cut out on the saved 7b run, saved as cache/eval/qwen2.5vl_7b_garmentcut. Text, shape and logo come from the saved run. Time per image swaps in the new timing for those checks plus the parsing. The garment cut out was used on 30 of 80 real and 55 of 326 synthetic pairs. No crashes.
3. Pattern similarity on worn images rose a lot with the new cut out, for example p031 from 0.32 to 0.58 and p067 from 0.43 to 0.73, but many still sit below the pattern cut off: the same garment on a body still looks different to DINOv2 than lying flat.
4. Retuned, as the user asked, with the 10 percent wrong flag cap measured on the 40 real calibration images only. Catches still come from the synthetic calibration changes. With that cap nothing in the first search ranges worked for pattern and text, so the ranges were widened: pattern patch similarity 0.40 to 0.94 and share up to 0.95, text match 30 to 95. Both changes were made after the first test results were seen, and the report says so.
5. Frozen thresholds now: colour 15.01, pattern patch 0.82 and share 0.90, text 40, product missing 0.02, shape steps 3, baseline 1 similarity 0.0, which means baseline 1 never flags. The first frozen values are kept in reports/thresholds_before_cutout.json, the first report in reports/eval_report_before_cutout.md and .json.
6. Measured on the test split, before, then new cut out with the old rule, then after:
   a. Real test images wrongly flagged: 27 of 40, 26 of 40, 20 of 40. Overall wrong flag rate 67.5%, 65.0%, 50.0%.
   b. Real test wrong flag rate per attribute after: pattern 12.5% from 42.5%, text 14.3% from 22.9%, logo 14.3% unchanged, colour 7.5% from 12.5%, shape 10.0% unchanged, product missing 2.5% unchanged.
   c. Synthetic test overall recall: 80.8%, 71.7%, 60.6%. Wrong flag rate on unchanged synthetic copies: 45.0%, 31.7%, 30.0%.
   d. Synthetic recall per attribute after: text 59.4% from 84.4%, pattern 43.3% from 73.3%, colour 35.0% from 65.0%, product missing 10.0% and shape 9.1% unchanged.
7. So the fixes cut false alarms on real images from 27 to 20 of 40, but half the real images are still wrongly flagged, and fewer real changes are caught. Logo and shape come from the vision model and were not changed by the cut out.
8. Still not done from the plan: the qwen2.5vl:3b against 7b comparison.

Phase 6 follow up 2, routing by image type, 2026-10-31
1. Asked by the user. Each output gets an image type: studio is flat; model_shot, lifestyle, festival_poster and instagram_post are scene. A synthetic copy takes its original's type: scene when the original is worn, which is p001, p010, p024 and p060.
2. On scene images colour and pattern are skipped. In driftcheck check they come back as unknown with applicable false and the reason. In the evaluation a skipped check is left out of the verdict and out of that check's counts, so it is neither a pass nor a fail. A colour or pattern change on a scene image counts as missed overall unless another check catches it.
3. No retune: thresholds are the ones frozen at the previous retune. The saved runs were decided again, nothing was rerun.
4. Image types: real test 18 flat and 22 scene, real calibration 18 and 22, synthetic test 143 and 16, synthetic calibration 151 and 16.
5. Measured on the test split, first report, then cut out fix and retune, then routing:
   a. Real test images wrongly flagged: 27 of 40, 20 of 40, 15 of 40. Overall wrong flag rate 67.5%, 50.0%, 37.5%.
   b. Real wrong flag rate per check with routing: colour 0.0% and pattern 0.0%, both on the 18 flat images only. Logo 14.3% on 35, text 14.3% on 35, shape 10.0% on 40, product missing 2.5% on 40.
   c. Synthetic test changes caught: 80 of 99, 60 of 99, 58 of 99. Recall 80.8%, 60.6%, 58.6%. Wrong flag rate on unchanged synthetic copies 45.0%, 30.0%, 30.0%.
   d. Synthetic recall per check with routing: text 59.4%, pattern 46.2% on 26 changes, colour 33.3% on 18 changes, product missing 10.0%, shape 9.1%.
6. The 15 real false alarms left come from logo, text, shape and product missing. Logo and shape are the vision model's describe and compare.
7. Previous report kept as reports/eval_report_before_routing.md and .json. The table at the top of reports/eval_report.md has all three columns.

Phase 7: Gradio demo, 2026-10-31
1. driftcheck demo opens driftcheck/apps/demo_app.py on port 7860. It always uses local Ollama, default qwen2.5vl:7b, set for that run only; .env is not changed.
2. Two uploads, original and generated, and an image type dropdown: auto, flat, scene. auto reads the style from a project file name like studio__gemini_app.jpg; an upload has none, so auto means scene.
3. It calls Checker.check, the same pipeline as the evaluation, with the garment cut out, routing and the frozen thresholds. The page shows the verdict in words, which checks fired, each check as fine, changed, not sure or not applicable with its reason, the text read in each image with the match scores, and the heatmap when the pattern check ran.
4. Three examples from the test split, picked with the saved evaluation and then run live to confirm: a clean pass, p031 studio; a colour change caught by colour, synthetic p041 colour_shift; a pattern change caught by pattern, synthetic p022 pattern_swap. The real outputs have no labelled changes, so both changed examples are synthetic, and the page says so. Examples only show when data/ has the files.
5. Checked: the app starts and serves its page.

Text decision fix in the evaluation, found while building the demo
1. Live, a text item OCR misses goes to the vision model and counts as found if its reading matches at the text threshold. The evaluation only used vision readings that matched at 95, the recording threshold, so at the tuned 40 it marked items missing that the live check finds. Example: p067 text_replace, where the vision model read NYc as NK, a score of 40.
2. driftcheck/evaluate.py now uses the saved vision readings like the live check. Thresholds unchanged. The report adds a fourth column.
3. Measured on the test split with the fix: real test images wrongly flagged 11 of 40, from 15 of 40. Text wrong flag rate 2.9%, from 14.3%. Synthetic changes caught 57 of 99, from 58. Synthetic text recall 56.2%, from 59.4%.
4. With a text threshold of 40, letter swaps such as BROOKLYN to UROKKLYH still count as a match, so the text check alone caught none of the synthetic text replacements in the test split. Worth a look before Phase 9.
5. Previous report kept as reports/eval_report_before_textfix.md and .json.

Phase 8, as the user named it: README and demo walkthrough, 2026-10-31
1. README.md rewritten for a reader at Scalio: the problem with links to the Play Store, App Store and refund policy pages from the project brief, what DriftCheck does, how it works with the five checks and routing, how to run the CLI and the demo, results, limitations and next steps.
2. Every number in README.md comes from reports/eval_report.json, including the first report numbers, which are in its comparison section. Written with the write-docs skill rules.
3. Results as stated: 11 of 40 real test images wrongly flagged, 57 of 99 synthetic changes caught, 13 of 60 unchanged synthetic copies wrongly flagged, precision 81.4%, the table per check, baselines, and speed.
4. The honesty note says all tuning and every change happened after the first test results were seen.
5. Next steps lead with the strict text compare: compare strictly when OCR reads a word clearly in both images, keep the loose match only when OCR is unsure.
6. docs/demo.md: a five minute walkthrough of the three examples and what to point at.
7. The demo example label now says striped tee instead of t-shirt, to keep hyphens out of the docs.

Note on phase numbers: the original plan had Phase 8 as the FastAPI service and Phase 9 as docs and the pitch. The user called this README work Phase 8. The FastAPI service, the Dockerfile, docs/demo_script.md, docs/message_to_scalio.md and the v0.1 tag are not done yet.

v0.1, tagged on commit c42591f on 2026-10-31: the first working version with the demo and an honest evaluation.

## Next

1. User reviews README.md and docs/demo.md.
2. Not done, deferred: the FastAPI service with POST /check and GET /health, and its Dockerfile.
3. Not done from the original plan: docs/demo_script.md and docs/message_to_scalio.md.

## Commands

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[all]"
.venv\Scripts\driftcheck info
.venv\Scripts\driftcheck ping
.venv\Scripts\driftcheck ingest
.venv\Scripts\driftcheck select
.venv\Scripts\driftcheck scripts
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
2. Make the last 4 outputs in Gemini: studio for p013, p060, p067 and instagram_post for p068, saved under their checklist names. Then run driftcheck register.
3. Run driftcheck split once all products are ingested, then driftcheck label and label every output. Phase 6 needs this.

## GitHub

Remote origin is https://github.com/MohammedZaid-AI/driftcheck.git. Push after every phase.
