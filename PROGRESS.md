# DriftCheck progress

Goal: check if an AI generated product image changed the original product.

Status: phase 2 done. Next is phase 3 (product card). Phases 3 to 5 do not need real photos. Phase 6 needs labels.

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
2. sleeve_erase paints out the outer upper bands of the garment. On the shirt it also cuts the shoulders. The label shape_changed is right but the edit is cruder than a real AI change.

Fixes made on the way:
1. normalise() used to drop Hindi and Kannada vowel signs. It now keeps combining marks.
2. Model downloads failed with CERTIFICATE_VERIFY_FAILED because Python did not trust the Windows certificate store. The CLI now calls truststore at start, as pip does.

## Next

Phase 3: product card from the vision provider plus a pixel palette.

## Commands

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[all]"
.venv\Scripts\driftcheck info
.venv\Scripts\driftcheck ping
.venv\Scripts\driftcheck ingest
.venv\Scripts\driftcheck jobs --generator qwen_edit
.venv\Scripts\driftcheck jobs --generator gemini_app
.venv\Scripts\driftcheck register --from data\jobs\qwen_edit\result
.venv\Scripts\driftcheck label
.venv\Scripts\driftcheck split
.venv\Scripts\driftcheck synth
.venv\Scripts\python -m pytest
```

Use Python 3.12. The default python on this machine is 3.14, which torch and easyocr may not support yet.

## Waiting on me

1. Put 30 to 40 product photos in data/inbox/, then run driftcheck ingest.
2. Make outputs with the Kaggle notebook or the Gemini app, see docs/make_outputs.md, then run driftcheck register.
3. Run driftcheck split once all products are ingested, then driftcheck label and label every output. Phase 6 needs this.

## GitHub

Remote origin is https://github.com/MohammedZaid-AI/driftcheck.git. Push after every phase.
