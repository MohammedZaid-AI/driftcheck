# DriftCheck progress

Goal: check if an AI generated product image changed the original product.

Status: phase 0 done, next is phase 1 (ingest, Kaggle notebook, manual route and register).

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

## Next

Phase 1: driftcheck ingest with the category from the vision provider, a Kaggle notebook with Qwen Image Edit, the manual Gemini app route, and driftcheck register.

## Commands

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[all]"
.venv\Scripts\driftcheck info
.venv\Scripts\driftcheck ping
.venv\Scripts\python -m pytest
```

Use Python 3.12. The default python on this machine is 3.14, which torch and easyocr may not support yet.

## Waiting on me

1. Put 30 to 40 product photos in data/inbox/.
2. Say what test.jpg in the project root is. It is not committed.

## GitHub

Remote origin is https://github.com/MohammedZaid-AI/driftcheck.git. Push after every phase.
