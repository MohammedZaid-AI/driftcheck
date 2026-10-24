# DriftCheck progress

Goal: check if an AI generated product image changed the original product.

Status: phase 0 done, next is phase 1 (ingest and generate).

## Done

Phase 0: setup
1. Package layout in driftcheck/ with pyproject.toml. Core deps install by default. Extras: vision, app, dev, all.
2. Settings from .env and config/models.json in driftcheck/config.py.
3. Logging in driftcheck/log.py.
4. Disk cache in driftcheck/cache.py. Keys are a SHA256 of model name, prompt and image bytes. Stored under cache/.
5. Retry with backoff in driftcheck/retry.py. At most 4 tries on 408, 429, 5xx, timeouts and connection errors.
6. CLI skeleton in driftcheck/cli.py. Commands info works. ingest, generate, register, synth, check and eval are stubs that name their phase.
7. pytest setup with 16 tests. Tests blank out GEMINI_API_KEY.

Model names in config/models.json, taken from https://ai.google.dev/gemini-api/docs/models on 2026-10-24:
1. vision: gemini-3.8-flash
2. fast: gemini-3.5-flash-lite
3. image: gemini-3.1-flash-image

## Next

Phase 1: driftcheck ingest, driftcheck generate and driftcheck register.

## Commands

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[all]"
.venv\Scripts\driftcheck info
.venv\Scripts\python -m pytest
```

Use Python 3.12. The default python on this machine is 3.14, which torch and easyocr may not support yet.

## Waiting on me

1. Copy .env.example to .env and add GEMINI_API_KEY.
2. Put 30 to 40 product photos in data/inbox/.
3. Decide on a public GitHub repo called driftcheck. The gh CLI is not installed on this machine yet.
