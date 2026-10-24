# DriftCheck rules
DriftCheck checks if an AI generated product image changed the original product.
Always:
1. Read PROGRESS.md at the start of every session.
2. Never invent, estimate or round up a result. Only report numbers the eval script measured.
3. Never commit .env, API keys or images in data/ except data/samples.
4. Tests never call real APIs. Use mocks.
5. Keep code simple and readable with type hints.
6. Use the skills in .claude/skills for their tasks.

# Setup decisions
1. Vision model calls go through driftcheck/vision. One setting, VISION_PROVIDER, picks the provider: ollama (default), gemini or hf. Never call a model SDK directly from a check.
2. Default model is qwen2.5vl:3b on local Ollama, which runs fully on the 6 GB RTX 3050 in about 15s per image. qwen2.5vl:7b is the option for the final overnight eval, at about 2 minutes per image.
3. Wherever the plan says Gemini, such as the ingest category, the product card, the shape and logo judge and baseline 2, use the selected vision provider.
4. Images are resized to a max side of 1024 before sending.
5. easyocr is the main source for text. The vision model is only the backup. For text and logos, crop to the product with rembg first and send a zoomed crop of each text or logo area. In a test, qwen2.5vl:3b got sleeves, neckline and colour right but put the text label in the wrong place.
6. No API image generation. Outputs come from a Kaggle notebook running Qwen Image Edit, or by hand from the free Gemini app. driftcheck register records them.
7. Windows: give commands that work in PowerShell. Use Python 3.12 in .venv and the CPU only build of torch. Ollama does the GPU work.
8. Phase 6 compares qwen2.5vl:3b and qwen2.5vl:7b on speed and accuracy.
