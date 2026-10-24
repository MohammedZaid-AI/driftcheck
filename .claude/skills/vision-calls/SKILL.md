---
name: vision-calls
description: Use whenever code calls a vision model, local Ollama, Gemini, Hugging Face or any other model API.
---
1. Every call goes through VisionClient in driftcheck/vision. Never call the ollama, google-genai or huggingface_hub SDK from a check.
2. Read provider and model names from config/vision.json or VISION_PROVIDER and VISION_MODEL. Check the provider's docs before adding a new model name.
3. Every call goes through the disk cache, keyed by a hash of the provider, model name, prompt, schema and image bytes.
4. Use structured output with a pydantic schema whenever a result is parsed.
5. Images are resized to a max side of 1024. For text and logos, crop to the product first and send a zoomed crop. easyocr is the main text source and the model is the backup.
6. Retry with backoff on rate limits and timeouts, at most 4 tries.
7. Log every real call with time taken, so the eval can report calls, time and cost per image.
8. If the provider fails for good, the check returns unknown with a clear reason. Never crash the whole run.
9. In tests, mock the provider.
