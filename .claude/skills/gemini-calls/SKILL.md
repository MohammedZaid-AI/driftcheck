---
name: gemini-calls
description: Use whenever code calls the Gemini API or any other paid model API.
---
1. Read model names from config. Check Google's docs before adding a new model name.
2. Every call goes through the disk cache, keyed by a hash of the model name, prompt and image bytes.
3. Use structured output with a pydantic schema whenever a result is parsed.
4. Retry with backoff on rate limits and timeouts, at most 4 tries.
5. Log every real call with time taken, so the eval can report calls and cost per image.
6. If the API fails for good, the check returns unknown with a clear reason. Never crash the whole run.
7. In tests, mock the client.
