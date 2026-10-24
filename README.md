# DriftCheck

DriftCheck checks if an AI generated product image changed the original product.

You give it the original product photo and an AI output. It reports what changed and where.

This project is under construction. Results will be added once the evaluation has run.

## Setup on Windows

You need Python 3.12 and the Ollama app with qwen2.5vl:3b pulled. Run these in PowerShell from the project folder:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[all]"
copy .env.example .env
ollama pull qwen2.5vl:3b
```

Torch is the CPU only build. Ollama does the GPU work.

## Commands

```
.venv\Scripts\driftcheck info
.venv\Scripts\driftcheck ping
.venv\Scripts\python -m pytest
```

## Vision provider

Set VISION_PROVIDER in .env to ollama, gemini or hf. Gemini needs GEMINI_API_KEY. Hugging Face needs HF_TOKEN. Model names live in config/vision.json.
