# DriftCheck

DriftCheck checks if an AI generated product image changed the original product.

You give it the original product photo and an AI output. It reports what changed and where.

This project is under construction. Results will be added once the evaluation has run.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[all]"
copy .env.example .env
```

Then put your Gemini API key in `.env`.

## Commands

```
driftcheck info
pytest
```
