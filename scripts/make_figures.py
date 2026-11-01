"""Run DriftCheck on every pair in examples/ and save one result figure per pair in docs/images/.

Uses the full pipeline with local Ollama, qwen2.5vl:7b, and the frozen thresholds.
Writes examples/results.json with each verdict, so the README can be checked against it.

    .venv\\Scripts\\python scripts\\make_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["VISION_PROVIDER"] = "ollama"  # this run only; .env is not changed
os.environ.setdefault("VISION_MODEL", "qwen2.5vl:7b")
from driftcheck.config import load_settings, use_system_certificates  # noqa: E402

FIGURES = ROOT / "docs" / "images"


def caption(width: int, lines: list[str]) -> Image.Image:
    bar = Image.new("RGB", (width, 18 + 28 * len(lines)), (245, 245, 242))
    draw = ImageDraw.Draw(bar)
    for i, line in enumerate(lines):
        draw.text((12, 10 + 28 * i), line, fill=(20, 20, 20), font=ImageFont.load_default(size=20 if i == 0 else 17))
    return bar


def figure(checker, title: str, original: Path, generated: Path, image_type: str, out: Path) -> dict:
    """Run DriftCheck on one pair, save the figure to out and return what it found."""
    from driftcheck.apps.demo_app import CHECK_NAMES, VERDICT_WORDS
    from driftcheck.heatmap import draw_heatmap
    from driftcheck.pipeline import load_image

    report = checker.check(original, generated, image_type=image_type)
    fired = [CHECK_NAMES[c.name] for c in report.checks if c.status == "fail"]
    verdict = VERDICT_WORDS[report.verdict]
    heatmap = draw_heatmap(load_image(original), load_image(generated), report, output_label="Edited image")
    top = caption(heatmap.width, [
        title,
        f"DriftCheck says: {verdict}. Checks that fired: {', '.join(fired) if fired else 'none'}.",
        "No edit, only saved again, to show a clean pass. Not a real AI or Scalio output." if title.startswith("Unchanged")
        else "Edited on purpose to show how DriftCheck responds. Not a real AI or Scalio output.",
    ])
    picture = Image.new("RGB", (heatmap.width, top.height + heatmap.height), "white")
    picture.paste(top, (0, 0))
    picture.paste(heatmap, (0, top.height))
    out.parent.mkdir(parents=True, exist_ok=True)
    picture.save(out)
    print(f"{out.name}: {verdict}, fired {fired}")
    return {"title": title, "verdict": verdict, "fired": fired, "image_type": image_type,
            "checks": {CHECK_NAMES[c.name]: [c.status, c.applicable, c.reason] for c in report.checks},
            "model": os.environ["VISION_MODEL"]}


def main() -> None:
    use_system_certificates()
    from driftcheck.apps.demo_app import examples
    from driftcheck.pipeline import Checker

    checker = Checker.from_settings(load_settings())
    results = []
    for example in examples():
        name = f"example_{example.slug}.png"
        found = figure(checker, example.title, example.original, example.generated, example.image_type, FIGURES / name)
        results.append({"example": example.slug, **found, "figure": f"docs/images/{name}"})
    (ROOT / "examples" / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
