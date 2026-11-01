"""Gradio demo: upload an original and a generated image, run DriftCheck, see why.

It calls the same Checker the evaluation uses, with the garment cut out, routing by
image type and the frozen thresholds. This file only turns the report into words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from driftcheck.config import PROJECT_ROOT
from driftcheck.dataset import parse_output_id
from driftcheck.evaluate import image_type as image_type_for_style
from driftcheck.pipeline import Report, load_image

IMAGE_TYPES = ["auto", "flat", "scene"]
VERDICT_WORDS = {"drifted": "Changed", "faithful": "Looks fine", "unknown": "Not sure"}
CHECK_NAMES = {
    "product_missing": "Product present",
    "colour": "Colour",
    "text": "Text",
    "pattern": "Pattern",
    "shape": "Shape",
    "logo": "Logo",
}


@dataclass
class Example:
    slug: str
    title: str
    original: Path
    generated: Path
    image_type: str  # flat or scene, as in the evaluation


EXAMPLES_DIR = PROJECT_ROOT / "examples"

# Made by scripts/make_examples.py from the user's own photo, with one change edited in on
# purpose. They ship with the repo, so the demo has examples on any machine after a clone.
# They are not real AI outputs and not Scalio outputs.
EXAMPLE_KEYS = [
    ("unchanged", "Unchanged copy of the shirt", "plain_shirt_unchanged.jpg"),
    ("colour", "Colour edited on purpose: black shirt made blue", "plain_shirt_colour.jpg"),
    ("text", "Text edited on purpose: the word SALE added", "plain_shirt_text.jpg"),
    ("stripes", "Pattern edited on purpose: stripes added", "plain_shirt_stripes.jpg"),
]
EXAMPLE_ORIGINAL = "plain_shirt_original.jpg"


def examples(folder: Path = EXAMPLES_DIR) -> list[Example]:
    """The example pairs whose files exist in the examples folder."""
    original = folder / EXAMPLE_ORIGINAL
    if not original.exists():
        return []
    return [Example(slug, title, original, folder / name, "flat")
            for slug, title, name in EXAMPLE_KEYS if (folder / name).exists()]


def guess_image_type(choice: str, generated_path: str | None) -> str:
    """auto reads the style from a file named like studio__gemini_app.jpg. An upload has no
    style name, so auto falls back to scene, the safer choice: it skips colour and pattern."""
    if choice in ("flat", "scene"):
        return choice
    parsed = parse_output_id(Path(generated_path).stem) if generated_path else None
    return image_type_for_style(parsed[0], None) if parsed else "scene"


def check_word(status: str, applicable: bool) -> str:
    if not applicable:
        return "not applicable"
    return {"pass": "fine", "fail": "changed", "unknown": "not sure"}[status]


def explain(report: Report) -> dict:
    """Plain words for every part of a report."""
    fired = [CHECK_NAMES[c.name] for c in report.checks if c.status == "fail"]
    rows = [[CHECK_NAMES[c.name], check_word(c.status, c.applicable), c.reason] for c in report.checks]
    text = report.check("text")
    items = text.details.get("items", [])
    read_original = [o.text for o in report.card.ocr] or ["no text found"]
    read_generated = text.details.get("output_text") or ["no text found"]
    matches = [[i["text"], i["best_match"] or "nothing close", f"{i['score']:.0f}",
                "found" if i["found_by"] else "missing or changed"] for i in items]
    pattern = report.check("pattern")
    verdict = VERDICT_WORDS[report.verdict]
    summary = f"## {verdict}\n\n"
    summary += f"Checks that fired: {', '.join(fired)}" if fired else "No check fired."
    summary += f"\n\nImage type: {report.cutout.get('image_type', '')}"
    return {
        "summary": summary,
        "rows": rows,
        "read_original": "\n".join(read_original),
        "read_generated": "\n".join(read_generated),
        "matches": matches,
        "pattern_ran": bool(pattern.details.get("similarity_grid")),
        "seconds": report.seconds,
    }


def run_check(checker, original_path: str | None, generated_path: str | None, choice: str):
    """Run DriftCheck and return everything the page shows."""
    from driftcheck.heatmap import draw_heatmap

    if not original_path or not generated_path:
        return "Upload both images first.", [], None, "", "", [], ""
    kind = guess_image_type(choice, generated_path)
    report = checker.check(original_path, generated_path, image_type=kind)
    report.cutout["image_type"] = kind
    words = explain(report)
    heatmap = (draw_heatmap(load_image(original_path), load_image(generated_path), report)
               if words["pattern_ran"] else None)
    timing = f"Took {words['seconds']:.1f} s. {report.vision_calls} vision calls, the rest came from the cache."
    if not words["pattern_ran"]:
        timing += " No heatmap: the pattern check did not run on this image."
    return (words["summary"], words["rows"], heatmap, words["read_original"], words["read_generated"],
            words["matches"], timing)


def build_demo(checker_factory: Callable, model_note: str, examples_dir: Path = EXAMPLES_DIR):
    import gradio as gr

    holder: dict = {}

    def checker():
        if "checker" not in holder:  # loading the models takes a while, do it on the first run only
            holder["checker"] = checker_factory()
        return holder["checker"]

    with gr.Blocks(title="DriftCheck demo") as demo:
        gr.Markdown(
            "# DriftCheck\nUpload an original product photo and an AI generated version. "
            "DriftCheck checks whether the product itself changed.\n\n"
            f"{model_note} Colour and pattern are skipped on scene photos, such as model shots, lifestyle "
            "scenes and posters, because the same product looks different on a body or in a scene."
        )
        with gr.Row():
            original = gr.Image(label="Original product photo", type="filepath", height=360)
            generated = gr.Image(label="Generated image", type="filepath", height=360)
        with gr.Row():
            kind = gr.Dropdown(IMAGE_TYPES, value="auto", label="Image type",
                               info="auto reads the style from a project file name; an upload counts as a scene")
            run = gr.Button("Run DriftCheck", variant="primary")
        summary = gr.Markdown()
        timing = gr.Markdown()
        checks = gr.Dataframe(headers=["Check", "Verdict", "Reason"], label="Each check", wrap=True)
        with gr.Row():
            read_original = gr.Textbox(label="Text read in the original", lines=6)
            read_generated = gr.Textbox(label="Text read in the generated image", lines=6)
        matches = gr.Dataframe(headers=["Original text", "Closest in generated", "Match score", "Result"],
                               label="Text matching", wrap=True)
        heatmap = gr.Image(label="Heatmap: pattern drift in red, missing text boxed in red, colour pairs below",
                           type="pil")

        outputs = [summary, checks, heatmap, read_original, read_generated, matches, timing]
        run.click(lambda o, g, k: run_check(checker(), o, g, k), [original, generated, kind], outputs)

        found = examples(examples_dir)
        if found:
            gr.Examples(
                examples=[[str(e.original), str(e.generated), e.image_type] for e in found],
                example_labels=[e.title for e in found],
                inputs=[original, generated, kind],
                label="Try an example. These were edited on purpose to show how DriftCheck responds. They are not real AI or Scalio outputs.",
            )
    return demo
