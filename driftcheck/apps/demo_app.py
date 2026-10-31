"""Gradio demo: upload an original and a generated image, run DriftCheck, see why.

It calls the same Checker the evaluation uses, with the garment cut out, routing by
image type and the frozen thresholds. This file only turns the report into words.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    title: str
    original: Path
    generated: Path
    image_type: str  # flat or scene, as in the evaluation


# Picked from the test split with the saved evaluation: a clean pass, and two deliberately
# changed copies that DriftCheck catches. The real Gemini outputs have no labelled changes.
EXAMPLE_KEYS = [
    ("Clean pass: white top, Gemini studio shot", "products/p031/original.jpg", "products/p031/outputs/studio__gemini_app.jpg", "flat"),
    ("Colour changed: NEYMAR jersey, colour shifted on purpose", "products/p041/original.jpg", "synthetic/p041/colour_shift.jpg", "flat"),
    ("Pattern changed: striped t-shirt, pattern shuffled on purpose", "products/p022/original.jpg", "synthetic/p022/pattern_swap.jpg", "flat"),
]


def examples(data_dir: Path) -> list[Example]:
    """The example pairs whose files exist. data/ is not in git, so a fresh clone has none."""
    found = []
    for title, original, generated, kind in EXAMPLE_KEYS:
        if (data_dir / original).exists() and (data_dir / generated).exists():
            found.append(Example(title, data_dir / original, data_dir / generated, kind))
    return found


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


def build_demo(checker_factory: Callable, data_dir: Path, model_note: str):
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

        found = examples(data_dir)
        if found:
            gr.Examples(
                examples=[[str(e.original), str(e.generated), e.image_type] for e in found],
                example_labels=[e.title for e in found],
                inputs=[original, generated, kind],
                label="Try an example from the test set",
            )
    return demo
