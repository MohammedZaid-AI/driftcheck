"""Gradio labelling app. Original and output side by side, one answer per attribute."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from driftcheck.dataset import list_outputs, output_image, product_dir
from driftcheck.labels import ATTRIBUTES, LabelStore

CHOICES = ["yes", "no", "n/a"]
_TO_VALUE: dict[str, bool | None] = {"yes": True, "no": False, "n/a": None}
_TO_CHOICE = {True: "yes", False: "no", None: "n/a"}

HELP = {
    "shape_changed": "Sleeves, neckline, length or silhouette changed",
    "colour_changed": "A product colour changed. Ignore lighting",
    "pattern_changed": "Print, embroidery or texture changed",
    "logo_changed": "Logo changed or missing. n/a if the product has no logo",
    "text_changed": "Any text changed or missing. n/a if the product has no text",
    "product_missing": "The product is not visible in the output",
}


class LabelSession:
    """The app logic without the UI, so it can be tested."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.items = list_outputs(data_dir)
        self.store = LabelStore(data_dir / "labels.json")

    def first_open(self) -> int:
        for i, (pid, oid) in enumerate(self.items):
            if not self.store.is_complete(pid, oid):
                return i
        return 0

    def done_count(self) -> int:
        return sum(self.store.is_complete(pid, oid) for pid, oid in self.items)

    def header(self, index: int) -> str:
        if not self.items:
            return "No registered outputs yet. Run driftcheck register first."
        pid, oid = self.items[index]
        state = "done" if self.store.is_complete(pid, oid) else "open"
        return f"**{index + 1} of {len(self.items)}** · {pid} / {oid} · {state} · {self.done_count()} of {len(self.items)} labelled"

    def view(self, index: int) -> list[Any]:
        """Header, original, output, one choice per attribute, notes."""
        if not self.items:
            return [self.header(0), None, None, *[None] * len(ATTRIBUTES), ""]
        pid, oid = self.items[index]
        row = self.store.get(pid, oid)
        choices = [_TO_CHOICE[row[name]] if name in row else None for name in ATTRIBUTES]
        original = str(product_dir(self.data_dir, pid) / "original.jpg")
        output = str(output_image(self.data_dir, pid, oid))
        return [self.header(index), original, output, *choices, row.get("notes", "")]

    def record(self, index: int, field: str, value: str | None) -> str:
        if self.items and value is not None:
            pid, oid = self.items[index]
            stored = value if field == "notes" else _TO_VALUE[value]
            self.store.set_value(pid, oid, field, stored)
        return self.header(index)

    def move(self, index: int, step: int) -> int:
        if not self.items:
            return 0
        return max(0, min(len(self.items) - 1, index + step))


def build_app(data_dir: Path):
    import gradio as gr

    session = LabelSession(data_dir)
    with gr.Blocks(title="DriftCheck labelling") as demo:
        index = gr.State(session.first_open())
        header = gr.Markdown()
        with gr.Row():
            original = gr.Image(label="Original", type="filepath", height=520, interactive=False)
            output = gr.Image(label="AI output", type="filepath", height=520, interactive=False)
        radios = [gr.Radio(CHOICES, label=name.replace("_", " "), info=HELP[name]) for name in ATTRIBUTES]
        notes = gr.Textbox(label="Notes", lines=2)
        with gr.Row():
            back = gr.Button("Back")
            forward = gr.Button("Next", variant="primary")

        view_outputs = [header, original, output, *radios, notes]

        # .input fires only when the user changes a value, not when the view is refreshed.
        for name, radio in zip(ATTRIBUTES, radios):
            radio.input(lambda i, v, name=name: session.record(i, name, v), [index, radio], header)
        notes.input(lambda i, v: session.record(i, "notes", v), [index, notes], header)

        back.click(lambda i: session.move(i, -1), index, index).then(session.view, index, view_outputs)
        forward.click(lambda i: session.move(i, 1), index, index).then(session.view, index, view_outputs)
        demo.load(session.view, index, view_outputs)
    return demo
