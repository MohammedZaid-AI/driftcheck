from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftcheck.apps.label_app import LabelSession, build_app
from driftcheck.dataset import OutputMeta, write_json
from driftcheck.ingest import ingest_inbox
from driftcheck.labels import ATTRIBUTES, LabelStore
from tests.helpers import FakeVisionClient, save_image


def _dataset(tmp_path: Path, outputs_per_product: int = 2) -> Path:
    for i in range(2):
        save_image(tmp_path / "inbox" / f"{i}.jpg", (40 * i, 80, 120))
    ingest_inbox(tmp_path, FakeVisionClient([{"category": "clothing", "product_type": "top"}] * 2))
    for pid in ("p001", "p002"):
        for style in ("studio", "lifestyle")[:outputs_per_product]:
            oid = f"{style}__gemini_app"
            image = save_image(tmp_path / "products" / pid / "outputs" / f"{oid}.jpg")
            write_json(image.with_suffix(".json"), OutputMeta(
                product_id=pid, output_id=oid, style=style, prompt="p", generator="gemini_app", model="m", date="2026-10-24",
            ))
    return tmp_path


def test_store_saves_every_change_and_keeps_null(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.json")
    store.set_value("p001", "studio__x", "logo_changed", None)
    store.set_value("p001", "studio__x", "colour_changed", True)
    rows = json.loads((tmp_path / "labels.json").read_text())
    assert rows == [{"product_id": "p001", "output_id": "studio__x", "logo_changed": None, "colour_changed": True}]
    assert not store.is_complete("p001", "studio__x")


def test_complete_label_round_trip(tmp_path: Path) -> None:
    store = LabelStore(tmp_path / "labels.json")
    for name in ATTRIBUTES:
        store.set_value("p001", "o", name, name == "colour_changed")
    store.set_value("p001", "o", "notes", "red became orange")
    labels = LabelStore(tmp_path / "labels.json").complete_labels()
    assert len(labels) == 1
    assert labels[0].colour_changed is True and labels[0].any_drift
    assert labels[0].notes == "red became orange"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LabelStore(tmp_path / "labels.json").set_value("p001", "o", "size_changed", True)


def test_session_walks_items_and_records(tmp_path: Path) -> None:
    session = LabelSession(_dataset(tmp_path))
    assert len(session.items) == 4
    assert session.first_open() == 0
    view = session.view(0)
    assert view[1].endswith("original.jpg") and view[2].endswith(".jpg")
    assert view[3:9] == [None] * 6

    for name in ATTRIBUTES:
        session.record(0, name, "n/a" if name == "logo_changed" else "no")
    assert "1 of 4 labelled" in session.header(0)
    assert session.first_open() == 1
    assert session.view(0)[3:9] == ["no", "no", "no", "n/a", "no", "no"]
    assert session.store.complete_labels()[0].logo_changed is None

    assert session.move(0, -1) == 0
    assert session.move(3, 1) == 3
    assert session.move(1, 1) == 2


def test_session_with_no_outputs(tmp_path: Path) -> None:
    session = LabelSession(tmp_path)
    assert "No registered outputs" in session.view(0)[0]


def test_app_builds(tmp_path: Path) -> None:
    assert build_app(_dataset(tmp_path)) is not None
