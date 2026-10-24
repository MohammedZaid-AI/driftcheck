"""Human labels in data/labels.json.

Each attribute is true, false or null. null means not applicable, for example
no logo on the product. A missing key means not labelled yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ATTRIBUTES = (
    "shape_changed",
    "colour_changed",
    "pattern_changed",
    "logo_changed",
    "text_changed",
    "product_missing",
)


class Label(BaseModel):
    product_id: str
    output_id: str
    shape_changed: bool | None
    colour_changed: bool | None
    pattern_changed: bool | None
    logo_changed: bool | None
    text_changed: bool | None
    product_missing: bool | None
    notes: str = ""

    @property
    def any_drift(self) -> bool:
        return any(getattr(self, name) is True for name in ATTRIBUTES)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class LabelStore:
    """Reads and writes labels.json. Every change is saved straight away."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        if path.exists():
            for row in json.loads(path.read_text(encoding="utf-8")):
                self._rows[(row["product_id"], row["output_id"])] = row

    def get(self, product_id: str, output_id: str) -> dict[str, Any]:
        return dict(self._rows.get((product_id, output_id), {}))

    def set_value(self, product_id: str, output_id: str, field: str, value: bool | None | str) -> None:
        if field not in ATTRIBUTES and field != "notes":
            raise ValueError(f"Unknown label field {field}")
        row = self._rows.setdefault((product_id, output_id), {"product_id": product_id, "output_id": output_id})
        row[field] = value
        self.save()

    def is_complete(self, product_id: str, output_id: str) -> bool:
        row = self._rows.get((product_id, output_id), {})
        return all(name in row for name in ATTRIBUTES)

    def complete_labels(self) -> list[Label]:
        return [Label.model_validate(row) for key, row in sorted(self._rows.items()) if self.is_complete(*key)]

    def save(self) -> None:
        write_json_atomic(self.path, [row for _, row in sorted(self._rows.items())])
