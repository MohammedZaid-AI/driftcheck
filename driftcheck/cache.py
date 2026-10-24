"""Disk cache for API responses, so a rerun costs nothing.

Keys are a SHA256 hash of every input: model name, prompt and image bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def make_key(*parts: str | bytes) -> str:
    """Hash the inputs into a stable key. Parts are length prefixed so
    ("ab", "c") and ("a", "bc") give different keys."""
    digest = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class DiskCache:
    """Stores JSON values and raw bytes under cache_dir/namespace/."""

    def __init__(self, cache_dir: Path, namespace: str) -> None:
        self.dir = Path(cache_dir) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str, suffix: str) -> Path:
        # Two level fan out keeps folders small.
        return self.dir / key[:2] / f"{key}{suffix}"

    def get(self, key: str) -> Any | None:
        path = self._path(key, ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        self._write(self._path(key, ".json"), json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key, ".bin")
        return path.read_bytes() if path.exists() else None

    def set_bytes(self, key: str, value: bytes) -> None:
        self._write(self._path(key, ".bin"), value)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        # Write to a temp file first so a crash never leaves half a file.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
