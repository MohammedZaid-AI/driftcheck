from __future__ import annotations

from pathlib import Path

from driftcheck.cache import DiskCache, make_key


def test_key_is_stable_and_depends_on_every_input() -> None:
    base = make_key("model-a", "prompt", b"\x00\x01")
    assert base == make_key("model-a", "prompt", b"\x00\x01")
    assert base != make_key("model-b", "prompt", b"\x00\x01")
    assert base != make_key("model-a", "other prompt", b"\x00\x01")
    assert base != make_key("model-a", "prompt", b"\x00\x02")


def test_key_parts_cannot_run_together() -> None:
    assert make_key("ab", "c") != make_key("a", "bc")


def test_json_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, "test")
    key = make_key("x")
    assert cache.get(key) is None
    cache.set(key, {"colour": "red", "items": [1, 2], "hindi": "नमस्ते"})
    assert cache.get(key) == {"colour": "red", "items": [1, 2], "hindi": "नमस्ते"}


def test_bytes_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, "images")
    key = make_key("y")
    assert cache.get_bytes(key) is None
    cache.set_bytes(key, b"\x89PNG data")
    assert cache.get_bytes(key) == b"\x89PNG data"


def test_survives_new_instance(tmp_path: Path) -> None:
    DiskCache(tmp_path, "ns").set("abc123", 42)
    assert DiskCache(tmp_path, "ns").get("abc123") == 42
