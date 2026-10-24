from __future__ import annotations

from pathlib import Path

from PIL import Image

from driftcheck.dataset import list_product_ids, read_meta
from driftcheck.ingest import ingest_inbox
from driftcheck.vision import VisionError
from tests.helpers import FakeVisionClient, save_image

KURTI = {"category": "clothing", "product_type": "Sleeveless Cotton Kurti", "worn": True}
SPICE = {"category": "packaged_food", "product_type": "masala spice pack", "worn": False}


def test_moves_photos_and_fills_category(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "a.jpg", (200, 0, 0))
    save_image(tmp_path / "inbox" / "b.png", (0, 0, 200))
    result = ingest_inbox(tmp_path, FakeVisionClient([KURTI, SPICE]))

    assert result.added == ["p001", "p002"]
    assert list(tmp_path.joinpath("inbox").iterdir()) == []
    meta = read_meta(tmp_path, "p001")
    assert meta.category == "clothing"
    assert meta.product_type == "sleeveless cotton kurti"
    assert meta.source_file == "a.jpg"
    assert meta.category_by == "fake/fake-vl"
    assert meta.worn is True
    assert read_meta(tmp_path, "p002").worn is False
    assert read_meta(tmp_path, "p002").category == "packaged_food"
    original = Image.open(tmp_path / "products" / "p002" / "original.jpg")
    assert original.format == "JPEG"
    assert (tmp_path / "products" / "p002" / "outputs").is_dir()


def test_ingests_real_sample_photo(tmp_path: Path, plain_shirt: Path) -> None:
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox" / "shirt.jpg").write_bytes(plain_shirt.read_bytes())
    shirt = {"category": "clothing", "product_type": "black long sleeve shirt", "worn": False}
    result = ingest_inbox(tmp_path, FakeVisionClient([shirt]))
    assert result.added == ["p001"]
    assert Image.open(tmp_path / "products" / "p001" / "original.jpg").size == (333, 500)
    assert plain_shirt.exists()  # the sample itself is never moved


def test_avif_is_converted_to_jpg(tmp_path: Path) -> None:
    (tmp_path / "inbox").mkdir()
    Image.new("RGB", (80, 60), (10, 120, 200)).save(tmp_path / "inbox" / "photo.avif", format="AVIF")
    result = ingest_inbox(tmp_path, None)
    assert result.added == ["p001"] and result.skipped == []
    original = Image.open(tmp_path / "products" / "p001" / "original.jpg")
    assert original.format == "JPEG" and original.size == (80, 60)
    assert read_meta(tmp_path, "p001").worn is None  # unknown until the vision step runs


def test_large_photo_is_capped(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "big.jpg", size=(4000, 3000))
    ingest_inbox(tmp_path, None)
    assert max(Image.open(tmp_path / "products" / "p001" / "original.jpg").size) == 2048


def test_duplicate_and_unsupported_files_are_skipped(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "a.jpg")
    ingest_inbox(tmp_path, None)
    save_image(tmp_path / "inbox" / "again.jpg")  # same pixels, same bytes
    (tmp_path / "inbox" / "notes.txt").write_text("hi")
    result = ingest_inbox(tmp_path, None)
    assert result.added == []
    assert result.duplicates == ["again.jpg"]
    assert result.skipped == ["notes.txt"]
    assert list_product_ids(tmp_path) == ["p001"]


def test_ids_continue_after_existing_products(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "a.jpg", (1, 2, 3))
    ingest_inbox(tmp_path, None)
    save_image(tmp_path / "inbox" / "b.jpg", (4, 5, 6))
    assert ingest_inbox(tmp_path, None).added == ["p002"]


def test_failed_category_is_retried_on_next_run(tmp_path: Path) -> None:
    save_image(tmp_path / "inbox" / "a.jpg")
    first = ingest_inbox(tmp_path, FakeVisionClient([VisionError("Ollama is down")]))
    assert first.category_failed == ["p001"]
    assert read_meta(tmp_path, "p001").category is None

    client = FakeVisionClient([KURTI])
    second = ingest_inbox(tmp_path, client)
    assert second.categorised == ["p001"]
    assert read_meta(tmp_path, "p001").category == "clothing"
    # A third run asks nothing because every product has a category.
    third_client = FakeVisionClient([])
    ingest_inbox(tmp_path, third_client)
    assert third_client.asked == []
