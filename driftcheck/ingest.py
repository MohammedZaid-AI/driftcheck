"""driftcheck ingest: move photos from data/inbox into product folders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from driftcheck.dataset import (
    IMAGE_SUFFIXES,
    Category,
    ProductMeta,
    list_product_ids,
    next_product_id,
    product_dir,
    read_meta,
    write_json,
)
from driftcheck.log import get_logger
from driftcheck.vision import VisionClient, VisionError

log = get_logger(__name__)

MAX_ORIGINAL_SIDE = 2048

CATEGORY_PROMPT = (
    "This is a product photo from an Indian online seller. "
    "Pick the category of the main product and give a short product type of two to four words, "
    "for example sleeveless cotton kurti, steel water bottle or masala spice pack."
)


class CategoryGuess(BaseModel):
    category: Category
    product_type: str = Field(description="two to four words, for example sleeveless cotton kurti")
    worn: bool = Field(description="true if a person or model is wearing the product in this photo, false if it is laid flat, on a hanger, on a mannequin or on its own")


@dataclass
class IngestResult:
    added: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    categorised: list[str] = field(default_factory=list)
    category_failed: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_original(source: Path, target: Path) -> None:
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((MAX_ORIGINAL_SIDE, MAX_ORIGINAL_SIDE), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        img.save(target, format="JPEG", quality=95)


def _known_hashes(data_dir: Path) -> set[str]:
    return {read_meta(data_dir, pid).sha256 for pid in list_product_ids(data_dir)}


def ingest_inbox(data_dir: Path, client: VisionClient | None) -> IngestResult:
    """Move every photo in data/inbox into its own product folder, then fill
    missing categories. Pass client=None to skip the vision step."""
    result = IngestResult()
    inbox = data_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    known = _known_hashes(data_dir)

    for source in sorted(p for p in inbox.iterdir() if p.is_file()):
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            log.warning("Skipping %s. Use jpg, png, webp or avif.", source.name)
            result.skipped.append(source.name)
            continue
        digest = _sha256(source)
        if digest in known:
            log.warning("Skipping %s. The same photo is already ingested.", source.name)
            result.duplicates.append(source.name)
            continue

        product_id = next_product_id(data_dir)
        folder = product_dir(data_dir, product_id)
        _save_original(source, folder / "original.jpg")
        (folder / "outputs").mkdir(exist_ok=True)
        meta = ProductMeta(
            product_id=product_id,
            source_file=source.name,
            sha256=digest,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        write_json(folder / "meta.json", meta)
        source.unlink()
        known.add(digest)
        result.added.append(product_id)
        log.info("%s -> %s", source.name, product_id)

    if client is not None:
        fill_categories(data_dir, client, result)
    return result


def fill_categories(data_dir: Path, client: VisionClient, result: IngestResult) -> None:
    """Ask the vision model for the category of every product that has none yet."""
    for product_id in list_product_ids(data_dir):
        meta = read_meta(data_dir, product_id)
        if meta.category is not None:
            continue
        folder = product_dir(data_dir, product_id)
        try:
            guess = client.ask(CATEGORY_PROMPT, [folder / "original.jpg"], CategoryGuess)
        except VisionError as error:
            log.warning("No category for %s: %s", product_id, error)
            result.category_failed.append(product_id)
            continue
        meta.category = guess.category
        meta.product_type = guess.product_type.strip().lower()
        meta.worn = guess.worn
        meta.category_by = f"{client.provider.name}/{client.model}"
        write_json(folder / "meta.json", meta)
        result.categorised.append(product_id)
        log.info("%s is %s: %s", product_id, meta.category, meta.product_type)
