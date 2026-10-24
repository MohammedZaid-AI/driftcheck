"""driftcheck ingest: move photos from data/inbox into product folders."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
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
from driftcheck.prompts import WEARABLE
from driftcheck.vision import VisionClient, VisionError

log = get_logger(__name__)

MAX_ORIGINAL_SIDE = 2048

CATEGORY_PROMPT = (
    "This is a product photo from an Indian online seller. "
    "Pick the category of the main product and give a short product type of two to four words, "
    "for example sleeveless cotton kurti, steel water bottle or masala spice pack. "
    "Use storefront for a photo of a shop front or shop sign board. "
    "Say if the product is worn: only when a person's body is visible wearing it. "
    "A hand holding a bottle or pack is not worn. A mannequin or a close up with no body is not worn."
)


class CategoryGuess(BaseModel):
    category: Category
    product_type: str = Field(description="two to four words, for example sleeveless cotton kurti")
    worn: bool = Field(description=(
        "true only if a person's body is visible wearing the product. "
        "false if a hand is only holding it, for a close up with no body visible, "
        "and for a mannequin, hanger, flat lay or the product alone"
    ))


@dataclass
class IngestResult:
    added: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    blank: list[str] = field(default_factory=list)
    categorised: list[str] = field(default_factory=list)
    category_failed: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MIN_PIXEL_STD = 2.0  # below this the photo is one flat colour, for example all black


def load_photo(source: Path) -> Image.Image:
    """Open any supported photo as RGB. Transparent areas become white, not black."""
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            flat = Image.new("RGB", rgba.size, "white")
            flat.paste(rgba, mask=rgba.getchannel("A"))
            img = flat
        img = img.convert("RGB")
        img.thumbnail((MAX_ORIGINAL_SIDE, MAX_ORIGINAL_SIDE), Image.Resampling.LANCZOS)
        return img


def is_blank(img: Image.Image) -> bool:
    return float(np.asarray(img).std()) < MIN_PIXEL_STD


def _known_hashes(data_dir: Path) -> set[str]:
    return {read_meta(data_dir, pid).sha256 for pid in list_product_ids(data_dir)}


def ingest_inbox(data_dir: Path, client: VisionClient | None, refresh: bool = False) -> IngestResult:
    """Copy every photo in data/inbox into its own product folder as original.jpg,
    move the source to data/inbox/done, then fill missing categories.
    Pass client=None to skip the vision step, refresh=True to redo every category."""
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

        photo = load_photo(source)
        if is_blank(photo):
            log.warning("Skipping %s. The image is one flat colour, so there is no product to see.", source.name)
            result.blank.append(source.name)
            continue

        product_id = next_product_id(data_dir)
        folder = product_dir(data_dir, product_id)
        (folder / "outputs").mkdir(parents=True, exist_ok=True)
        photo.save(folder / "original.jpg", format="JPEG", quality=95)
        meta = ProductMeta(
            product_id=product_id,
            source_file=source.name,
            sha256=digest,
            ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        write_json(folder / "meta.json", meta)
        # Keep the untouched source, so nothing is lost if a conversion goes wrong.
        done = inbox / "done"
        done.mkdir(exist_ok=True)
        source.replace(done / f"{product_id}_{source.name}")
        known.add(digest)
        result.added.append(product_id)
        log.info("%s -> %s", source.name, product_id)

    if client is not None:
        fill_categories(data_dir, client, result, refresh)
    return result


def fill_categories(data_dir: Path, client: VisionClient, result: IngestResult, refresh: bool = False) -> None:
    """Ask the vision model for the category of every product that has none yet, or of all with refresh."""
    for product_id in list_product_ids(data_dir):
        meta = read_meta(data_dir, product_id)
        if meta.category is not None and not refresh:
            continue
        folder = product_dir(data_dir, product_id)
        try:
            guess = client.ask(CATEGORY_PROMPT, [folder / "original.jpg"], CategoryGuess)
        except VisionError as error:
            log.warning("No category for %s: %s", product_id, error)
            result.category_failed.append(product_id)
            continue
        # Fields fixed by hand in config/selection.json are never overwritten.
        if "category" not in meta.manual_fields:
            meta.category = guess.category
        if "product_type" not in meta.manual_fields:
            meta.product_type = guess.product_type.strip().lower()
        if "worn" not in meta.manual_fields:
            # A spice pack in someone's hand is held, not worn.
            meta.worn = guess.worn and meta.category in WEARABLE
        meta.category_by = f"{client.provider.name}/{client.model}"
        write_json(folder / "meta.json", meta)
        result.categorised.append(product_id)
        log.info("%s is %s: %s", product_id, meta.category, meta.product_type)
