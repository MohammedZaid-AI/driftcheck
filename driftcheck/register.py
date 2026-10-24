"""Make generation jobs, and record outputs made in Kaggle or the Gemini app.

Naming rule: data/products/<product_id>/outputs/<style>__<generator>.jpg
Styles are model_shot, studio and lifestyle. The generator is a lowercase
name such as qwen_edit or gemini_app.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from driftcheck.dataset import (
    IMAGE_SUFFIXES,
    STYLES,
    OutputMeta,
    list_product_ids,
    output_id,
    parse_output_id,
    product_dir,
    read_meta,
    write_json,
)
from driftcheck.log import get_logger
from driftcheck.prompts import build_prompt

log = get_logger(__name__)

# Model names for generators that do not write their own json.
KNOWN_GENERATOR_MODELS = {"gemini_app": "Gemini app, free tier"}


class Job(BaseModel):
    product_id: str
    output_id: str
    style: str
    prompt: str


def make_jobs(data_dir: Path, generator: str) -> list[Job]:
    """One job per product and style that has no output from this generator yet."""
    jobs = []
    for product_id in list_product_ids(data_dir):
        meta = read_meta(data_dir, product_id)
        outputs = product_dir(data_dir, product_id) / "outputs"
        for style in STYLES:
            oid = output_id(style, generator)
            if (outputs / f"{oid}.jpg").exists():
                continue
            prompt = build_prompt(meta.category, meta.product_type, style)
            jobs.append(Job(product_id=product_id, output_id=oid, style=style, prompt=prompt))
    return jobs


def export_jobs(data_dir: Path, generator: str) -> Path:
    """Write data/jobs/<generator>/ with jobs.json, a zip for Kaggle and a checklist for the manual route."""
    jobs = make_jobs(data_dir, generator)
    out = data_dir / "jobs" / generator
    out.mkdir(parents=True, exist_ok=True)
    jobs_json = json.dumps([job.model_dump() for job in jobs], indent=2, ensure_ascii=False)
    (out / "jobs.json").write_text(jobs_json, encoding="utf-8")

    with zipfile.ZipFile(out / "kaggle_input.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("jobs.json", jobs_json)
        for product_id in sorted({job.product_id for job in jobs}):
            original = product_dir(data_dir, product_id) / "original.jpg"
            archive.write(original, f"products/{product_id}/original.jpg")

    lines = [f"# Manual route checklist for {generator}", ""]
    lines.append("For each item: open the original in the Gemini app, paste the prompt, download the result and save it with the exact name shown.")
    lines.append("")
    for job in jobs:
        folder = product_dir(data_dir, job.product_id)
        lines += [
            f"## {job.product_id} {job.style}",
            f"Original: {folder / 'original.jpg'}",
            f"Save as: {folder / 'outputs' / (job.output_id + '.jpg')}",
            "",
            job.prompt,
            "",
        ]
    (out / "manual_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %d jobs to %s", len(jobs), out)
    return out


@dataclass
class RegisterResult:
    imported: list[str] = field(default_factory=list)
    registered: list[str] = field(default_factory=list)
    already: list[str] = field(default_factory=list)
    bad_names: list[str] = field(default_factory=list)


def _to_jpg(path: Path) -> Path:
    """Convert png or webp to jpg next to it. Returns the jpg path."""
    if path.suffix.lower() == ".jpg":
        return path
    target = path.with_suffix(".jpg")
    with Image.open(path) as img:
        ImageOps.exif_transpose(img).convert("RGB").save(target, format="JPEG", quality=95)
    path.unlink()
    return target


def import_folder(source: Path, data_dir: Path, result: RegisterResult) -> None:
    """Copy outputs from an unzipped Kaggle result. Expects products/<pid>/outputs/ inside source."""
    known = set(list_product_ids(data_dir))
    for image in sorted(source.glob("**/products/*/outputs/*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        product_id = image.parent.parent.name
        if product_id not in known:
            log.warning("Skipping %s. Product %s is not in data/products.", image, product_id)
            continue
        target_dir = product_dir(data_dir, product_id) / "outputs"
        shutil.copy2(image, target_dir / image.name)
        sidecar = image.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, target_dir / sidecar.name)
        result.imported.append(f"{product_id}/{image.stem}")


def register_outputs(data_dir: Path, model: str | None = None, source: Path | None = None) -> RegisterResult:
    """Record every output image with a json next to it. Images that break the naming rule are reported."""
    result = RegisterResult()
    if source is not None:
        import_folder(source, data_dir, result)

    for product_id in list_product_ids(data_dir):
        meta = read_meta(data_dir, product_id)
        outputs = product_dir(data_dir, product_id) / "outputs"
        if not outputs.exists():
            continue
        for image in sorted(p for p in outputs.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            parsed = parse_output_id(image.stem)
            if parsed is None:
                result.bad_names.append(f"{product_id}/{image.name}")
                continue
            image = _to_jpg(image)
            style, generator = parsed
            sidecar = image.with_suffix(".json")
            if sidecar.exists():
                try:
                    OutputMeta.model_validate_json(sidecar.read_text(encoding="utf-8"))
                    result.already.append(f"{product_id}/{image.stem}")
                    continue
                except ValidationError as error:
                    log.warning("Rewriting %s, it was not valid: %s", sidecar, error)
            record = OutputMeta(
                product_id=product_id,
                output_id=image.stem,
                style=style,
                prompt=build_prompt(meta.category, meta.product_type, style),
                generator=generator,
                model=model or KNOWN_GENERATOR_MODELS.get(generator, "unknown"),
                date=datetime.fromtimestamp(image.stat().st_mtime).date().isoformat(),
            )
            write_json(sidecar, record)
            result.registered.append(f"{product_id}/{image.stem}")

    for name in result.bad_names:
        log.warning("Not registered: %s. Name it <style>__<generator>.jpg, for example studio__gemini_app.jpg", name)
    return result
