#!/usr/bin/env python3
"""Build a compact EPUB (under 3 MB) from bookepub.adoc and the source chapters.

Run from the project root:

    python scripts/build_compact_epub.py

The EPUB source uses images/ at full resolution, which produces a ~6 MB file.
This script makes temporary, screen-sized copies of every figure, points a
temporary copy of bookepub.adoc at them, reduces the base font size in the
generated CSS, and writes book.epub.  The original images and chapters are
never modified, and all temporary files are deleted afterwards.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ADOC = ROOT / "bookepub.adoc"
IMAGES = ROOT / "images"
BUILD = ROOT / "_epub_build"
BUILD_ADOC = ROOT / "_bookepub_build.adoc"
RAW_EPUB = BUILD / "book.epub"
FINAL_EPUB = ROOT / "book.epub"
SIZE_LIMIT = 3_000_000

MAX_W = 640
COLOURS = 48
COVER_W = 432
COVER_COLOURS = 32
CSS_OVERRIDE = "\nbody{font-size:0.9em}\npre,code,kbd{font-size:0.85em}\n"


def optimise_images() -> int:
    img_dir = BUILD / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for src in sorted(IMAGES.glob("ch*.png")) + [IMAGES / "cover.png"]:
        image = Image.open(src).convert("RGB")
        width, height = image.size
        is_cover = src.name == "cover.png"
        limit = COVER_W if is_cover else MAX_W
        if width > limit:
            image = image.resize(
                (limit, round(height * limit / width)), Image.LANCZOS)
        image = image.quantize(
            colors=COVER_COLOURS if is_cover else COLOURS,
            method=Image.Quantize.MEDIANCUT)
        image.save(img_dir / src.name, optimize=True)
        total += (img_dir / src.name).stat().st_size
    return total


def write_build_adoc() -> None:
    text = SOURCE_ADOC.read_text(encoding="utf-8")
    text = text.replace(":imagesdir: images", ":imagesdir: _epub_build/images")
    text = text.replace(":icons: font\n", "")
    BUILD_ADOC.write_text(text, encoding="utf-8")


def build() -> None:
    subprocess.run(
        f"asciidoctor-epub3 -o {RAW_EPUB.as_posix()} {BUILD_ADOC.name}",
        cwd=ROOT, check=True, shell=True)


def repack() -> None:
    with zipfile.ZipFile(RAW_EPUB) as source:
        entries = [(info, source.read(info.filename))
                   for info in source.infolist()]
    with zipfile.ZipFile(FINAL_EPUB, "w") as target:
        for info, data in entries:
            if info.filename == "EPUB/styles/epub3.css":
                data = data.decode("utf-8").rstrip() + CSS_OVERRIDE
                data = data.encode("utf-8")
            entry = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            entry.compress_type = (zipfile.ZIP_STORED
                                   if info.filename == "mimetype"
                                   else zipfile.ZIP_DEFLATED)
            entry.external_attr = info.external_attr
            target.writestr(entry, data)


def verify() -> None:
    size = FINAL_EPUB.stat().st_size
    with zipfile.ZipFile(FINAL_EPUB) as archive:
        assert archive.testzip() is None, "EPUB ZIP integrity failed"
        opf = archive.read("EPUB/package.opf").decode("utf-8")
        title = re.search(r"<dc:title[^>]*>(.*?)</dc:title>", opf)
        assert title and "Master Data Analysis with Python" in title.group(1)
        css = archive.read("EPUB/styles/epub3.css").decode("utf-8")
        assert "body{font-size:0.9em}" in css
    if size >= SIZE_LIMIT:
        raise SystemExit(
            f"EPUB is {size:,} bytes, above the {SIZE_LIMIT:,}-byte limit")
    print(f"book.epub: {size:,} bytes ({size / 1e6:.2f} MB) - OK")


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir()
    try:
        total = optimise_images()
        print(f"optimised figures: {total:,} bytes")
        write_build_adoc()
        build()
        repack()
        verify()
    finally:
        BUILD_ADOC.unlink(missing_ok=True)
        shutil.rmtree(BUILD, ignore_errors=True)


if __name__ == "__main__":
    main()
