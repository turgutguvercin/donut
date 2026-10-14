#!/usr/bin/env python3
"""Download all Turkish OCR dataset parts, verify SHA-256, and extract together.

Only Python's standard library is required. Existing verified parts are reused.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

DEFAULT_MANIFEST = "https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/download-manifest.json"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_part(part, directory, base_url):
    name = part["file"]
    if PurePosixPath(name).name != name or "\\" in name:
        raise ValueError("Manifest contains an unsafe part filename")
    target = directory / name
    if target.exists() and target.stat().st_size == part["bytes"] and checksum(target) == part["sha256"]:
        print("Verified existing " + name, flush=True)
        return target
    temporary = target.with_suffix(target.suffix + ".partial")
    url = part.get("url", urllib.parse.urljoin(base_url, name))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            if temporary.stat().st_size != part["bytes"] or checksum(temporary) != part["sha256"]:
                raise ValueError("Downloaded part failed size/SHA-256 verification: " + name)
            temporary.replace(target)
            print("Downloaded and verified " + name, flush=True)
            return target
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def extract_part(archive, destination):
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                raise ValueError("Archive contains an unsafe path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not supported")
            target = (root / info.filename).resolve()
            if root not in target.parents:
                raise ValueError("Archive entry escapes the destination")
        # ZipFile verifies each entry's CRC while reading it during extraction.
        bundle.extractall(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path("turkish-ocr-download"))
    parser.add_argument("--parts-dir", type=Path, help="Reuse ZIPs from this directory, e.g. an extracted GitHub tag archive")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    local_manifest = Path(args.manifest)
    if not urllib.parse.urlparse(args.manifest).scheme and local_manifest.is_file():
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        base_url = local_manifest.resolve().as_uri().rsplit("/", 1)[0] + "/"
    else:
        with urllib.request.urlopen(args.manifest, timeout=60) as source:
            manifest = json.load(source)
        base_url = args.manifest.rsplit("/", 1)[0] + "/"
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    parts_dir = args.parts_dir or args.output / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(lambda part: download_part(part, parts_dir, base_url), manifest["parts"]))
    if not args.download_only:
        for part in parts:
            extract_part(part, args.output)
        dataset = args.output / manifest["dataset_directory"]
        image_count = sum(1 for _ in (dataset / "images").glob("*/*.jpg"))
        with (dataset / "labels.txt").open(encoding="utf-8") as handle:
            label_count = sum(1 for _ in handle)
        if image_count != manifest["image_count"] or label_count != image_count:
            raise ValueError("Extracted image/label count mismatch")
        print(f"Ready: {dataset.resolve()} ({image_count:,} images and UTF-8 labels)")


if __name__ == "__main__":
    main()
