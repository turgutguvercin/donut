#!/usr/bin/env python3
"""Package a verified dataset into standalone ZIP parts below a byte limit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from download_tr_dataset import checksum


def package(root, output, max_bytes=90_000_000, base_url="", prefix="turkish-ocr-100k"):
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    paddle = json.loads((root / "paddle-preprocessing-audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or not paddle["passed"] or audit["images_decoded_and_hashed"] != paddle["samples"]:
        raise ValueError("Both dataset and Paddle preprocessing audits must pass")
    if output.exists():
        raise ValueError("Package destination already exists")
    output.mkdir(parents=True)
    metadata = sorted(p for p in root.rglob("*") if p.is_file() and "images" not in p.relative_to(root).parts)
    images = sorted((root / "images").glob("*/*.jpg"))
    if not prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in prefix):
        raise ValueError("Use a simple archive filename prefix")
    # A 200k JSONL exceeds the 100MB GitHub single-file limit even compressed.
    # Store bounded byte ranges as independently checksummed metadata shards.
    # The downloader reassembles them atomically, checking the original hash.
    assembled_files = []
    def entries(paths):
        result = []
        for path in paths:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            if size <= max_bytes - 2048:
                result.append((path, relative, 0, size))
                continue
            shards = []
            chunk_size = max_bytes // 2
            with path.open("rb") as source:
                for i, offset in enumerate(range(0, size, chunk_size)):
                    length = min(chunk_size, size - offset)
                    name = "_metadata_parts/" + relative.replace("/", "__") + f".{i:05d}.part"
                    block = source.read(length)
                    shards.append({"file": name, "bytes": length, "sha256": hashlib.sha256(block).hexdigest()})
                    result.append((path, name, offset, length))
            assembled_files.append({"file": relative, "bytes": size, "sha256": checksum(path), "parts": shards})
        return result

    batches, batch, size = [], [], 0
    for group in (entries(metadata), entries(images)):
        for entry in group:
            estimated = entry[3] + 2048
            if batch and size + estimated > max_bytes:
                batches.append(batch)
                batch, size = [], 0
            batch.append(entry)
            size += estimated
        if batch:
            batches.append(batch)
            batch, size = [], 0
    parts = []
    for index, files in enumerate(batches, 1):
        name = f"{prefix}-part-{index:02d}.zip"
        target = output / name
        image_count = 0
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path, relative, offset, length in files:
                name_in_archive = root.name + "/" + relative
                if offset == 0 and length == path.stat().st_size and not relative.startswith("_metadata_parts/"):
                    archive.write(path, name_in_archive)
                else:
                    with archive.open(name_in_archive, "w", force_zip64=True) as destination, path.open("rb") as source:
                        source.seek(offset)
                        remaining = length
                        while remaining:
                            block = source.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Metadata changed during packaging")
                            destination.write(block)
                            remaining -= len(block)
                image_count += relative.startswith("images/") and path.suffix.lower() == ".jpg"
        if target.stat().st_size > max_bytes:
            raise ValueError("ZIP exceeds requested size limit")
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity error")
        part = {"file": name, "bytes": target.stat().st_size, "sha256": checksum(target),
                "image_count": image_count, "entry_count": len(files)}
        if base_url:
            part["url"] = base_url.rstrip("/") + "/" + name
        parts.append(part)
        print(f"Verified {name}: {part['bytes']:,} bytes, {image_count:,} images", flush=True)
    manifest = {"schema_version": 2 if assembled_files else 1, "dataset_directory": root.name, "image_count": len(images),
                "parts": parts, "total_download_bytes": sum(p["bytes"] for p in parts),
                "instructions": "Use download_dataset.py to download/extract every independent ZIP and reassemble any sharded metadata with checksum verification.",
                "license": "CC BY-SA 4.0; see attribution and font notices inside the dataset."}
    if assembled_files:
        manifest["assembled_files"] = assembled_files
    (output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(p["sha256"] + "  " + p["file"] + "\n" for p in parts), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("download_tr_dataset.py"), output / "download_dataset.py")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=90_000_000)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--prefix", default="turkish-ocr-100k")
    args = parser.parse_args()
    package(args.dataset, args.output, args.max_bytes, args.base_url, args.prefix)


if __name__ == "__main__":
    main()
