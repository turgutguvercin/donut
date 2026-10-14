#!/usr/bin/env python3
"""Package a verified dataset into standalone ZIP parts below a byte limit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from download_tr_dataset import checksum


def package(root, output, max_bytes=45_000_000, base_url=""):
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    paddle = json.loads((root / "paddle-preprocessing-audit.json").read_text(encoding="utf-8"))
    if not audit["passed"] or not paddle["passed"] or audit["images_decoded_and_hashed"] != paddle["samples"]:
        raise ValueError("Both dataset and Paddle preprocessing audits must pass")
    if output.exists():
        raise ValueError("Package destination already exists")
    output.mkdir(parents=True)
    metadata = sorted(p for p in root.rglob("*") if p.is_file() and "images" not in p.relative_to(root).parts)
    images = sorted((root / "images").glob("*/*.jpg"))
    # Metadata compression is efficient; JPEG payloads are already compressed.
    # Use metadata-only parts first, then pack image parts by exact source size
    # with a conservative allowance for ZIP headers and compression expansion.
    batches, batch, size = [], [], 0
    for group in (metadata, images):
        for path in group:
            estimated = path.stat().st_size + 1024
            if estimated > max_bytes:
                # A large JSONL stays one entry in a dedicated archive, provided
                # compression brings the archive below the byte limit.
                temporary = output / "compressed-large.zip"
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    archive.write(path, root.name + "/" + path.relative_to(root).as_posix())
                if temporary.stat().st_size > max_bytes:
                    raise ValueError(f"A single compressed file exceeds the ZIP limit: {path}")
                if batch:
                    batches.append(batch)
                    batch, size = [], 0
                batches.append([path])
                temporary.unlink()
                continue
            if batch and size + estimated > max_bytes:
                batches.append(batch)
                batch, size = [], 0
            batch.append(path)
            size += estimated
        if batch:
            batches.append(batch)
            batch, size = [], 0
    parts = []
    for index, files in enumerate(batches, 1):
        name = f"turkish-ocr-100k-part-{index:02d}.zip"
        target = output / name
        image_count = 0
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, root.name + "/" + path.relative_to(root).as_posix())
                image_count += path.suffix.lower() == ".jpg" and "images" in path.relative_to(root).parts
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
    manifest = {"schema_version": 1, "dataset_directory": root.name, "image_count": len(images),
                "parts": parts, "total_download_bytes": sum(p["bytes"] for p in parts),
                "instructions": "Download every part and extract all into the same directory. These are independent ZIP archives, not a split ZIP stream.",
                "license": "CC BY-SA 4.0; see attribution and font notices inside the dataset."}
    (output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(p["sha256"] + "  " + p["file"] + "\n" for p in parts), encoding="utf-8")
    shutil.copy2(Path(__file__).with_name("download_tr_dataset.py"), output / "download_dataset.py")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=45_000_000)
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()
    package(args.dataset, args.output, args.max_bytes, args.base_url)


if __name__ == "__main__":
    main()
