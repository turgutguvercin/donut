import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_tr_dataset import checksum, download_part, extract_part, assemble_metadata
from package_tr_dataset import package
from prepare_paddle_tr import training_config


def test_verified_archive_reuse_and_utf8_extraction(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "part.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset/labels.txt", "images/line.jpg\tÇiğdem ışığı gördü.\n".encode("utf-8"))
    part = {"file": "part.zip", "bytes": archive.stat().st_size, "sha256": checksum(archive)}
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    copied = download_part(part, downloads, source.as_uri() + "/")
    # A verified cached part must be usable even without access to its source.
    archive.unlink()
    assert download_part(part, downloads, source.as_uri() + "/") == copied
    extract_part(copied, tmp_path / "extracted")
    assert (tmp_path / "extracted/dataset/labels.txt").read_text(encoding="utf-8").endswith("Çiğdem ışığı gördü.\n")


def test_extraction_rejects_path_traversal_and_corrupt_zip(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "do not extract")
    with pytest.raises(ValueError, match="unsafe path"):
        extract_part(archive, tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()
    archive.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        extract_part(archive, tmp_path / "destination")


def test_paddle_full_and_bucket_configs_are_explicit_and_preserve_template():
    template = {"Global": {"max_text_length": 25},
                "Architecture": {"Head": {"head_list": [{"CTCHead": {}}, {"NRTRHead": {"max_text_length": 25}}]}},
                "Train": {"sampler": {"name": "MultiScaleSampler"}, "loader": {}},
                "Eval": {"loader": {}}}
    full = training_config(template, "./data", 4096, 256)
    bucket = training_config(template, "./data", 320, 256, bucket_only=True)
    assert template["Global"]["max_text_length"] == 25
    assert full["Global"]["max_text_length"] == full["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] == 256
    for mode, split in (("Train", "train"), ("Eval", "val")):
        assert full[mode]["dataset"]["label_file_list"] == [f"./data/{split}.txt"]
        assert bucket[mode]["dataset"]["label_file_list"] == [f"./data/{split}-w320.txt"]
        assert full[mode]["dataset"]["transforms"][2] == {"RecResizeImg": {"image_shape": [3, 48, 4096], "padding": True}}
        assert "sampler" not in full[mode]


def test_large_metadata_roundtrip_and_checksum_failure(tmp_path):
    root = tmp_path / "dataset"
    (root / "images/000").mkdir(parents=True)
    (root / "images/000/a.jpg").write_bytes(b"small image fixture")
    (root / "labels.txt").write_text("images/000/a.jpg\tİl Müdürlüğü 2021\n", encoding="utf-8")
    payload = ('{"text":"Çiğdem, şüphe ve ışık"}\n' * 1000).encode("utf-8")
    (root / "metadata.jsonl").write_bytes(payload)
    (root / "audit.json").write_text('{"passed":true,"images_decoded_and_hashed":1}')
    (root / "paddle-preprocessing-audit.json").write_text('{"passed":true,"samples":1}')
    parts = tmp_path / "parts"
    manifest = package(root, parts, max_bytes=8192, prefix="turkish-ocr-200k")
    assert manifest["schema_version"] == 2 and manifest["image_count"] == 1
    assert all(p["bytes"] <= 8192 and p["file"].startswith("turkish-ocr-200k-") for p in manifest["parts"])
    destination = tmp_path / "download"
    for part in manifest["parts"]:
        extract_part(parts / part["file"], destination)
    dataset = destination / "dataset"
    record = manifest["assembled_files"][0]
    first = dataset / record["parts"][0]["file"]
    saved = first.read_bytes()
    first.write_bytes(b"bad" + saved[3:])
    with pytest.raises(ValueError, match="shard failed"):
        assemble_metadata(dataset, manifest)
    assert not (dataset / "metadata.jsonl").exists()
    first.write_bytes(saved)
    assemble_metadata(dataset, manifest)
    assert (dataset / "metadata.jsonl").read_bytes() == payload
    assert not list((dataset / "_metadata_parts").glob("*.part"))
    assemble_metadata(dataset, manifest)  # A complete verified download is resumable.


def test_metadata_reassembly_rejects_escape(tmp_path):
    record = {"file": "../escape", "bytes": 0, "sha256": "", "parts": []}
    with pytest.raises(ValueError, match="Unsafe"):
        assemble_metadata(tmp_path, {"assembled_files": [record]})
