#!/usr/bin/env python3
"""Check Turkish font coverage/rendering and generated OCR metadata."""
import argparse
import json
from pathlib import Path
from fontTools.ttLib import TTFont
from PIL import Image
from synthtiger.layers import TextLayer

CHARACTERS = "çğışöüÇĞİŞÖÜıI"


def validate_fonts(directory):
    fonts = sorted(directory.glob("*.ttf"))
    assert fonts, "No fonts found"
    for path in fonts:
        with TTFont(path) as font:
            cmap = font.getBestCmap()
            assert all(ord(c) in cmap and cmap[ord(c)] != ".notdef" for c in CHARACTERS), path
        for char in CHARACTERS:
            layer = TextLayer(char, path=str(path), size=48)
            assert layer.image[..., 3].max() > 0, (path, char)


def validate_output(root, count):
    names, labels = set(), []
    for metadata in root.glob("*/metadata.jsonl"):
        for line in metadata.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            label = json.loads(row["ground_truth"])["gt_parse"]["text_sequence"]
            assert isinstance(label, str) and label.strip() and "\ufffd" not in label
            path = metadata.parent / row["file_name"]
            assert path not in names, f"Duplicate image: {path}"
            names.add(path)
            with Image.open(path) as image:
                image.verify()
            labels.append(label)
    assert len(names) == count, (len(names), count)
    assert set(root.glob("*/*.jpg")) == names, "Unlabelled images"
    missing = set(CHARACTERS) - set("".join(labels))
    assert not missing, f"Characters absent from sample labels: {missing}"
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    validate_fonts(Path("resources/font/en"))
    labels = validate_output(args.output, args.count)
    print(f"Validated {len(labels)} images and UTF-8 OCR labels; all Turkish characters rendered and represented.")
