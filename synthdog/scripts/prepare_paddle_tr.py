#!/usr/bin/env python3
"""Write PP-OCRv4 training configs and check data with pinned upstream operators.

This checks actual PaddleOCR preprocessing and label encoding, without claiming
to train or evaluate a model. Downloaded source is SHA-256 verified before the
selected, dependency-free upstream operators are compiled.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import yaml

COMMIT = "2172c14bf65ceacf37a839ebbe61117a6339a865"
REFERENCES = {
    "rec_img_aug.py": ("ppocr/data/imaug/rec_img_aug.py", "62216925c7d74ec006e91d9c5c7f5993874460283ed911b97935aead19dd9082"),
    "label_ops.py": ("ppocr/data/imaug/label_ops.py", "e1ddcbb8e844d21682291ed6a37663f55159591b8871369efd923a7e1b19d622"),
    "en_PP-OCRv4_rec.yml": ("configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml", "aaa6d9e91f63bf80f1fbde4ab7e826c6b40a57f81d4e6a4c07b5b43208787cc6"),
}

# YAML configurations are adapted from Apache-2.0-licensed PaddleOCR. Preserve
# its complete license with distributed configurations, not just a URL header.
LICENSE_URL = f"https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{COMMIT}/LICENSE"


def reference_files(cache):
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (remote, checksum) in REFERENCES.items():
        path = cache / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{COMMIT}/{remote}"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Upstream reference checksum mismatch: {path}")
        result[name] = content.decode("utf-8")
    return result


def upstream_operators(references):
    names = {"resize_norm_img", "BaseRecLabelEncode", "CTCLabelEncode", "NRTRLabelEncode", "MultiLabelEncode"}
    nodes = []
    for name in ("rec_img_aug.py", "label_ops.py"):
        tree = ast.parse(references[name], filename=name)
        nodes.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
    if {node.name for node in nodes} != names:
        raise ValueError("Pinned upstream operator definitions changed")
    namespace = {"cv2": cv2, "np": np, "math": math, "copy": copy}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "verified_paddleocr_operators", "exec"), namespace)
    return namespace


def training_config(template, data_dir, width, max_length, bucket_only=False):
    config = copy.deepcopy(template)
    config["Global"].update(character_dict_path=f"{data_dir}/turkish_dict.txt", max_text_length=max_length,
                            save_model_dir=f"./output/rec_tr_w{width}", distributed=False,
                            infer_img=None, pretrained_model=None, checkpoints=None)
    config["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] = max_length
    suffix = f"-w{width}" if bucket_only else ""
    for mode, split in (("Train", "train"), ("Eval", "val")):
        config[mode].pop("sampler", None)
        config[mode]["dataset"] = {
            "name": "SimpleDataSet", "data_dir": data_dir,
            "label_file_list": [f"{data_dir}/{split}{suffix}.txt"],
            "transforms": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                {"RecResizeImg": {"image_shape": [3, 48, width], "padding": True}},
                {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
            ],
        }
        config[mode]["loader"].update(batch_size_per_card=max(1, min(32, 8192 // width)),
                                      num_workers=2, shuffle=mode == "Train", drop_last=False)
    return config


def prepare(root, references, data_dir):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    recipe = metadata["config"]
    output = root / "paddle"
    output.mkdir(exist_ok=True)
    template = yaml.safe_load(references["en_PP-OCRv4_rec.yml"])
    header = ("# Adapted from PaddleOCR (Apache-2.0), commit " + COMMIT + "\n"
              "# https://github.com/PaddlePaddle/PaddleOCR/blob/" + COMMIT + "/LICENSE\n"
              "# Set pretrained_model to a compatible checkpoint before fine-tuning.\n")
    configs = {"rec_tr_all.yml": training_config(template, data_dir, recipe["max_width"], recipe["max_text_length"])}
    for width in recipe["width_buckets"]:
        configs[f"rec_tr_w{width}.yml"] = training_config(template, data_dir, width, recipe["max_text_length"], True)
    for name, config in configs.items():
        (output / name).write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output / "README.txt").write_text(
        f"PaddleOCR PP-OCRv4 configuration reference: release/2.8 commit {COMMIT}.\n"
        f"Place this dataset at {data_dir}, relative to the PaddleOCR checkout, or regenerate the configs with --data-dir.\n"
        "rec_tr_all.yml covers ALL training and validation samples, right-padded to 4096. Its small batch size is a starting point.\n"
        "rec_tr_w*.yml uses ONLY the matching width bucket and its validation subset. These are not seven automatic joint-training stages.\n"
        "For efficient joint training, use batches of similar widths and retain all bucket lists; a custom bucket sampler is not installed here.\n"
        "The default pretrained_model is null. Choose compatible pretrained weights for fine-tuning; a changed alphabet requires a compatible/new output head.\n"
        "Do not use a static 320-pixel deployment/export input for these long lines. Configure matching dynamic/fixed width at inference.\n"
        "Max label length is 256 in Global and NRTRHead. Use the supplied case-sensitive dictionary with use_space_char=true.\n"
        "Extra RecAug/RecConAug are disabled because the samples already contain planned augmentations and concatenation could exceed label limits.\n"
        "Validation below runs actual pinned preprocessing/label operators; it is not a model-training or OCR-accuracy test.\n",
        encoding="utf-8")
    return configs


def check(root, references):
    operators = upstream_operators(references)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    max_length = metadata["config"]["max_text_length"]
    encoder = operators["MultiLabelEncode"](max_text_length=max_length,
               character_dict_path=str(root / "turkish_dict.txt"), use_space_char=True, gtc_encode="NRTRLabelEncode")
    count, one_pixel_rounding = 0, 0
    cv2.setNumThreads(1)
    with (root / "metadata.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            image = cv2.imread(str(root / row["file_name"]))
            if image is None or image.shape != (48, row["width"], 3):
                raise ValueError("Paddle/OpenCV could not decode the expected image shape")
            # Encoding does not need to copy a full image twice for this audit.
            encoded = encoder({"label": row["text"], "image": np.zeros((1, 1, 3), dtype="uint8")})
            if encoded is None or int(encoded["length"]) != len(row["text"]):
                raise ValueError("Upstream encoder rejected/truncated a label")
            decoded = "".join(encoder.ctc_encode.character[int(index)] for index in encoded["label_ctc"][:len(row["text"])] )
            if decoded != row["text"]:
                raise ValueError("Paddle CTC label round trip changed Turkish characters")
            normalized, valid_ratio = operators["resize_norm_img"](image, [3, 48, row["width_bucket"]])
            actual_width = round(valid_ratio * row["width_bucket"])
            if normalized.shape != (3, 48, row["width_bucket"]) or not row["width"] <= actual_width <= row["width"] + 1:
                raise ValueError("Paddle preprocessing would squeeze an image")
            if not np.isfinite(normalized).all() or np.any(normalized[:, :, actual_width:] != 0):
                raise ValueError("Invalid normalization or padding")
            one_pixel_rounding += actual_width != row["width"]
            count += 1
            if count % 10000 == 0:
                print(f"Paddle operators verified {count} images/labels", flush=True)
    if count != metadata["count"]:
        raise ValueError("Incomplete Paddle preprocessing audit")
    report = {"passed": True, "samples": count, "paddleocr_commit": COMMIT,
              "reference_sha256": {name: value[1] for name, value in REFERENCES.items()},
              "operators": ["resize_norm_img", "MultiLabelEncode/CTCLabelEncode/NRTRLabelEncode"],
              "height": 48, "maximum_width": metadata["config"]["max_width"], "maximum_label_length_setting": max_length,
              "labels_truncated_or_characters_dropped": 0, "images_horizontally_squeezed": 0,
              "images_with_one_pixel_upstream_ceil_rounding": one_pixel_rounding,
              "model_training_or_accuracy_evaluated": False,
              "scope": "Pinned upstream preprocessing/encoding operators compiled from checksum-verified source; Paddle neural-network runtime was not exercised."}
    (root / "paddle-preprocessing-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/paddleocr-reference"))
    parser.add_argument("--data-dir", default="./train_data/turkish-recognition-100k-v1")
    parser.add_argument("--configs-only", action="store_true")
    args = parser.parse_args()
    references = reference_files(args.cache)
    prepare(args.dataset, references, args.data_dir)
    license_path = args.cache / "PaddleOCR-LICENSE.txt"
    if not license_path.exists():
        with urllib.request.urlopen(LICENSE_URL, timeout=60) as response:
            license_path.write_bytes(response.read())
    (args.dataset / "paddle/LICENSE-PaddleOCR.txt").write_bytes(license_path.read_bytes())
    if not args.configs_only:
        print(json.dumps(check(args.dataset, references), indent=2))


if __name__ == "__main__":
    main()
