#!/usr/bin/env python3
"""Prepare PP-OCRv5_server_rec configs and audit actual pinned label/resize ops.

This does not train a neural network. Full-width padding is a correctness-first
configuration; the per-width files are subsets, not an automatic bucket sampler.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import yaml

import prepare_paddle_tr as shared

COMMIT = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"
REFERENCES = {
    "label_ops.py": ("ppocr/data/imaug/label_ops.py", "89c75bfb515c0d8ea3e8ebb5728335fcb8032e8d2e3e3060eb059527e7df61c7"),
    "rec_img_aug.py": ("ppocr/data/imaug/rec_img_aug.py", "77607eb2a5e6e5d01829565a97161845ecd85a35a4d827435198f3ed4a32f9ad"),
    "PP-OCRv5_server_rec.yml": ("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml", "d38c678ec4075aa0c65750d2759ac4396694f6ffb29004b127e0e8b3d11fcef3"),
    "LICENSE": ("LICENSE", "3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e"),
}


def reference_files(cache):
    cache.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, (path, expected) in REFERENCES.items():
        target = cache / name
        if not target.exists():
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{COMMIT}/{path}", timeout=45) as response:
                target.write_bytes(response.read())
        content = target.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Pinned PaddleOCR reference mismatch: {name}")
        result[name] = content.decode("utf-8")
    return result


def training_config(template, data_dir, width, max_length=256, bucket_only=False, recommended=True):
    config = shared.training_config(template, data_dir, width, max_length, bucket_only)
    config["Global"].update(model_name="PP-OCRv5_server_rec", d2s_train_image_shape=[3, 48, width],
                            save_model_dir=f"./output/PP-OCRv5_server_rec_tr_w{width}")
    # Do not silently use the v5 template's 320px multi-scale sampler or 25-char
    # NRTR budget. Check every vocabulary-dependent layer when loading weights.
    for mode, split in (("Train", "train"), ("Eval", "val")):
        suffix = "-recommended" if recommended else ""
        suffix += f"-w{width}" if bucket_only else ""
        config[mode]["dataset"]["label_file_list"] = [f"{data_dir}/{split}{suffix}.txt"]
        config[mode]["loader"]["batch_size_per_card"] = max(1, min(16, 4096 // width))
    return config


def prepare(root, references, data_dir):
    recipe = json.loads((root / "metadata.json").read_text())["config"]
    template = yaml.safe_load(references["PP-OCRv5_server_rec.yml"])
    out = root / "paddle-v5"
    out.mkdir(exist_ok=True)
    configs = {
        "rec_tr_recommended.yml": training_config(template, data_dir, recipe["max_width"], recipe["max_text_length"]),
        "rec_tr_all.yml": training_config(template, data_dir, recipe["max_width"], recipe["max_text_length"], recommended=False),
    }
    for width in recipe["width_buckets"]:
        configs[f"rec_tr_w{width}.yml"] = training_config(template, data_dir, width, recipe["max_text_length"], bucket_only=True)
    for name, config in configs.items():
        header = f"# PaddleOCR PP-OCRv5_server_rec, Apache-2.0, commit {COMMIT}\n"
        (out / name).write_text(header + yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "LICENSE-PaddleOCR.txt").write_text(references["LICENSE"])
    (out / "README.md").write_text(f"""# Turkish PP-OCRv5_server_rec data configuration

Pinned official PaddleOCR commit: {COMMIT}. Architecture: SVTR_HGNet / PPHGNetV2_B4.
`rec_tr_recommended.yml` uses quality-filtered training/validation lists.
`rec_tr_all.yml` includes ALL images, including intentionally ambiguous challenge crops.
Both use width 4096 and batch size 1 as a correctness-first starting point, not a Colab memory guarantee.
`rec_tr_w*.yml` each uses only ONE width subset; they do not automatically train all buckets together.
For efficient full-data training, implement a width-aware sampler and keep every bucket represented.

Global, NRTRHead and label encoders use max_text_length=256. The pinned NRTR encoder reserves
start/end tokens, so it accepts at most 254 actual characters. The official default 25 accepts 23.
Images remain 48px high with proportional widths; right padding does not squeeze long lines.
`d2s_train_image_shape` matches the selected width. Use a matching export/inference width policy.

Use the official Chinese/English PP-OCRv5_server_rec training checkpoint:
https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
`pretrained_model` is deliberately unset until a correct weight-transfer step is performed.
The new UTF-8 dictionary changes character indices/class counts. Preserve compatible feature weights;
explicitly remap shared character rows or initialize affected CTC/NRTR output/embedding layers.
Simply swapping dictionaries or accepting unchecked shape mismatches is not a fine-tuning workflow.

RecAug/RecConAug are omitted because these images already contain measured degradation and long labels.
The upstream RecAug DOES contain blur; absence of blur in that operator is not the problem being addressed.
The accompanying audit executes checksum-pinned resize and CTC/NRTR encoding definitions on every sample.
It is not model training, an OCR accuracy test, or evidence that the model will fit every GPU.
""", encoding="utf-8")
    return configs


def check(root, references):
    capacity = json.loads((root / "ctc-capacity-audit.json").read_text())
    if not capacity["passed"] or capacity["ctc_width_stride"] != 8:
        raise ValueError("Finalize buckets against the measured v5 stride-eight capacity first")
    # Reuse the existing complete decode/resize/CTC audit with pinned v5 source.
    previous = shared.COMMIT, shared.REFERENCES
    shared.COMMIT, shared.REFERENCES = COMMIT, REFERENCES
    try:
        report = shared.check(root, references)
    finally:
        shared.COMMIT, shared.REFERENCES = previous
    operators = shared.upstream_operators(references)
    max_length = json.loads((root / "metadata.json").read_text())["config"]["max_text_length"]
    dictionary = str(root / "turkish_dict.txt")
    encoder = operators["NRTRLabelEncode"](max_length, dictionary, use_space_char=True)
    for length, accepted in ((max_length - 2, True), (max_length - 1, False), (max_length, False)):
        result = encoder({"label": "a" * length})
        if (result is not None) != accepted:
            raise ValueError("NRTR start/end boundary differs from configured expectation")
    default = operators["NRTRLabelEncode"](25, dictionary, use_space_char=True)
    assert default({"label": "a" * 23}) is not None
    assert default({"label": "a" * 24}) is None
    count = 0
    for line in (root / "metadata.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        encoded = encoder({"label": row["text"]})
        if encoded is None or len(row["text"]) > max_length - 2:
            raise ValueError("NRTR rejected label")
        tokens = encoded["label"]
        if tokens[0] != 2 or tokens[len(row["text"]) + 1] != 3:
            raise ValueError("Missing NRTR start/end marker")
        decoded = "".join(encoder.character[int(i)] for i in tokens[1:len(row["text"]) + 1])
        if decoded != row["text"]:
            raise ValueError("NRTR label round trip lost characters")
        if np.any(tokens[len(row["text"]) + 2:] != 0):
            raise ValueError("Incorrect NRTR padding")
        count += 1
    report.update(model_configuration="PP-OCRv5_server_rec", nrtr_actual_character_capacity=max_length - 2,
                  nrtr_start_end_and_roundtrip_samples=count, nrtr_default_25_accepts_23_rejects_24=True,
                  vocabulary_dependent_pretrained_weights_require_adaptation=True,
                  ctc_width_stride=8, all_samples_checked_against_real_ctc_capacity=capacity["samples"])
    (root / "paddle-preprocessing-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/paddleocr-v5-reference"))
    parser.add_argument("--data-dir", default="./train_data/turkish-recognition-200k-v4")
    args = parser.parse_args()
    refs = reference_files(args.cache)
    prepare(args.dataset, refs, args.data_dir)
    print(json.dumps(check(args.dataset, refs), indent=2))


if __name__ == "__main__":
    main()
