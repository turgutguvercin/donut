#!/usr/bin/env python3
"""Attach PP-OCRv6 medium training settings to the verified Turkish v4 dataset.

The base images, labels, splits, dictionary and v5 configurations are preserved.
The v6 overlay reuses width lists only after measuring the real v6 architecture.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import yaml

from paddle_v6_utils import (COMMIT, MODEL, WIDTHS, BATCH_SIZES, MAX_LENGTH,
                             REFERENCES, characters, checksum, choose_bucket, references)


def training_config(template, data_dir, adapter_dir, bucketed=True, recommended=True):
    if template["Architecture"]["Backbone"] != {"name": "PPLCNetV4", "model_size": "medium"}:
        raise ValueError("Expected the pinned v6 medium architecture")
    config = copy.deepcopy(template)
    config["Global"].update(model_name=MODEL, max_text_length=MAX_LENGTH,
        character_dict_path=f"{adapter_dir}/turkish_dict.txt", use_space_char=True,
        distributed=False, epoch_num=30, seed=20261015, infer_img=None,
        pretrained_model="./pretrained/PP-OCRv6_medium_rec_tr", checkpoints=None,
        save_model_dir="./output/PP-OCRv6_medium_rec_tr", save_epoch_step=1,
        d2s_train_image_shape=[3, 48, 4096], cal_metric_during_train=False,
        uniform_output_enabled=False)
    config["TurkishRecognitionPooling"] = {"enabled": True,
        "mode": "inference_pool_train_and_eval", "ctc_width_stride": 8}
    config["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] = MAX_LENGTH
    config["Optimizer"]["lr"].update(learning_rate=0.0001, warmup_epoch=1)
    config["Metric"].update(ignore_space=False, is_filter=False)
    for mode, split in (("Train", "train"), ("Eval", "val")):
        config[mode].pop("sampler", None)
        suffix = "-recommended" if recommended else ""
        config[mode]["dataset"] = {
            "name": "SimpleDataSet", "data_dir": data_dir,
            "label_file_list": [f"{data_dir}/{split}{suffix}.txt"], "ratio_list": [1.0],
            "transforms": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                {"RecResizeImg": {"image_shape": [3, 48, 4096], "padding": True}},
                {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
            ],
        }
        config[mode]["loader"].update(batch_size_per_card=1, drop_last=False,
                                     num_workers=2, shuffle=mode == "Train")
    if bucketed:
        config["TurkishWidthBuckets"] = {"enabled": True, "widths": list(WIDTHS),
            "batch_sizes": dict(BATCH_SIZES), "seed": 20261015,
            "launcher": "train_paddle_tr_v6.py"}
    return config


def prepare(root, out, shape_path, refs, data_dir):
    probe = json.loads(shape_path.read_text())
    if (not probe["passed"] or probe["model"] != MODEL or probe["paddleocr_commit"] != COMMIT
            or probe["config_sha256"] != REFERENCES[MODEL + ".yml"][1]):
        raise ValueError("A verified full PP-OCRv6_medium_rec shape probe is required")
    measured = {r["input_shape"][-1]: r["time_steps"] for r in probe["tested_widths"]}
    if set(measured) != set(WIDTHS) or any(r["input_shape"][1:3] != [3, 48] for r in probe["tested_widths"]):
        raise ValueError("Shape probe must cover every width at height 48")
    meta = json.loads((root / "metadata.json").read_text())
    if not json.loads((root / "audit.json").read_text())["passed"]:
        raise ValueError("Base dataset audit must pass")
    charset = set(characters(root / "turkish_dict.txt"))
    split_counts, quality_counts, buckets = Counter(), Counter(), Counter()
    expected = {}
    with (root / "metadata.jsonl").open(encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            label = row["text"]
            if len(label) > MAX_LENGTH - 2 or set(label) - charset or row["height"] != 48:
                raise ValueError("Unsupported label or image height")
            w = choose_bucket(row["width"], label, measured)
            if w != row["width_bucket"]:
                raise ValueError("Base width lists differ from measured v6 capacity; regenerate lists before reuse")
            if row["file_name"] in expected:
                raise ValueError("Duplicate sample path")
            expected[row["file_name"]] = (label, row["split"], w, row["recommended_for_standard_training"])
            split_counts[row["split"]] += 1
            quality_counts[row["split"], row["recommended_for_standard_training"]] += 1
            buckets[w] += 1
    if len(expected) != meta["count"]:
        raise ValueError("Dataset count mismatch")
    required = {name: checksum(root / name) for name in ("metadata.jsonl", "turkish_dict.txt", "labels.txt")}
    for split in ("train", "val", "test"):
        for suffix, quality in (("", None), ("-recommended", True), ("-challenge", False)):
            target = {p: d for p, d in expected.items() if d[1] == split and (quality is None or d[3] == quality)}
            for width in (None, *WIDTHS):
                filename = f"{split}{suffix}" + (f"-w{width}" if width else "") + ".txt"
                lines = (root / filename).read_text(encoding="utf-8").splitlines()
                seen = set()
                for record in lines:
                    name, label = record.split("\t", 1)
                    if name in seen or name not in target or target[name][0] != label or (width and target[name][2] != width):
                        raise ValueError(f"Inconsistent label/bucket list: {filename}")
                    seen.add(name)
                if seen != {p for p, d in target.items() if width is None or d[2] == width}:
                    raise ValueError(f"Missing rows: {filename}")
                required[filename] = checksum(root / filename)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "turkish_dict.txt", out / "turkish_dict.txt")
    shutil.copy2(shape_path, out / "model-shape-audit.json")
    template = yaml.safe_load(refs[MODEL + ".yml"])
    adapter_dir = f"{data_dir}/paddle-v6-medium"
    for name, bucketed, recommended in (("rec_tr_v6_medium.yml", True, True),
            ("rec_tr_v6_medium_all.yml", True, False), ("rec_tr_v6_medium_fixed.yml", False, True)):
        config = training_config(template, data_dir, adapter_dir, bucketed, recommended)
        (out / name).write_text(f"# PaddleOCR Apache-2.0 reference: {COMMIT}\n" +
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "LICENSE-PaddleOCR.txt").write_text(refs["LICENSE"], encoding="utf-8")
    report = {"passed": True, "model": MODEL, "paddleocr_commit": COMMIT, "images": len(expected),
        "base_images_labels_and_splits_changed": 0, "base_width_buckets_changed": 0,
        "height": 48, "maximum_width": max(WIDTHS), "ctc_width_stride": probe["ctc_width_stride"],
        "maximum_text_length_setting": MAX_LENGTH, "nrtr_actual_character_capacity": MAX_LENGTH - 2,
        "dictionary_symbols_including_space": len(charset), "split_counts": dict(split_counts),
        "recommended_counts": {s: quality_counts[s, True] for s in split_counts},
        "challenge_counts": {s: quality_counts[s, False] for s in split_counts},
        "width_counts": dict(buckets), "required_base_file_sha256": required,
        "all_samples_checked_against_real_v6_ctc_capacity": len(expected),
        "all_full_recommended_challenge_and_width_lists_checked": True}
    (out / "dataset-compatibility.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--shape-report", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path(".cache/paddleocr-v6-reference"))
    p.add_argument("--data-dir", default="./train_data/turkish-recognition-200k-v4")
    a = p.parse_args()
    report = prepare(a.dataset, a.output or a.dataset / "paddle-v6-medium", a.shape_report, references(a.cache), a.data_dir)
    print(json.dumps({k: v for k, v in report.items() if k != "required_base_file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
