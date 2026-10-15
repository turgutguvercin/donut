#!/usr/bin/env python3
"""Assign PP-OCRv5 buckets using measured full-model CTC capacity.

Image bytes, transcriptions and split membership remain unchanged. V3's v4
capacity assumption (stride four) must not be reused for v5 server (stride eight).
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil

from download_tr_dataset import checksum
import generate_tr_dataset as dataset


def choose_bucket(width, text, buckets, stride=8):
    steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
    required = max(width, stride * steps)
    try:
        return next(w for w in buckets if w >= required)
    except StopIteration:
        raise ValueError(f"No bucket has CTC capacity for {len(text)} characters / {steps} time steps")


def finalize(root, shape_report):
    probe = json.loads(shape_report.read_text())
    if not probe["passed"] or probe["model"] != "PP-OCRv5_server_rec" or probe["ctc_width_stride"] != 8:
        raise ValueError("A real PP-OCRv5 server shape probe is required")
    config_meta = json.loads((root / "metadata.json").read_text())
    config = config_meta["config"]
    actual_steps = {r["input_shape"][-1]: r["time_steps"] for r in probe["tested_widths"]}
    if set(actual_steps) != set(config["width_buckets"]):
        raise ValueError("Probe must cover every configured width")
    if not json.loads((root / "official-audit.json").read_text())["passed"]:
        raise ValueError("Complete dataset generation/audit first")
    report_path = root / "ctc-capacity-audit.json"
    before = checksum(root / "metadata.jsonl")
    if report_path.exists():
        prior = json.loads(report_path.read_text())
        if prior["result_metadata_sha256"] == before:
            return prior
        raise ValueError("Previously finalized metadata has changed")
    changes, widths, count, max_label, max_steps = Counter(), Counter(), 0, 0, 0
    temporary = root / "metadata-v5.jsonl.partial"
    with ExitStack() as stack:
        streams = {}
        for split in ("train", "val", "test"):
            for width in config["width_buckets"]:
                for subset in ("", "-recommended", "-challenge"):
                    streams[split, width, subset] = stack.enter_context((root / f"{split}{subset}-w{width}.txt").open("w", encoding="utf-8"))
        output = stack.enter_context(temporary.open("w", encoding="utf-8"))
        source = stack.enter_context((root / "metadata.jsonl").open(encoding="utf-8"))
        for line in source:
            row = json.loads(line)
            text = row["text"]
            if len(text) > config["max_text_length"] - 2:
                raise ValueError("Label exceeds NRTR start/end-aware capacity")
            old = row["width_bucket"]
            new = choose_bucket(row["width"], text, config["width_buckets"])
            steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
            if actual_steps[new] < steps:
                raise ValueError("Real model has insufficient CTC time steps")
            if new != old:
                changes[f"{old}->{new}"] += 1
            row["width_bucket"] = new
            record = row["file_name"] + "\t" + text + "\n"
            streams[row["split"], new, ""].write(record)
            subset = "-recommended" if row["recommended_for_standard_training"] else "-challenge"
            streams[row["split"], new, subset].write(record)
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            widths[new] += 1
            count += 1
            max_label, max_steps = max(max_label, len(text)), max(max_steps, steps)
    if count != config_meta["count"]:
        raise ValueError("Metadata count mismatch")
    temporary.replace(root / "metadata.jsonl")
    config["ctc_width_stride"] = 8
    config_meta["ctc_capacity_source"] = "Full official v5 server CPU forward outputs, model-shape-audit.json"
    config_meta["generator_sha256"][Path(__file__).name] = checksum(Path(__file__))
    dataset.write_json(root / "metadata.json", config_meta)
    dataset.write_json(root / "recipe.json", config)
    shutil.copy2(shape_report, root / "model-shape-audit.json")
    report = {"passed": True, "samples": count, "ctc_width_stride": 8, "image_bytes_changed": 0,
              "labels_changed": 0, "split_memberships_changed": 0,
              "bucket_changes": dict(changes), "samples_moved_to_wider_bucket": sum(changes.values()),
              "final_bucket_counts": dict(widths), "maximum_label_length": max_label,
              "nrtr_actual_character_capacity": config["max_text_length"] - 2,
              "maximum_ctc_steps_required_including_repeats": max_steps,
              "measured_ctc_steps_by_width": actual_steps,
              "input_metadata_sha256": before, "result_metadata_sha256": checksum(root / "metadata.jsonl"),
              "scope": "Every label checked against real full-model CTC output lengths and NRTR capacity; no model accuracy evaluated."}
    dataset.write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shape-report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.dataset, args.shape_report), indent=2), flush=True)
    # Rerun image/split/bucket audits after metadata and bucket-list changes.
    report = dataset.audit(args.dataset)
    report["ctc_width_stride"] = 8
    report["ctc_capacity_reference"] = "ctc-capacity-audit.json and model-shape-audit.json"
    dataset.write_json(args.dataset / "audit.json", report)


if __name__ == "__main__":
    main()
