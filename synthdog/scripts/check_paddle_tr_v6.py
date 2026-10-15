#!/usr/bin/env python3
"""Exercise the actual PP-OCRv6 medium network and Turkish data operators."""
import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from paddle_v6_utils import COMMIT, MODEL, WIDTHS, REFERENCES, characters, checksum, check_checkout
from prepare_paddle_v6_weights import architecture


def local_config(adapter, root, all_data=False):
    name = "rec_tr_v6_medium_all.yml" if all_data else "rec_tr_v6_medium.yml"
    config = yaml.safe_load((adapter / name).read_text())
    config["Global"]["character_dict_path"] = str(adapter / "turkish_dict.txt")
    for mode in ("Train", "Eval"):
        ds = config[mode]["dataset"]
        ds["data_dir"] = str(root)
        ds["label_file_list"] = [str(root / Path(p).name) for p in ds["label_file_list"]]
        config[mode]["loader"]["num_workers"] = 0
    return config


def shapes(paddleocr, dictionary, output):
    import paddle
    from ppocr.modeling.architectures import build_model
    import subprocess
    content = subprocess.check_output(["git", "show", "HEAD:configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml"], cwd=paddleocr)
    import hashlib
    if hashlib.sha256(content).hexdigest() != REFERENCES[MODEL + ".yml"][1]:
        raise ValueError("Unexpected model configuration")
    chars = characters(dictionary)
    np.random.seed(20261015)
    paddle.seed(20261015)
    paddle.set_device("cpu")
    model = build_model(architecture(yaml.safe_load(content), chars))
    model.eval()
    rows = []
    for width in WIDTHS:
        start = time.monotonic()
        with paddle.no_grad():
            out = model(paddle.zeros([1, 3, 48, width]))
        if list(out.shape) != [1, width // 8, len(chars) + 1] or not bool(paddle.isfinite(out).all()):
            raise ValueError("Unexpected v6 CTC output/capacity")
        row = {"input_shape": [1, 3, 48, width], "ctc_shape": list(out.shape),
               "time_steps": out.shape[1], "elapsed_seconds": round(time.monotonic() - start, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"passed": True, "model": MODEL, "paddleocr_commit": COMMIT,
        "config_sha256": hashlib.sha256(content).hexdigest(), "paddle_version": paddle.__version__,
        "device": "cpu", "ctc_width_stride": 8, "tested_widths": rows,
        "pretrained_weights_loaded": False, "accuracy_evaluated": False,
        "scope": "Full pinned model, random-weight eval forwards; dimensions only."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def data(root, adapter, output):
    from paddle_tr_v6_runtime import TurkishBucketDataset, TurkishBucketSampler
    compatibility = json.loads((adapter / "dataset-compatibility.json").read_text())
    for name, sha in compatibility["required_base_file_sha256"].items():
        if checksum(root / name) != sha:
            raise ValueError(f"Base file changed: {name}")
    config = local_config(adapter, root, all_data=True)
    total, per_split = 0, {}
    logger = logging.getLogger(__name__)
    for split in ("train", "val", "test"):
        mode = "Train" if split == "train" else "Eval"
        config[mode]["dataset"]["label_file_list"] = [str(root / f"{split}.txt")]
        dataset = TurkishBucketDataset(config, mode, logger)
        sampler = TurkishBucketSampler(dataset, config["TurkishWidthBuckets"]["batch_sizes"], True, 20261015)
        first = [i for batch in sampler for i in batch]
        second = [i for batch in sampler for i in batch]
        if sorted(first) != list(range(len(dataset))) or sorted(second) != sorted(first) or (len(dataset) > 1 and first == second):
            raise ValueError("Epoch sampler loses, duplicates or fails to reshuffle samples")
        for batch in sampler:
            if len({dataset.records[i][2] for i in batch}) != 1:
                raise ValueError("Mixed image widths in a batch")
        for i in range(len(dataset)):
            dataset[i]  # Actual DecodeImage/MultiLabelEncode/RecResizeImg/KeepKeys.
            total += 1
            if total % 10000 == 0:
                print(f"Actual v6 operators checked {total} images/labels", flush=True)
        per_split[split] = len(dataset)
    if total != compatibility["images"]:
        raise ValueError("Incomplete dataset validation")
    report = {"passed": True, "model": MODEL, "paddleocr_commit": COMMIT,
        "samples": total, "splits": per_split, "native_paddleocr_operators": True,
        "labels_rejected_truncated_or_changed": 0, "horizontal_squeezing": 0,
        "ctc_and_nrtr_utf8_roundtrips_checked": total,
        "sampler_two_epochs_exact_coverage": True, "partial_batches_retained": True,
        "sampler_reshuffles_between_epochs": True, "all_batches_same_width": True,
        "scope": "All samples passed native upstream decoding/encoding/normalization; no accuracy claim."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def smoke(root, adapter, checkpoint, output):
    import paddle
    from ppocr.modeling.architectures import build_model
    from ppocr.losses import build_loss
    from paddle_tr_v6_runtime import TurkishBucketDataset, build_dataloader
    from paddle_v6_utils import WEIGHTS_SHA256
    from paddle_tr_v6_pool import install_width_preserving_pool
    report_path = checkpoint.with_suffix(".transfer.json")
    transfer = json.loads(report_path.read_text())
    if (not transfer["passed"] or transfer["source_sha256"] != WEIGHTS_SHA256
            or transfer["checkpoint_sha256"] != checksum(checkpoint)
            or transfer["target_dictionary_sha256"] != checksum(adapter / "turkish_dict.txt")):
        raise ValueError("Checkpoint transfer report does not match these weights/dictionary")
    config = local_config(adapter, root)
    install_width_preserving_pool(config)
    config["Global"]["use_gpu"] = False
    chars = characters(adapter / "turkish_dict.txt")
    paddle.set_device("cpu")
    paddle.seed(20261015)
    model = build_model(architecture(config, chars))
    missing, unexpected = model.set_state_dict(paddle.load(str(checkpoint)))
    if missing or unexpected:
        raise ValueError("Adapted checkpoint did not load completely")
    model.train()
    loss_fn = build_loss(copy.deepcopy(config["Loss"]))
    optimizer = paddle.optimizer.Adam(learning_rate=1e-5, parameters=model.parameters(),
        grad_clip=paddle.nn.ClipGradByGlobalNorm(5))
    dataset = TurkishBucketDataset(config, "Train")
    rows = []
    for width in WIDTHS:
        index = max(dataset.groups[width], key=lambda i: len(dataset.records[i][1]))
        batch = [paddle.to_tensor(np.expand_dims(v, 0)) for v in dataset[index]]
        before = model.head.ctc_head.fc.weight.numpy().copy()
        start = time.monotonic()
        prediction = model(batch[0], data=batch[1:])
        label = dataset.records[index][1]
        required = len(label) + sum(a == b for a, b in zip(label, label[1:]))
        if prediction["ctc"].shape[1] != width // 8 or prediction["ctc"].shape[1] < required:
            raise ValueError("Training forward lost CTC capacity; check the backbone pooling override")
        losses = loss_fn(prediction, batch)
        if not all(bool(paddle.isfinite(value).all()) for value in losses.values()):
            raise ValueError("Non-finite training loss")
        losses["loss"].backward()
        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters or not all(bool(paddle.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("Missing or non-finite gradients")
        if float(losses["CTCLoss"]) <= 0 or not bool(paddle.any(model.head.ctc_head.fc.weight.grad != 0)):
            raise ValueError("CTC training was silently skipped")
        optimizer.step()
        optimizer.clear_grad()
        if np.array_equal(before, model.head.ctc_head.fc.weight.numpy()):
            raise ValueError("Optimizer did not update weights")
        row = {"width": width, "characters": len(dataset.records[index][1]),
            "ctc_steps": prediction["ctc"].shape[1],
            "minimum_ctc_steps": required,
            "ctc_loss": float(losses["CTCLoss"]), "nrtr_loss": float(losses["NRTRLoss"]),
            "finite_gradients": True, "weights_updated": True,
            "elapsed_seconds": round(time.monotonic() - start, 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Also exercise the real collated DataLoader, not just individual samples.
    config["TurkishWidthBuckets"]["batch_sizes"] = {w: 2 for w in WIDTHS}
    loader = build_dataloader(config, "Eval", paddle.CPUPlace(), logging.getLogger(__name__))
    first = next(iter(loader))
    assert first[0].shape[0] == 2 and first[0].shape[1:3] == [3, 48]
    report = {"passed": True, "model": MODEL, "paddleocr_commit": COMMIT,
        "pretrained_weights_loaded": True, "optimizer_steps": len(rows), "device": "cpu",
        "width_preserving_training_pooling": True, "nonzero_ctc_gradients_all_widths": True,
        "widths": rows, "native_dataloader_collation_passed": True,
        "maximum_label_length_exercised": max(r["characters"] for r in rows),
        "accuracy_evaluated": False, "production_training_performed": False,
        "scope": "Seven short CTC+NRTR backward/Adam checks on real recommended crops; not a trained OCR deliverable."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("shapes", "data", "smoke"))
    p.add_argument("--paddleocr", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    check_checkout(a.paddleocr)
    sys.path.insert(0, str(a.paddleocr.resolve()))
    adapter = a.adapter or a.dataset / "paddle-v6-medium"
    if a.mode == "shapes":
        report = shapes(a.paddleocr, a.dataset / "turkish_dict.txt", a.output)
    elif a.mode == "data":
        report = data(a.dataset, adapter, a.output)
    else:
        if a.checkpoint is None:
            p.error("smoke mode requires --checkpoint")
        report = smoke(a.dataset, adapter, a.checkpoint, a.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
