#!/usr/bin/env python3
"""Measure real PP-OCRv5_server_rec CTC time steps on CPU, without OCR claims."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import paddle
import yaml

COMMIT = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddleocr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.paddleocr, text=True).strip()
    if commit != COMMIT:
        raise ValueError("Use the pinned official PaddleOCR checkout")
    sys.path.insert(0, str(args.paddleocr.resolve()))
    from ppocr.modeling.architectures import build_model
    path = args.paddleocr / "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"
    cfg = yaml.safe_load(path.read_text())["Architecture"]
    cfg["Head"]["out_channels_list"] = {"CTCLabelDecode": 222, "NRTRLabelDecode": 225}
    cfg["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] = 256
    paddle.seed(20261016)
    paddle.set_device("cpu")
    model = build_model(copy.deepcopy(cfg))
    model.eval()
    results = []
    for width in (320, 640, 960, 1280, 1920, 2560, 4096):
        started = time.monotonic()
        with paddle.no_grad():
            prediction = model(paddle.zeros([1, 3, 48, width]))
        shape = list(prediction.shape)
        if shape != [1, width // 8, 222] or not bool(paddle.isfinite(prediction).all()):
            raise ValueError(f"Unexpected real CTC output at width {width}: {shape}")
        row = {"input_shape": [1, 3, 48, width], "ctc_shape": shape,
               "time_steps": shape[1], "elapsed_seconds": round(time.monotonic() - started, 3)}
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"passed": True, "paddleocr_commit": commit, "paddle_version": paddle.__version__,
              "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "model": "PP-OCRv5_server_rec", "device": "cpu", "training_performed": False,
              "pretrained_weights_loaded": False, "accuracy_evaluated": False,
              "scope": "Actual full official recognition architecture, randomly initialized, eval forwards; shapes/capacity only.",
              "ctc_width_stride": 8, "tested_widths": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
