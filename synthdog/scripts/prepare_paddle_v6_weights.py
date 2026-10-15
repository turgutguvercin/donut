#!/usr/bin/env python3
"""Remap official PP-OCRv6 medium pretrained CTC/NRTR weights to Turkish.

Weights are downloaded from the official training-model URL and SHA-256 pinned.
All feature tensors must match. Character indices, spaces, special tokens and
the upstream NRTR head's trailing extra output are explicitly transferred.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from paddle_v6_utils import (MODEL, COMMIT, MAX_LENGTH, VOCAB_AXES, WEIGHTS_URL,
    WEIGHTS_SHA256, characters, check_checkout, checksum, references, remap_axis,
    tokens, verified_download)


def architecture(template, chars):
    cfg = copy.deepcopy(template["Architecture"])
    if cfg["Backbone"] != {"name": "PPLCNetV4", "model_size": "medium"}:
        raise ValueError("Expected the v6 medium backbone")
    cfg["Head"]["out_channels_list"] = {"CTCLabelDecode": len(chars) + 1,
                                        "NRTRLabelDecode": len(chars) + 4}
    cfg["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] = MAX_LENGTH
    return cfg


def adapt(paddleocr, dictionary, cache, output, seed=20261015):
    check_checkout(paddleocr)
    sys.path.insert(0, str(paddleocr.resolve()))
    import paddle
    from ppocr.modeling.architectures import build_model
    refs = references(cache)
    source_path = verified_download(WEIGHTS_URL, cache / (MODEL + "_pretrained.pdparams"), WEIGHTS_SHA256)
    old, new = characters(cache / "ppocrv6_dict.txt"), characters(dictionary)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.set_device("cpu")
    model = build_model(architecture(yaml.safe_load(refs[MODEL + ".yml"]), new))
    initial = model.state_dict()
    source = paddle.load(str(source_path))
    if set(source) != set(initial):
        raise ValueError(f"Pretrained model keys differ: missing={set(initial)-set(source)}, extra={set(source)-set(initial)}")
    adapted, remapped, copied = {}, [], []
    for name, target in initial.items():
        value = source[name]
        value = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        if name in VOCAB_AXES:
            axis, head = VOCAB_AXES[name]
            value = remap_axis(value, target.numpy(), tokens(old, head), tokens(new, head), axis)
            remapped.append(name)
        elif list(value.shape) != list(target.shape):
            raise ValueError(f"Unexpected feature tensor shape mismatch: {name}")
        else:
            copied.append(name)
        adapted[name] = paddle.to_tensor(value, dtype=target.dtype)
    missing, unexpected = model.set_state_dict(adapted)
    if missing or unexpected:
        raise ValueError("Incomplete state-dict transfer")
    output = Path(str(output).removesuffix(".pdparams") + ".pdparams")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Choose a new output checkpoint path")
    paddle.save(model.state_dict(), str(output))
    shared, added = sorted(set(new) & set(old)), sorted(set(new) - set(old))
    required = set("çÇğĞıİöÖşŞüÜIi")
    if not required <= set(shared):
        raise ValueError("Not all Turkish letters were transferred from pretraining")
    report = {"passed": True, "model": MODEL, "paddleocr_commit": COMMIT,
        "source_url": WEIGHTS_URL, "source_sha256": WEIGHTS_SHA256,
        "target_dictionary_sha256": checksum(dictionary), "checkpoint_sha256": checksum(output),
        "feature_tensors_copied_exactly": len(copied), "vocabulary_tensors_remapped": remapped,
        "shared_characters_including_space": len(shared), "new_initialized_characters": added,
        "all_turkish_letters_transferred": True, "ctc_output_classes": len(tokens(new, "ctc")),
        "nrtr_encoder_classes": len(new) + 4, "nrtr_head_output_classes": len(tokens(new, "nrtr")),
        "nrtr_trailing_extra_output_preserved": True, "seed": seed,
        "model_training_performed": False}
    output.with_suffix(".transfer.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paddleocr", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path(".cache/paddleocr-v6-reference"))
    p.add_argument("--output", type=Path, default=Path("pretrained/PP-OCRv6_medium_rec_tr"))
    a = p.parse_args()
    print(json.dumps(adapt(a.paddleocr, a.dictionary, a.cache, a.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
