#!/usr/bin/env python3
"""Run the official PaddleOCR trainer with all Turkish width buckets together.

Run from the pinned PaddleOCR checkout, with this script's directory available.
The upstream source tree is not modified. This launcher is required even for
fixed-width training: upstream otherwise pools all training widths to 40 steps.
"""
from pathlib import Path
import json
import sys

from paddle_v6_utils import COMMIT, MODEL, check_checkout, checksum


def main():
    root = Path.cwd()
    check_checkout(root)
    sys.path.insert(0, str(root))
    import tools.train as trainer
    import tools.program as program
    from paddle_tr_v6_runtime import build_dataloader
    from paddle_tr_v6_pool import install_width_preserving_pool
    trainer.build_dataloader = build_dataloader
    program.build_dataloader = build_dataloader
    config, device, logger, writer = program.preprocess(is_train=True)
    install_width_preserving_pool(config)
    if config["Global"].get("model_name") != MODEL or config["Architecture"]["Backbone"] != {"name": "PPLCNetV4", "model_size": "medium"}:
        raise ValueError("Expected the PP-OCRv6 medium recognition configuration")
    checkpoint = config["Global"].get("checkpoints") or config["Global"].get("pretrained_model")
    if not checkpoint or not Path(str(checkpoint).removesuffix(".pdparams") + ".pdparams").is_file():
        raise ValueError("Create the vocabulary-remapped pretrained checkpoint first, or provide a compatible resume checkpoint")
    if not config["Global"].get("checkpoints"):
        path = Path(str(checkpoint).removesuffix(".pdparams") + ".pdparams")
        report = json.loads(path.with_suffix(".transfer.json").read_text())
        if (not report["passed"] or report["model"] != MODEL
                or report["checkpoint_sha256"] != checksum(path)
                or report["target_dictionary_sha256"] != checksum(config["Global"]["character_dict_path"])):
            raise ValueError("Pretrained checkpoint transfer report does not match the Turkish dictionary")
    logger.info("Pinned PaddleOCR %s; Turkish width buckets cover the complete selected split", COMMIT)
    trainer.main(config, device, logger, writer)


if __name__ == "__main__":
    main()
