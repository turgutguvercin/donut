#!/usr/bin/env python3
"""Package the audited V6 overlay without copying any images or model binaries."""
import argparse
import json
from pathlib import Path
import subprocess
import zipfile

from paddle_v6_utils import MODEL, WIDTHS, checksum

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "paddle_v6_utils.py", "prepare_paddle_tr_v6.py", "paddle_tr_v6_runtime.py",
    "paddle_tr_v6_pool.py", "train_paddle_tr_v6.py", "prepare_paddle_v6_weights.py",
    "check_paddle_tr_v6.py",
)
REPORTS = (
    "dataset-compatibility.json", "model-shape-audit.json", "native-data-audit.json",
    "training-smoke-audit.json", "official-trainer-audit.json", "weights-transfer.json",
    "export-inference-audit.json",
)


def package(adapter, output):
    reports = {name: json.loads((adapter / name).read_text()) for name in REPORTS}
    if not all(r["passed"] and r["model"] == MODEL for r in reports.values()):
        raise ValueError("Every V6 audit must pass before packaging")
    smoke = reports["training-smoke-audit.json"]
    if (not smoke["width_preserving_training_pooling"] or not smoke["nonzero_ctc_gradients_all_widths"]
            or {r["width"]: r["ctc_steps"] for r in smoke["widths"]} != {w: w // 8 for w in WIDTHS}):
        raise ValueError("Actual training must preserve every width's CTC capacity")
    if reports["native-data-audit.json"]["samples"] != reports["dataset-compatibility.json"]["images"]:
        raise ValueError("Incomplete data audit")
    if output.exists():
        raise ValueError("Choose a fresh archive path")
    payload = {name: (adapter / name).read_bytes() for name in (*REPORTS,
        "rec_tr_v6_medium.yml", "rec_tr_v6_medium_all.yml", "rec_tr_v6_medium_fixed.yml",
        "turkish_dict.txt", "LICENSE-PaddleOCR.txt")}
    payload.update({"scripts/" + name: (ROOT / "scripts" / name).read_bytes() for name in SCRIPTS})
    payload["requirements-paddle-v6.txt"] = (ROOT / "requirements-paddle-v6.txt").read_bytes()
    payload["LICENSE-MIT.txt"] = (ROOT.parent / "LICENSE").read_bytes()
    guide = (ROOT / "docs/turkish-ppocrv6-medium.md").read_text()
    guide = guide.replace("](turkish-official-200k.md)", "](https://github.com/turgutguvercin/donut/blob/feat/synthdog-turkish/synthdog/docs/turkish-official-200k.md)")
    payload["README.md"] = guide.encode("utf-8")
    import hashlib
    manifest = {"model": MODEL,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "base_release": "turkish-ocr-200k-v4", "images_in_archive": 0,
        "files": {n: hashlib.sha256(data).hexdigest() for n, data in sorted(payload.items())}}
    payload["adapter-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo("paddle-v6-medium/" + name, date_time=(2026, 10, 15, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        for name, data in payload.items():
            if archive.read("paddle-v6-medium/" + name) != data:
                raise ValueError("ZIP round trip failed")
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": checksum(output),
            "files": len(payload), "crc_and_content_roundtrip_passed": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(package(a.adapter, a.output), indent=2))


if __name__ == "__main__":
    main()
