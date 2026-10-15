"""Pinned PP-OCRv6 medium references and vocabulary-safe weight transfer."""
import hashlib
from pathlib import Path
import subprocess
import urllib.request

import numpy as np

COMMIT = "dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf"
MODEL = "PP-OCRv6_medium_rec"
WIDTHS = (320, 640, 960, 1280, 1920, 2560, 4096)
BATCH_SIZES = {320: 32, 640: 16, 960: 8, 1280: 8, 1920: 4, 2560: 2, 4096: 1}
MAX_LENGTH = 256
REFERENCES = {
    "PP-OCRv6_medium_rec.yml": ("configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml", "5ebee6044fc713b645c2204ed51de67d6834830b0eafeb27b3cd76717c942af0"),
    "ppocrv6_dict.txt": ("ppocr/utils/dict/ppocrv6_dict.txt", "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d"),
    "label_ops.py": ("ppocr/data/imaug/label_ops.py", "89c75bfb515c0d8ea3e8ebb5728335fcb8032e8d2e3e3060eb059527e7df61c7"),
    "rec_img_aug.py": ("ppocr/data/imaug/rec_img_aug.py", "77607eb2a5e6e5d01829565a97161845ecd85a35a4d827435198f3ed4a32f9ad"),
    "LICENSE": ("LICENSE", "3840c5c0c61c294264d2dd77b8777be6ddd90121ef4e0e64abcd22edea581d6e"),
}
WEIGHTS_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_rec_pretrained.pdparams"
WEIGHTS_SHA256 = "c4a110423c23739f157ff870e27f41649d8c89e4c8df5230c2cd2e9fd90a7e92"
VOCAB_AXES = {
    "head.ctc_head.fc.weight": (1, "ctc"),
    "head.ctc_head.fc.bias": (0, "ctc"),
    "head.gtc_head.embedding.embedding.weight": (0, "nrtr"),
    "head.gtc_head.tgt_word_prj.weight": (1, "nrtr"),
}


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_download(url, path, sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if checksum(path) != sha256:
            raise ValueError(f"Checksum mismatch: {path}")
        return path
    temporary = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            target.write(block)
    if checksum(temporary) != sha256:
        raise ValueError(f"Downloaded artifact checksum mismatch: {path}")
    temporary.replace(path)
    return path


def references(cache):
    return {name: verified_download(f"https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{COMMIT}/{remote}",
                                   Path(cache) / name, sha).read_text(encoding="utf-8")
            for name, (remote, sha) in REFERENCES.items()}


def check_checkout(path):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if revision != COMMIT:
        raise ValueError(f"Use PaddleOCR commit {COMMIT}; found {revision}")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "ppocr", "tools", "configs/rec/PP-OCRv6"], cwd=path):
        raise ValueError("Pinned PaddleOCR runtime files have local modifications")


def characters(path):
    with Path(path).open("rb") as source:
        chars = [line.decode("utf-8").strip("\n").strip("\r") for line in source]
    if not chars or any(len(c) != 1 or c in " \t\r\n" for c in chars) or len(set(chars)) != len(chars):
        raise ValueError("Expected unique single-character dictionary entries; space is added by the encoder")
    return chars + [" "]


def tokens(chars, head):
    if head == "ctc":
        return ["<blank>"] + chars
    if head != "nrtr":
        raise ValueError("Unknown head")
    # The pinned Transformer adds one extra output class after the encoder's
    # vocabulary. Preserve this trailing row too; it is NOT a fifth prefix token.
    return ["<blank>", "<unk>", "<s>", "</s>"] + chars + ["<nrtr_unused_output>"]


def remap_axis(source, initialized, old_tokens, new_tokens, axis):
    """Preserve initialization for new symbols; remap shared symbols by identity."""
    if len(set(old_tokens)) != len(old_tokens) or len(set(new_tokens)) != len(new_tokens):
        raise ValueError("Ambiguous vocabulary indices")
    if source.shape[axis] != len(old_tokens) or initialized.shape[axis] != len(new_tokens):
        raise ValueError("Checkpoint class count does not match its dictionary")
    if source.ndim != initialized.ndim or any(a != b for i, (a, b) in enumerate(zip(source.shape, initialized.shape)) if i != axis):
        raise ValueError("Non-vocabulary parameter dimensions differ")
    out = initialized.copy()
    old = {c: i for i, c in enumerate(old_tokens)}
    for i, c in enumerate(new_tokens):
        if c in old:
            src, dst = [slice(None)] * source.ndim, [slice(None)] * source.ndim
            src[axis], dst[axis] = old[c], i
            out[tuple(dst)] = source[tuple(src)]
    return out


def choose_bucket(width, label, measured_steps):
    required = len(label) + sum(a == b for a, b in zip(label, label[1:]))
    for w in sorted(measured_steps):
        if w >= width and measured_steps[w] >= required:
            return w
    raise ValueError("No width bucket can preserve both image width and CTC capacity")


def bucket_batches(groups, batch_sizes, seed, epoch=0, shuffle=True):
    """One appearance per index, including partial batches; no bucket is dropped."""
    rng = np.random.default_rng(seed + epoch)
    batches = []
    for width, indices in sorted(groups.items()):
        order = list(indices)
        if batch_sizes[width] < 1:
            raise ValueError("Batch size must be positive")
        if shuffle:
            rng.shuffle(order)
        batches.extend(order[i:i + batch_sizes[width]] for i in range(0, len(order), batch_sizes[width]))
    if shuffle:
        rng.shuffle(batches)
    return batches
