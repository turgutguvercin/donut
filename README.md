# Turkish OCR 200k — PP-OCRv6_medium_rec adapter

[Download adapter ZIP](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/turkish-v6-medium-adapter-v1.zip) · [Training guide](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/TRAINING-GUIDE.md) · [Checksums](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/SHA256SUMS) · [Validation](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/validation.json)

**41,350 bytes; 23 files.** Extract into your existing `turkish-recognition-200k-v4/` directory. It adds `paddle-v6-medium/`. The original 200,000 images, labels, category counts and splits are preserved; no image redownload is needed. The [200k base release](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-200k-v4) is required.

```bash
curl -fLO https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/turkish-v6-medium-adapter-v1.zip
curl -fLO https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-v6-medium-adapter-v1/SHA256SUMS
sha256sum -c SHA256SUMS
python -m zipfile -e turkish-v6-medium-adapter-v1.zip /path/to/turkish-recognition-200k-v4
```

Use the included **train_paddle_tr_v6.py launcher**, not a direct tools/train.py call. The pinned upstream V6 backbone pools every training width to 40 CTC steps. The launcher uses its existing inference pooling during training too, preserving 40–512 steps for widths 320–4096. Learnable tensors and inference behavior stay compatible with official pretrained weights.

- Height 48; proportional width buckets 320/640/960/1280/1920/2560/4096, all included in each epoch, including partial batches.
- Global, NRTR and label encoder limit 256; 254 actual-character capacity, maximum dataset label 252.
- Turkish dictionary: 220 entries plus space. Character-aware transfer copies 437 feature tensors and remaps CTC/NRTR vocabulary tensors. All Turkish letters transfer; four additional Unicode symbols initialize deterministically.
- Recommended train/validation/test counts: **151,402 / 8,391 / 8,410**. Full-list option: **180,000 / 10,000 / 10,000**. Ambiguous/severely degraded crops keep their challenge flags.
- Pinned PaddleOCR commit dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf and tested Python 3.11 training dependencies; separate from SynthDoG rendering.

**Cloud validation:** 98 repository tests; native decoding, CTC/NRTR encoding and resizing for all 200,000 images; complete two-epoch sampler coverage; two-worker train/eval collation; seven real pretrained-model backward/Adam checks across all widths; official trainer evaluation/checkpoint integration; export and inference at widths 320 and 4096. These are CPU integration checks, not full training, a GPU-memory guarantee, or a real-document accuracy benchmark.

No images, raw corpora, fonts or model-weight binaries are included in this adapter. Preserve the base dataset's CC BY-SA attribution. Helper code is MIT; Paddle-derived configurations and pooling adaptation are Apache-2.0.

[Implementation PR](https://github.com/turgutguvercin/donut/pull/2)
