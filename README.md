# Turkish OCR recognition — full 100,000-image dataset v2

**100,000 images, with the new camera-angle, exposure and shadow categories integrated throughout the dataset.**
90,000 train / 5,000 validation / 5,000 test. Exact UTF-8 transcriptions, metadata and PaddleOCR configurations are included.

[Camera-angle comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/comparison-1.jpg) · [Exposure and shadows](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/comparison-2.jpg) · [Actual dataset samples](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/preview-01.jpg)

## New balanced categories

| Factor | Counts |
|---|---|
| Camera geometry | 40,000 frontal; 20,000 skew; **10,000 each** left/mild, right/mild, left/strong, right/strong |
| Exposure | 50,000 normal; **12,500 each** low light, underexposed, bright, overexposed |
| Lighting | 50,000 none; **12,500 each** gradient, soft shadow, hard shadow, partial shadow |
| Font family | **20 × 5,000**; 62 real font files |
| Content | **10 × 10,000**, including long sentences, Turkish words, prices/dates/IDs and confusables |
| Confusable group | **8 × 1,250**, with equal targeted-member counts |

Curves, underlines, case, real font styles/sizes, print damage, occlusion, blur/noise, spacing, neighboring-line fragments and tight detector crops retain their planned quotas.
Balance refers to explicit weighted marginal quotas. Some difficult combinations are restricted to keep useful text; see [the camera guide](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/CAMERA-GUIDE.md) and [actual category counts](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/category-counts.json).

## Download

All **14 ZIP parts** total **902.4 MB**.
The parts are independent archives. Extract every part into the same directory; parts with no images hold required metadata, labels, configs and notices.

**Automatic download:** save [download_dataset.py](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/download_dataset.py), then run:

```bash
python download_dataset.py --output turkish-ocr-v2
```

It downloads all parts, verifies SHA-256 and ZIP CRCs, extracts them together, and checks all 100,000 images and labels.
Verified completed parts are reused after interruptions. Output: `turkish-ocr-v2/turkish-recognition-100k-v2/`.

**One browser download:** [ZIP containing all parts](https://github.com/turgutguvercin/donut/archive/refs/tags/turkish-ocr-100k-v2.zip).
Unzip it, enter the extracted directory, and run the included helper without redownloading the parts:

```bash
python download_dataset.py --manifest download-manifest.json --parts-dir . --output .
```

Or download each part below and extract all into the same directory:

| Archive | Size | Images |
|---|---:|---:|
| [turkish-ocr-100k-part-01.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-01.zip) | 9.6 MB | 0 |
| [turkish-ocr-100k-part-02.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-02.zip) | 67.2 MB | 0 |
| [turkish-ocr-100k-part-03.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-03.zip) | 8.9 MB | 0 |
| [turkish-ocr-100k-part-04.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-04.zip) | 78.9 MB | 9,411 |
| [turkish-ocr-100k-part-05.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-05.zip) | 78.6 MB | 9,742 |
| [turkish-ocr-100k-part-06.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-06.zip) | 78.7 MB | 9,781 |
| [turkish-ocr-100k-part-07.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-07.zip) | 78.7 MB | 9,733 |
| [turkish-ocr-100k-part-08.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-08.zip) | 78.9 MB | 9,461 |
| [turkish-ocr-100k-part-09.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-09.zip) | 78.8 MB | 9,610 |
| [turkish-ocr-100k-part-10.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-10.zip) | 78.7 MB | 9,725 |
| [turkish-ocr-100k-part-11.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-11.zip) | 78.7 MB | 9,558 |
| [turkish-ocr-100k-part-12.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-12.zip) | 78.7 MB | 9,611 |
| [turkish-ocr-100k-part-13.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-13.zip) | 78.7 MB | 9,714 |
| [turkish-ocr-100k-part-14.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/turkish-ocr-100k-part-14.zip) | 29.5 MB | 3,654 |

[SHA-256 checksums](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/SHA256SUMS) · [Manifest](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/download-manifest.json)

## PaddleOCR and validation

Images are **48 pixels high** with proportional width up to **4,096**. Width buckets: 320 / 640 / 960 / 1280 / 1920 / 2560 / 4096.
Set maximum label length to **256**. Do not squeeze long lines to 320 pixels. The included full-data and per-bucket PP-OCRv4 configs use the UTF-8 dictionary.

All 100,000 images passed [dataset audits](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/audit.json) and [actual pinned Paddle preprocessing/label encoding](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/paddle-preprocessing-audit.json).
[Fresh deterministic replay](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/replay-audit.json) verifies saved images/metadata across all font files and camera/lighting categories.
70 repository tests pass. Font proofs: [1](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/font-proof-01.jpg), [2](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/font-proof-02.jpg), [3](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/font-proof-03.jpg), [4](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/font-proof-04.jpg).
14 separate controlled comparison crops are included outside the train/validation/test splits.
No model training or real-world OCR accuracy is claimed. Severe crops deliberately retain source labels despite missing details and have flags/filter lists.

The [earlier v1](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-100k-v1) remains available as a separate version.
Use one version's grouped splits for an experiment; cross-version split isolation is not guaranteed.
[Implementation pull request](https://github.com/turgutguvercin/donut/pull/1).

Generated images/labels are **CC BY-SA 4.0**, attributed to Turkish Wikipedia contributors through exact [revision references](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/trwiki.sources.json).
[License and attribution](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v2/LICENSE-DATASET.txt). Raw corpora and font binaries are excluded. Keep notices when redistributing.
