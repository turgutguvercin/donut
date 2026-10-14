# Turkish OCR recognition — 100,000 images

**90,000 train / 5,000 validation / 5,000 test**, with UTF-8 labels and all category metadata.
20 primary font families × 5,000 images; 62 real font files. Ten content classes × 10,000.
Eight targeted confusable groups × 1,250. [Full category quotas](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/DATASET-GUIDE.md).

Images are **48 pixels high**, preserve proportions, and have variable width up to **4,096**.
PaddleOCR width buckets: 320 / 640 / 960 / 1280 / 1920 / 2560 / 4096. Max label setting: 256.
The download includes all-image and per-bucket PP-OCRv4 configs, dictionaries, labels, hashes, licenses, reproduction code and previews.

## Download

All ZIP parts total **985.6 MB**. The files below are independent archives. Extract every part into the same folder.
Parts with zero images contain the required shared labels, metadata, configs and notices.

**Automatic download:** save [download_dataset.py](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/download_dataset.py), then run:

```bash
python download_dataset.py --output turkish-ocr-download
```

It downloads all parts, verifies SHA-256 and ZIP integrity, extracts them together, and checks 100,000 images/labels.
Completed verified parts are reused after interruptions. The resulting dataset is under `turkish-ocr-download/turkish-recognition-100k-v1`.

**One browser download:** [ZIP containing all parts](https://github.com/turgutguvercin/donut/archive/refs/tags/turkish-ocr-100k-v1.zip).
Unzip it, enter the extracted folder, and run the included helper to verify/extract the parts without downloading them again:

```bash
python download_dataset.py --manifest download-manifest.json --parts-dir . --output .
```

Or download/extract each part manually:

| Archive | Size | Images |
|---|---:|---:|
| [turkish-ocr-100k-part-01.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-01.zip) | 8.5 MB | 0 |
| [turkish-ocr-100k-part-02.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-02.zip) | 53.9 MB | 0 |
| [turkish-ocr-100k-part-03.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-03.zip) | 8.5 MB | 0 |
| [turkish-ocr-100k-part-04.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-04.zip) | 79.9 MB | 8,613 |
| [turkish-ocr-100k-part-05.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-05.zip) | 79.8 MB | 8,638 |
| [turkish-ocr-100k-part-06.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-06.zip) | 79.6 MB | 8,757 |
| [turkish-ocr-100k-part-07.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-07.zip) | 79.5 MB | 8,988 |
| [turkish-ocr-100k-part-08.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-08.zip) | 79.7 MB | 8,729 |
| [turkish-ocr-100k-part-09.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-09.zip) | 79.9 MB | 8,635 |
| [turkish-ocr-100k-part-10.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-10.zip) | 79.7 MB | 8,786 |
| [turkish-ocr-100k-part-11.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-11.zip) | 79.6 MB | 8,843 |
| [turkish-ocr-100k-part-12.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-12.zip) | 79.8 MB | 8,572 |
| [turkish-ocr-100k-part-13.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-13.zip) | 79.8 MB | 8,747 |
| [turkish-ocr-100k-part-14.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-14.zip) | 79.8 MB | 8,628 |
| [turkish-ocr-100k-part-15.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/turkish-ocr-100k-part-15.zip) | 37.4 MB | 4,064 |

[SHA-256 checksums](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/SHA256SUMS) · [Download manifest](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/download-manifest.json)

## Previews and checks

[New categories, page 1](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/preview-01.jpg) · [Crop errors and typography](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/preview-02.jpg) · [Confusables and long lines](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/preview-03.jpg)

Actual licensed font renderings: [1](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/font-proof-01.jpg) · [2](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/font-proof-02.jpg) · [3](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/font-proof-03.jpg) · [4](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/font-proof-04.jpg).

All 100,000 images and labels were audited. [Dataset audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/audit.json) · [Actual PaddleOCR preprocessing/encoding audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/paddle-preprocessing-audit.json).
60 repository tests passed in the cloud. [Implementation pull request](https://github.com/turgutguvercin/donut/pull/1).

Balance means the explicit weighted quotas in the guide; styles are equal within each family's available files.
5,000 examples are clean controls. 7,000 severe crops deliberately erase marks/letter fragments and retain source labels;
they are flagged potentially ambiguous, with optional filtered split lists. Synthetic test data is not a real-world OCR benchmark.
No OCR model was trained for this dataset delivery.

Generated images/labels are **CC BY-SA 4.0**, attributed to Turkish Wikipedia contributors with exact [revision references](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/trwiki.sources.json).
[License and attribution](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/LICENSE-DATASET.txt). Font binaries and raw corpora are excluded. Preserve the notices when redistributing.
