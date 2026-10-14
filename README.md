# Turkish OCR — 100,000 images with balanced Wikipedia vocabulary (v3)

**400 Turkish Wikipedia articles, 20 topics, 52,387 distinct readable word forms.**
90,000 training / 5,000 validation / 5,000 test images with UTF-8 labels, metadata, source attribution and PaddleOCR configs.
V2 had eight articles and 13,389 word forms; v3 expands the text while preserving all visual-category quotas.

[Topic examples 1](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/topics-preview-1.jpg) · [Topic examples 2](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/topics-preview-2.jpg) ·
[Camera comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/comparison-1.jpg) · [Shadows/exposure](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/comparison-2.jpg) · [Actual difficult samples](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/preview-01.jpg)

## Balance

| Factor | Allocation |
|---|---|
| Natural text | 60,000 images: **3,000 per topic, 150 per article** |
| Source split | **360 train / 20 validation / 20 test articles**; each topic has 18 / 1 / 1 |
| Natural content | 25 images per article in each of six word/line/sentence classes |
| Structured content | Dates, prices, generated IDs and confusable exercises: **10,000 each** |
| Fonts | **20 families × 5,000**; 62 real font files; styles balanced within available family faces |
| Confusables | **Eight groups × 1,250**, with equal targeted-member counts |
| Camera | 40,000 frontal; 20,000 skew; 10,000 each left/right × mild/strong |
| Exposure | 50,000 normal; 12,500 each low light, underexposed, bright, overexposed |
| Lighting | 50,000 none; 12,500 each gradient, soft shadow, hard shadow, partial shadow |

All existing font styles/sizes, underlines, case, curves, backgrounds, blur/noise, spacing, printing,
occlusion, neighboring-line fragments and tight/severe detector crops retain their quotas.
Balance means weighted marginal categories, plus exact article/topic allocations; not every combination is equally frequent.
Common words can repeat; individual word frequencies are not forced equal.

Target and neighboring context articles are disjoint between splits. Whole canonical source/target strings are also disjoint;
individual words and substrings can overlap. See [source balance](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/source-balance.json),
[category counts](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/category-counts.json) and [vocabulary report](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/vocabulary-report.json).
Vocabulary counts normalize Turkish case/apostrophes, exclude numeric/ID values and artificial confusable exercises,
and include inflections, names and abbreviations as separate surface forms. Training contains
**50,261** distinct readable forms.
Every Turkish alphabet letter appears in training. The report records two foreign letters (ć, ň) present only in held-out labels.

## Download

All **14 independent ZIP parts** total **912.8 MB**.
Save [download_dataset.py](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/download_dataset.py), then run:

```bash
python download_dataset.py --output turkish-ocr-v3
```

The helper verifies SHA-256 and ZIP CRCs, extracts all parts together and checks the 100,000 images and labels.
Completed verified parts are reused after interruptions. Output: `turkish-ocr-v3/turkish-recognition-100k-v3/`.

**Browser download:** [ZIP containing all parts](https://github.com/turgutguvercin/donut/archive/refs/tags/turkish-ocr-100k-v3.zip).
Unzip it, enter the extracted directory and run:

```bash
python download_dataset.py --manifest download-manifest.json --parts-dir . --output .
```

Or download every part below and extract them into the same directory. Parts with zero images contain required metadata/configs/notices.

| Archive | Size | Images |
|---|---:|---:|
| [turkish-ocr-100k-part-01.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-01.zip) | 9.8 MB | 0 |
| [turkish-ocr-100k-part-02.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-02.zip) | 75.4 MB | 0 |
| [turkish-ocr-100k-part-03.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-03.zip) | 10.2 MB | 0 |
| [turkish-ocr-100k-part-04.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-04.zip) | 78.9 MB | 9,413 |
| [turkish-ocr-100k-part-05.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-05.zip) | 78.5 MB | 9,757 |
| [turkish-ocr-100k-part-06.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-06.zip) | 78.6 MB | 9,806 |
| [turkish-ocr-100k-part-07.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-07.zip) | 78.7 MB | 9,696 |
| [turkish-ocr-100k-part-08.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-08.zip) | 78.8 MB | 9,429 |
| [turkish-ocr-100k-part-09.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-09.zip) | 78.8 MB | 9,571 |
| [turkish-ocr-100k-part-10.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-10.zip) | 78.6 MB | 9,797 |
| [turkish-ocr-100k-part-11.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-11.zip) | 78.7 MB | 9,596 |
| [turkish-ocr-100k-part-12.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-12.zip) | 78.8 MB | 9,567 |
| [turkish-ocr-100k-part-13.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-13.zip) | 78.7 MB | 9,644 |
| [turkish-ocr-100k-part-14.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/turkish-ocr-100k-part-14.zip) | 30.1 MB | 3,724 |

[SHA-256 checksums](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/SHA256SUMS) · [Download manifest](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/download-manifest.json)

## PaddleOCR and validation

Images are **48 pixels high**, with proportional widths up to **4,096**. Seven width buckets preserve aspect ratio
and CTC capacity. Maximum label length is 256. Long sentences are not squeezed to width 320.
Included PP-OCRv4 configs use the UTF-8 dictionary; choose suitable pretrained weights before fine-tuning.

All 100,000 images passed [dataset audits](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/audit.json) and [pinned actual Paddle preprocessing/encoding](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/paddle-preprocessing-audit.json).
Duplicate image hashes: 0. [Fresh replay](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/replay-audit.json) matched exact image hashes and metadata for
940 samples across every source article and font file. All 60,000 natural target sources were verified against their actual article text.
81 repository tests pass. Raw corpus reconstruction is hash checked; exact revisions and source manifests are included.
No model training or real-world OCR accuracy is claimed. Real detector crops remain necessary for evaluation.
Severe crops keep source labels despite missing marks; ambiguity flags and optional filtered split lists are included.
The 14 controlled comparison crops are outside the train/validation/test samples.

[Reproduction guide](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/VOCABULARY-GUIDE.md) · [Camera guide](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/CAMERA-GUIDE.md) ·
[Implementation PR](https://github.com/turgutguvercin/donut/pull/1).
Older releases remain available separately. Use one version's splits; mixing versions does not guarantee isolation.

Generated images and labels: **CC BY-SA 4.0**, attributed to Turkish Wikipedia contributors through exact
[revision references](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/trwiki.sources.json). Retain [license/attribution](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/LICENSE-DATASET.txt), the manifest and font notices.
No font binaries or raw Wikipedia corpus are distributed. Reproduction code is MIT; Paddle-derived configs are Apache-2.0.
