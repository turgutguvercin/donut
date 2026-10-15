# Turkish OCR — 200,000 images with official-document fields (v4)

**Download:** [helper](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/download_dataset.py) · [manifest](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/download-manifest.json) ·
[small 100-image sample ZIP](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/sample-100.zip) · [all parts in one browser ZIP](https://github.com/turgutguvercin/donut/archive/refs/tags/turkish-ocr-200k-v4.zip)

[Official examples 1](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/official-preview-01.jpg) · [Official examples 2](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/official-preview-02.jpg) ·
[Official examples 3](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/official-preview-03.jpg) · [Clean font styles](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/official-preview-04-clean.jpg) · [Other visual categories](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/preview-01.jpg)

All **59 independent ZIP parts** total **2918.0 MB**.
Save download_dataset.py and run:

```bash
python download_dataset.py --output turkish-ocr-v4
```

The helper verifies SHA-256 and CRC, extracts every part, reassembles large metadata and checks 200,000 images/labels.
Output: turkish-ocr-v4/turkish-recognition-200k-v4/.
For the browser ZIP, extract its outer folder and run the included helper:

```bash
python download_dataset.py --manifest download-manifest.json --parts-dir . --output .
```

**Metadata-only parts are required.** Manual ZIP extraction alone does not reassemble metadata.jsonl.
The small sample ZIP is immediately extractable and contains 100 training examples plus UTF-8 labels.

[Turkish setup guide](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/OFFICIAL-GUIDE.md) · [Image audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/audit.json) ·
[Real v5 CTC shape probe](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/model-shape-audit.json) · [Quality subsets](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/quality-report.json) ·
[Implementation PR](https://github.com/turgutguvercin/donut/pull/2)


Preserves all 100,000 v3 images/transcriptions/splits and adds **100,000 original official-style synthetic crops**.
This is line recognition data, not a document detection dataset. No real scanned/photographed document crops have been added.

| Split | Complete | Recommended | Challenge |
|---|---:|---:|---:|
| train | 180,000 | 151,402 | 28,598 |
| val | 10,000 | 8,391 | 1,609 |
| test | 10,000 | 8,410 | 1,590 |

Use **paddle-v5/rec_tr_recommended.yml** for the default filtered training/validation lists.
The complete lists retain deliberately ambiguous v3 crops. Quality flags are measurable proxies, not a human-readability guarantee.
The challenge lists contain excluded examples; they are not ordinary unambiguous training targets.

## Added official text

- official_verification: 10000
- official_footer_contact: 10000
- official_confusables_codes: 10000
- official_body_short: 10000
- official_heading: 10000
- official_date_reference: 10000
- official_subject: 10000
- official_amount: 10000
- official_body_long: 10000
- official_confusables_words: 10000

Times New Roman: 20,000 new images; Arial: 20,000; 18 other families share 60,000.
Each primary family has 5,000 examples per style: regular, bold, italic and bold italic.
Within-line mixed typography, underlines and varying glyph heights are included.
13 targeted confusable groups cover l/1/I/İ/ı/i, O/0, S/5, B/8, rn/m, cl/d, Turkish diacritics, 2021/202I and İl/İ1.
Dates, amounts, phone numbers, email/URL/KEP and verification fields use original generated values.
Reserved example.org contact domains are synthetic. Source template families recur across splits.

30,000 new crops use physical downsample→blur→noise→JPEG→upsample capture, with recorded glyph-height estimates.
V3's 400 Wikipedia articles across 20 topics and all previous visual categories remain intact.
source-balance.json describes the preserved Wikipedia portion; official-balance.json describes the added synthetic portion.
Balance is explicit marginal quotas, not all combinations equally frequent. Filtered-list distributions are reported separately.
Readable word forms: **52,442** overall.
No extra Wikipedia corpus was downloaded for this expansion.

## Measured PP-OCRv5 capacity

Height 48, width ≤4096; buckets 320/640/960/1280/1920/2560/4096.
Actual full PP-OCRv5_server_rec CPU forwards return 40/80/120/160/240/320/512 CTC steps: **width stride 8**.
The finalizer moved **11,092** samples into wider buckets without changing image bytes, labels or splits.
The Global/NRTR/encoder limit of 256 allows 254 real characters, reserving start/end tokens.
The actual longest label has 252 characters. The default 25 only allows 23.
Adjacent repeated characters are included in CTC capacity checks.

All 200,000 images passed decoding/hashes, UTF-8/NFC, actual font glyph coverage, quotas, split and crop checks.
All 200,000 passed pinned actual Paddle resize/CTC/NRTR operators, with no truncation or horizontal squeezing.
89 repository tests passed; **569** new samples replayed with exact metadata/image hashes.
All 100,000 base images/labels/splits were preserved. Duplicate image hashes: 0.
No model training, pretrained-model accuracy evaluation, or real-document accuracy claim.
The CPU shape probe uses random weights solely to measure architecture output dimensions.

## Sources, download and reproduction

See OFFICIAL-GUIDE.md for commands and limitations; official_document_references.json records the inspected public regulation.
Government PDF pages are not included as training images. Generated text follows observed field/typography conventions.
Keep LICENSE-DATASET.txt, trwiki.sources.json, font notices and provenance. No raw corpus or font binaries are distributed.
Generated images/labels: CC BY-SA 4.0; reproduction code: MIT; Paddle-derived configurations: Apache-2.0.

Use the published download_dataset.py helper for all ZIP parts and checksummed metadata reassembly.
Large metadata is stored in verified shards; manual ZIP extraction alone does not assemble metadata.jsonl.
V3 and v4 share 100,000 images. Do not combine their splits as independent datasets.

## Every download part

| Archive | Size | Images |
|---|---:|---:|
| [turkish-ocr-200k-part-01.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-01.zip) | 16.1 MB | 0 |
| [turkish-ocr-200k-part-02.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-02.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-03.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-03.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-04.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-04.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-05.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-05.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-06.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-06.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-07.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-07.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-08.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-08.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-09.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-09.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-10.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-10.zip) | 7.4 MB | 0 |
| [turkish-ocr-200k-part-11.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-11.zip) | 7.3 MB | 0 |
| [turkish-ocr-200k-part-12.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-12.zip) | 6.2 MB | 0 |
| [turkish-ocr-200k-part-13.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-13.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-14.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-14.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-15.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-15.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-16.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-16.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-17.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-17.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-18.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-18.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-19.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-19.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-20.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-20.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-21.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-21.zip) | 5.8 MB | 0 |
| [turkish-ocr-200k-part-22.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-22.zip) | 15.0 MB | 0 |
| [turkish-ocr-200k-part-23.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-23.zip) | 23.3 MB | 0 |
| [turkish-ocr-200k-part-24.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-24.zip) | 71.2 MB | 8,504 |
| [turkish-ocr-200k-part-25.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-25.zip) | 70.7 MB | 8,811 |
| [turkish-ocr-200k-part-26.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-26.zip) | 70.7 MB | 8,805 |
| [turkish-ocr-200k-part-27.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-27.zip) | 70.9 MB | 8,739 |
| [turkish-ocr-200k-part-28.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-28.zip) | 71.0 MB | 8,667 |
| [turkish-ocr-200k-part-29.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-29.zip) | 71.1 MB | 8,567 |
| [turkish-ocr-200k-part-30.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-30.zip) | 71.0 MB | 8,660 |
| [turkish-ocr-200k-part-31.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-31.zip) | 71.0 MB | 8,686 |
| [turkish-ocr-200k-part-32.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-32.zip) | 71.0 MB | 8,648 |
| [turkish-ocr-200k-part-33.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-33.zip) | 71.0 MB | 8,654 |
| [turkish-ocr-200k-part-34.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-34.zip) | 70.9 MB | 8,694 |
| [turkish-ocr-200k-part-35.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-35.zip) | 75.3 MB | 6,517 |
| [turkish-ocr-200k-part-36.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-36.zip) | 80.1 MB | 4,204 |
| [turkish-ocr-200k-part-37.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-37.zip) | 80.1 MB | 4,229 |
| [turkish-ocr-200k-part-38.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-38.zip) | 80.2 MB | 4,145 |
| [turkish-ocr-200k-part-39.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-39.zip) | 80.4 MB | 4,111 |
| [turkish-ocr-200k-part-40.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-40.zip) | 80.2 MB | 4,154 |
| [turkish-ocr-200k-part-41.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-41.zip) | 80.3 MB | 4,104 |
| [turkish-ocr-200k-part-42.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-42.zip) | 80.3 MB | 4,132 |
| [turkish-ocr-200k-part-43.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-43.zip) | 80.2 MB | 4,170 |
| [turkish-ocr-200k-part-44.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-44.zip) | 80.3 MB | 4,149 |
| [turkish-ocr-200k-part-45.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-45.zip) | 80.2 MB | 4,175 |
| [turkish-ocr-200k-part-46.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-46.zip) | 80.2 MB | 4,178 |
| [turkish-ocr-200k-part-47.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-47.zip) | 80.2 MB | 4,193 |
| [turkish-ocr-200k-part-48.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-48.zip) | 80.2 MB | 4,152 |
| [turkish-ocr-200k-part-49.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-49.zip) | 80.1 MB | 4,226 |
| [turkish-ocr-200k-part-50.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-50.zip) | 80.0 MB | 4,273 |
| [turkish-ocr-200k-part-51.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-51.zip) | 79.9 MB | 4,308 |
| [turkish-ocr-200k-part-52.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-52.zip) | 80.2 MB | 4,194 |
| [turkish-ocr-200k-part-53.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-53.zip) | 80.1 MB | 4,236 |
| [turkish-ocr-200k-part-54.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-54.zip) | 80.0 MB | 4,276 |
| [turkish-ocr-200k-part-55.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-55.zip) | 80.2 MB | 4,163 |
| [turkish-ocr-200k-part-56.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-56.zip) | 80.0 MB | 4,262 |
| [turkish-ocr-200k-part-57.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-57.zip) | 80.2 MB | 4,235 |
| [turkish-ocr-200k-part-58.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-58.zip) | 80.3 MB | 4,118 |
| [turkish-ocr-200k-part-59.zip](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-200k-v4/turkish-ocr-200k-part-59.zip) | 31.8 MB | 1,661 |
