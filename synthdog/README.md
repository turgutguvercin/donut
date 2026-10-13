# SynthDoG 🐶: Synthetic Document Generator

SynthDoG is synthetic document generator for visual document understanding (VDU).

![image](../misc/sample_synthdog.png)

## Prerequisites

- python>=3.6
- [synthtiger](https://github.com/clovaai/synthtiger) (`pip install synthtiger`)

## Usage

```bash
# Set environment variable (for macOS)
$ export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

synthtiger -o ./outputs/SynthDoG_en -c 50 -w 4 -v template.py SynthDoG config_en.yaml

{'config': 'config_en.yaml',
 'count': 50,
 'name': 'SynthDoG',
 'output': './outputs/SynthDoG_en',
 'script': 'template.py',
 'verbose': True,
 'worker': 4}
{'aspect_ratio': [1, 2],
     .
     .
 'quality': [50, 95],
 'short_size': [720, 1024]}
Generated 1 data (task 3)
Generated 2 data (task 0)
Generated 3 data (task 1)
     .
     .
Generated 49 data (task 48)
Generated 50 data (task 49)
46.32 seconds elapsed
```

Some important arguments:

- `-o` : directory path to save data.
- `-c` : number of data to generate.
- `-w` : number of workers.
- `-s` : random seed.
- `-v` : print error messages.

To generate ECJK samples:
```bash
# english
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_en.yaml

# chinese
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_zh.yaml

# japanese
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_ja.yaml

# korean
synthtiger -o {dataset_path} -c {num_of_data} -w {num_of_workers} -v template.py SynthDoG config_ko.yaml
```

## Turkish

Run from `synthdog/`. Python 3.11 is recommended. SynthTiger 1.2.1
requires NumPy 1.x and Pillow 9.5; newer Pillow removed its font sizing API.
On Linux, building Pillow from source also requires a compiler and JPEG/zlib
headers (for example `build-essential libjpeg-dev zlib1g-dev`).

```bash
python3.11 -m venv ../.venv-synthdog
. ../.venv-synthdog/bin/activate
pip install -r requirements-tr.txt
python scripts/prepare_tr_corpus.py
synthtiger -o outputs/SynthDoG_tr -c 20 -w 2 -s 42 -v template.py SynthDoG config_tr.yaml
python scripts/validate_tr.py outputs/SynthDoG_tr --count 20
python -m pytest tests -q
```

Use a fresh output directory for each generation run: metadata is appended.
The output contains JPEG documents and `metadata.jsonl` in the existing
train/validation/test splits. Each row identifies its image and stores OCR text
in `ground_truth` → `gt_parse` → `text_sequence`; both JSON layers use UTF-8.
The validation command checks image/label correspondence, JSON decoding, font
coverage and actual SynthTiger rendering of `çğışöüÇĞİŞÖÜıI`, and sample coverage.
A small random sample may omit a rare character; use more samples in that case.

The corpus script downloads paragraphs from eight Turkish Wikipedia articles
through `tr.wikipedia.org` (HTTPS access required). It removes reference markers,
normalizes text to NFC, and writes `resources/corpus/trwiki.txt` with a
`trwiki.sources.json` manifest containing exact revision URLs and SHA-256 hashes.
Keep that manifest to reproduce the same corpus:

```bash
python scripts/prepare_tr_corpus.py --manifest resources/corpus/trwiki.sources.json
# Or select a larger collection on the first download:
python scripts/prepare_tr_corpus.py --titles Türkiye Türkçe İstanbul Ankara İzmir Bilim Müzik Edebiyat
```

Wikipedia text is CC BY-SA 4.0, attributed to Turkish Wikipedia contributors.
Revision URLs link to the articles and their contributor histories. Retain the
manifest and license information with redistributed text or text-bearing samples;
comply with attribution and share-alike requirements. The script does not commit
third-party text. Downloaded Turkish corpora and generated outputs are ignored.

`config_tr.yaml` reuses the existing Noto Sans and Noto Serif fonts in
`resources/font/en`; both support Turkish lowercase and uppercase glyphs.
They are licensed under SIL OFL 1.1, with copyright and license in `../NOTICE`.
No new font binaries are needed. English, Chinese, Japanese, and Korean
configurations remain available with their original resources.
