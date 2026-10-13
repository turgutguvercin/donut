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
coverage and actual SynthTiger rendering of `çğışöüÇĞİŞÖÜıI`, and reports sample coverage.
Sample labels must contain distinctive Turkish characters. Rare uppercase letters
may be absent from a small random sample; every glyph is rendered separately
during font validation regardless of its frequency in Wikipedia text.

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

### PaddleOCR recognition crops

For recognition training, generate single-line images with UTF-8 transcriptions.
The generator uses five system font families (DejaVu Sans, Serif and Sans Mono;
Liberation Sans and Serif) with real regular, bold, italic/oblique and bold italic
files. On Debian/Ubuntu, these are provided by `fonts-dejavu-core`,
`fonts-dejavu-extra` and `fonts-liberation`; install those packages if missing.
No new font binaries are committed. DejaVu uses the Bitstream Vera license
(DejaVu changes are public domain); Liberation uses SIL OFL 1.1. System copyright
notices are copied into generated output when available.

From `synthdog/`, with the Turkish environment activated and corpus prepared:

```bash
python scripts/generate_tr_recognition.py --output outputs/PaddleOCR_tr_recognition --count 20 --seed 42
# Scale up using a fresh directory:
python scripts/generate_tr_recognition.py --output outputs/PaddleOCR_tr_recognition_10k --count 10000 --seed 42
```

A 20-image batch covers all 20 font variants, ten procedural backgrounds
(plain, textured/aged, colored forms, receipt, ruled/grid paper, uneven lighting,
photocopy and dark signage), and five difficulty types (clean, defocus, motion
blur, downsampled and noisy). Text size, rotation and JPEG quality also vary.
Lines retain whole words and accents; the generator checks font coverage for
the selected text. Backgrounds are generated locally without third-party images.

Typography also includes large uppercase headings (48–64px source fonts),
medium body text (26–38px) and small text (18–22px). Half of a balanced batch has
underlines, drawn below the glyph bounds so Turkish descenders remain intact.
Heading labels use Turkish uppercase rules: `i` → `İ` and `ı` → `I`. Underline
strokes are decoration and are omitted from the transcription. Font sizes vary
relative to the crop even though every output image is normalized to height 48.
Profiles rotate across font families to avoid tying one family to one size or case.

`labels.txt` contains `images/line_00000.jpg<TAB>exact transcription`, as required
by PaddleOCR recognition datasets. Set `data_dir` to the output directory,
`label_file_list` to `labels.txt`, and `use_space_char: true`. Crops are 48 pixels
high with variable width; use suitable recognition resizing. `turkish_dict.txt`
covers the sample labels plus all Turkish letters. For a larger training corpus,
use one stable, complete character dictionary compatible with the recognizer's
output head and pretrained weights. No model training is performed by this command.

`metadata.json` records font and image hashes, style, text role and case,
underline geometry, source font size, background, degradation,
dimensions and the seed. Small batches have one contact sheet; larger batches
preview up to 100 samples in numbered 20-image contact sheets.
Every image/label pair is validated before completion. A fixed seed and identical
corpus, fonts and dependencies reproduce the same samples. Use a fresh output
directory; existing files are never overwritten. Retain Wikipedia attribution
and the accompanying revision manifest when sharing the text-bearing samples.

To use Times New Roman, Arial, Georgia, Verdana or other additional fonts, provide
a JSON catalog of licensed local font files. Family and style names are read
from each font itself; fonts with missing Turkish glyphs or incorrect checksums
are rejected. For a custom catalog, families are sampled equally, even if some
have fewer style variants.

```json
{"fonts": [{"path": "/path/to/times.ttf", "license": "Microsoft Core Fonts EULA",
            "license_file": "/path/to/EULA.txt", "sha256": "actual_font_sha256"}]}
```

```bash
python scripts/generate_tr_recognition.py --font-manifest /path/to/fonts.json --output outputs/PaddleOCR_tr_100 --count 100 --seed 42
```

Paths can also be relative to the catalog. Use the SHA-256 of each actual font;
omit that field only when verification against a pre-existing catalog is not
needed. License notices are required and copied into the output; font binaries
are never copied. Microsoft Core Fonts have a proprietary EULA and must be
obtained through an authorized source. They are not supplied by this repository.
