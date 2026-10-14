# Turkish recognition dataset: 100,000 images

This recipe generates recognition crops and exact UTF-8 target transcriptions.
It does not generate text-detection polygons or train an OCR model.

**Published dataset:** [download 100,000 images and labels](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-100k-v1)
(986 MB across 15 independent ZIP parts, with an automatic downloader).
[Preview](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/preview-03.jpg)
and [full audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v1/audit.json).
The published run uses source commit `dd133a34a5d90e86d849a34d263f195fd299e24e`,
has no duplicate image hashes, and passed pinned PaddleOCR preprocessing and
label encoding for every sample. A 232-sample replay covered all 62 font files.

For a new run with explicit camera sides, stronger perspective, low light and
soft/hard/partial shadows, see the [camera and lighting extension](turkish-camera-lighting.md).
Those additional categories are not included in the published v1 dataset above.

Activate the Turkish environment described in the main README, prepare
`resources/corpus/trwiki.txt` with `scripts/prepare_tr_corpus.py`, and supply a
licensed font catalog in the format documented in the README. From `synthdog/`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/generate_tr_dataset.py \
  --font-manifest /path/to/recognition-fonts.json \
  --output outputs/turkish-recognition-100k \
  --count 100000 --seed 20261014 --workers 4

python scripts/generate_tr_dataset.py \
  --font-manifest /path/to/recognition-fonts.json \
  --output outputs/turkish-recognition-100k --validate-only
```

The seed, corpus hash, exact Wikipedia revisions, font hashes/licenses, generator
hashes, per-image seeds, rendering parameters and image hashes are included.
Worker count does not affect image content. Use the pinned dependencies in
`requirements-tr.txt` for image replay. Output directories must be new; a failed
run leaves its completed images and metadata available for diagnosis.

## What balanced means

`config_tr_recognition_100k.json` defines explicit **weighted marginal quotas**.
Each non-control variant in a group has equal coverage, except the deliberately
documented blur and typography mixes below. This is not every possible Cartesian
combination and is not a promise of equal raw character frequencies in natural
Turkish. `planned-counts.json`, `category-counts.json`, `pair-counts.json` and
`categories.csv` expose the actual distribution.

| Category | Quota at 100,000 |
|---|---|
| Primary font family | 5,000 each with the supplied 20-family catalog |
| Real font style | Equal within the styles actually available for each family |
| Content | 10,000 each: ASCII word, Turkish-marked word, short span, medium span, long sentence, extra-long sentence, confusables, date, price, synthetic document ID |
| Confusable group | 1,250 each; every member appears exactly once in the targeted span |
| Text role/size | Approximately one third small, body and heading; split rounding differs by at most two |
| Case | 45,000 original, 45,000 Turkish uppercase, 10,000 deliberately mixed confusable examples |
| Decoration | 50,000 plain; 50,000 underlined |
| Baseline | 25,000 each: straight, arch, bowl, wave |
| Background | 10,000 each of ten procedural paper/form/surface backgrounds |
| Geometry | 50,000 without extra geometry; 25,000 skew; 25,000 perspective |
| Extra lighting | 50,000 none; 25,000 shadow; 25,000 gradient; some background textures already include uneven illumination |
| Contrast | 60,000 normal; 30,000 low; 10,000 faint |
| Degradation | 30,000 clean; 20,000 defocus; 10,000 each motion, low resolution, noise, strong defocus; 5,000 each strong motion and combined blur |
| Detector crop | 50,000 clean; 5,000 each: neighbor above, below, both, vertical shift, tight top, bottom, both vertical edges, left, right, both horizontal edges |
| Clipping severity | Of the 35,000 clipping examples, 28,000 mild and 7,000 severe |
| Spacing | 50,000 normal; 12,500 each tight, wide, irregular word gaps, touching/overlapping tracking |
| Printing | 50,000 ordinary; 12,500 each broken strokes, missing ink, bleed, dot matrix |
| Occlusion | 50,000 none; 12,500 each synthetic stamp, signature stroke, fold, strikethrough |
| Mixed typography | 50,000 uniform; 20,000 mixed fonts; 10,000 each mixed styles/sizes; 5,000 each superscript/subscript |

The eight confusable groups are `iıİI`, `cçCÇ`, `sşSŞ`, `gğGĞ`, `uüUÜ`,
`oöOÖ`, `O0`, and `l1`. The targeted spans have equal member counts; unrelated
characters in surrounding text are not part of that balancing claim. Natural
content supplies letters in words as well as these explicit exercises.

Mixed typography records every actual font run and its hash. A family without
an italic or bold face is never given an artificial style; a mixed-style run can
use another licensed family. Primary-family quotas do not count secondary runs.
Superscript/subscript examples append a visible footnote/index digit and include
it in the label. Long sentences retain their full text; sentence boundaries use
the existing punctuation/abbreviation heuristic.

There are 5,000 clean reference controls. Severe crops use normal contrast,
ordinary spacing and clean rendering without additional print damage or
occlusion. Faint text avoids blur/print damage/occlusion, and low contrast avoids
strong blur. These constraints preserve the quotas but intentionally restrict
some combinations; the pair counts make that visible. Irregular word gaps use
multiword text, so this category cannot be wasted on a single word.

## Crops, labels and splits

`labels.txt` contains `relative/path.jpg<TAB>exact text`. Neighbors, stamps,
signatures and underlines are not part of the target transcription. Neighbor
transcriptions appear in `metadata.jsonl`. All corpus-derived target and context
text is selected within its assigned split. Canonical target labels and source
text groups must be disjoint between **90,000 train / 5,000 val / 5,000 test**.
Words can naturally occur within longer sentences in other splits; the grouping
prevents identical full target/source strings, not all shared substrings.

Mild cropping preserves at least 97% of target ink and 50% of each connected
component. Severe cropping preserves at least 70% of total ink and can erase
dots, umlauts, cedillas or edge letters. These measurements exclude underline
strokes and are taken before final resizing, occlusion and blur. Original labels
remain on severe examples, so some are visually ambiguous. Their metadata sets
`potentially_ambiguous: true`; `*-without-severe-clips.txt` provides optional
filtered splits. Do not interpret synthetic validation scores as real OCR accuracy.

## PaddleOCR dimensions

Images are RGB JPEGs, **48 pixels high with variable width, at most 4,096**.
Long lines are not squeezed to 320 pixels. If a proportional 48-pixel-high line
exceeds 4,096 pixels, it is uniformly reduced to that width and padded vertically
back to 48. The scale and glyph bounds are recorded. The dictionary includes
all observed characters plus Turkish letters; use `use_space_char: true`.

Width buckets are `320, 640, 960, 1280, 1920, 2560, 4096`, with one label list
per split and bucket. A bucket must fit both the image width and enough CTC
timesteps for the label, including adjacent repeated characters at width stride
four. Some condensed long lines therefore use a larger padding bucket.
Right-pad within a bucket. For PaddleOCR configurations
that always resize to a fixed width, use 4,096 for the full set or use a matching
bucket configuration and bucket list. A 320-wide resize for all samples destroys
long-line detail. Keep image height 48 and allow labels up to 256 characters
in both the label encoder and recognition head. The standard English PP-OCRv4
example defaults to only 25 characters and must be adapted.

Prepare version-pinned PP-OCRv4 configs and validate using its real resize and
CTC/NRTR label-encoding operators (the script downloads checksum-verified
upstream source; it does not install or train Paddle's neural-network runtime):

```bash
python scripts/prepare_paddle_tr.py \
  --dataset outputs/turkish-recognition-100k \
  --cache /tmp/paddleocr-reference \
  --data-dir ./train_data/turkish-recognition-100k-v1
```

The reference is PaddleOCR `release/2.8` commit
`2172c14bf65ceacf37a839ebbe61117a6339a865`. `paddle/rec_tr_all.yml` uses every
training/validation sample, right-padded to 4,096, with a small batch size.
`paddle/rec_tr_w*.yml` uses **only that bucket's samples**. Efficient joint
training with mixed bucket widths needs a suitable batch sampler; these seven
configs are not automatic sequential training stages. Additional RecAug and
RecConAug are disabled to preserve the planned effects and label lengths.

After installing a compatible PaddleOCR environment, place the dataset under
`PaddleOCR/train_data/turkish-recognition-100k-v1`. From that PaddleOCR checkout:

```bash
python tools/train.py -c train_data/turkish-recognition-100k-v1/paddle/rec_tr_all.yml
```

The generated config starts with `pretrained_model: null`. For fine-tuning, set
it to an appropriate checkpoint and account for the changed dictionary/output
head. Use matching width settings during export/inference. Training is not run
by the dataset generator, and this command is provided as a starting configuration.

The audit verifies all images, file hashes, UTF-8/NFC labels, actual font-run
cmap coverage, counts, split isolation, dictionary coverage, crop bounds, width
buckets, and CTC sequence capacity assuming width stride four. It does not
measure recognition accuracy or certify the readability of every hard example.

## Downloading and packaging

The distribution uses independent ZIP archives (at most 90 MB each by default)
with a SHA-256 manifest. Extract
**all** parts into the same directory; they are not segments of a split ZIP
stream. The standard-library downloader verifies each part and its ZIP CRCs,
then checks the extracted image/label counts:

```bash
python scripts/download_tr_dataset.py --output turkish-ocr-download
```

Already verified ZIPs are reused if a download is interrupted. If you downloaded
and extracted the complete GitHub tag archive containing all ZIPs, run its
included helper locally without downloading those ZIPs again:

```bash
python download_dataset.py --manifest download-manifest.json --parts-dir . --output .
```

To package your own completed run after both audits pass:

```bash
python scripts/package_tr_dataset.py \
  --dataset outputs/turkish-recognition-100k \
  --output outputs/turkish-recognition-100k-parts
```

## Attribution and redistribution

Turkish Wikipedia contributors: CC BY-SA 4.0,
<https://creativecommons.org/licenses/by-sa/4.0/>. The exact revision URLs and
resource hashes accompany the dataset in `trwiki.sources.json`. This generated
dataset is distributed under CC BY-SA 4.0 with that attribution. Transformations
include text selection, authored fields/confusable spans, rendering, typography,
geometric distortions, cropping/context, noise, lighting, print effects,
occlusions, blur, scaling and compression. Keep the attribution and notices when
redistributing it. Font licenses/notices accompany the data; font binaries and
the downloaded raw corpus are excluded from the code repository and archives.
