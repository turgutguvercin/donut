# Turkish recognition with balanced Wikipedia sources

`config_tr_recognition_balanced.json` extends the camera/lighting recipe with
explicit article sampling. It preserves all existing visual-category quotas.
The older recipes and datasets remain independently reproducible.

**Published v3:** [100,000 images, downloads and audits](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-100k-v3).
Actual labels contain **52,387 distinct readable word forms**
(50,261 in training), up from **13,389** in v2.
All 100,000 images passed dataset and pinned Paddle preprocessing/encoding checks;
duplicate image hashes: 0. Fresh deterministic replay of
940 samples matched exact image hashes and metadata across all 400 source
articles, 62 font files, camera/lighting combinations and width buckets.
All 60,000 natural source strings were checked against their assigned article text.
81 repository tests pass. No neural-network accuracy is claimed.
All Turkish alphabet letters appear in training; foreign letters `ć` and `ň`
appear only in held-out labels, as recorded in the vocabulary report.

Download and verify all parts from `synthdog/`:

```bash
python scripts/download_tr_dataset.py --output turkish-ocr-v3 \
  --manifest https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-100k-v3/download-manifest.json
```

The result is `turkish-ocr-v3/turkish-recognition-100k-v3/`. The release also offers
a browser ZIP containing all parts; extract it and use its helper with
`--manifest download-manifest.json --parts-dir . --output .` for local verification.
The source commit is `6fbcfa37f2d08c11a28efd1132b813216a3ece0a`.

## Source balance

The corpus preparation script selects **400 usable articles across 20 editorial
topics**. Each topic has 18 training articles, one validation article and one test
article. Article titles, revisions, content hashes, topic assignments and splits
are pinned in a source manifest. The same article never supplies target text or
neighboring context to more than one split.

For a 100,000-image run:

| Allocation | Count |
|---|---:|
| Train / validation / test images | 90,000 / 5,000 / 5,000 |
| Train / validation / test articles | 360 / 20 / 20 |
| Natural-text images | 60,000 |
| Natural-text images per topic | 3,000 |
| Natural-text images per article | 150 |
| Samples per article for each of the six natural content types | 25 |
| Dates / prices / generated IDs / confusable exercises | 10,000 each |

Topics cover Turkish and world geography, history, biography, literature,
art/architecture, music, cinema/media, language/education, health, biology,
environment, physics/astronomy, chemistry/materials, mathematics, computing,
engineering/transport, economics/law, sport/daily life and food/agriculture.
These are curated topic assignments, not claims about Wikipedia's categories.

Allocation balances articles within each topic, content type and split, so long
articles cannot dominate simply because they contain more text. Common words
can recur; word counts are not forced equal. Sentence wording stays intact.
Visual balance remains the documented weighted marginal balance, rather than a
full Cartesian product. Font styles balance within each family's available faces.

Canonical target/source strings also retain the existing Turkish-aware hash
split filter. Whole-label equality cannot cross splits; individual words and
substrings can still overlap. Article holdouts give a stronger source separation
than the earlier eight-article corpus. Real detector crops remain necessary for
measuring deployment accuracy. Use one dataset version's splits in an experiment;
mixing versions does not guarantee isolation.

## Prepare and generate

Use the prepared Python environment and the verified licensed font catalog.
Run from `synthdog/`. Cache and corpus outputs should stay outside the Git checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/prepare_tr_balanced_corpus.py \
  --catalog trwiki_topics.json --font-manifest /path/to/recognition-fonts.json \
  --cache /tmp/trwiki-cache --output /tmp/tr-corpus/trwiki.txt

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/generate_tr_dataset.py \
  --config config_tr_recognition_balanced.json \
  --corpus /tmp/tr-corpus/trwiki.txt --font-manifest /path/to/recognition-fonts.json \
  --output outputs/turkish-recognition-100k-v3 --count 100000 \
  --seed 20261014 --workers 4

python scripts/prepare_paddle_tr.py \
  --dataset outputs/turkish-recognition-100k-v3 \
  --cache /tmp/paddleocr-reference --data-dir ./train_data/turkish-recognition-100k-v3

python scripts/analyze_tr_vocabulary.py --dataset outputs/turkish-recognition-100k-v3
```

New article selection can change when Wikipedia changes. For exact reproduction,
use the distributed `trwiki.sources.json` instead of the candidate catalog:

```bash
python scripts/prepare_tr_balanced_corpus.py \
  --manifest /path/to/dataset/trwiki.sources.json \
  --cache /tmp/trwiki-cache --output /tmp/tr-corpus/trwiki.txt
```

Requests are paced, retry transient failures and rate limits, and verify cached
article hashes. Replay requests the exact revisions and verifies the resulting
manifest and corpus hash. Font binaries and raw Wikipedia paragraphs are excluded
from the code branch and dataset download. Derived images/labels retain
CC BY-SA 4.0 licensing, source attribution and font notices.

## Output and checks

Recognition images retain height 48 and proportional widths up to 4,096, with
the existing seven PaddleOCR width buckets and maximum label length 256. Longer
text is not squeezed to width 320. All camera, exposure, shadows, curves, blur,
font styles, underlines, sizes, crop errors and confusable groups remain enabled.
If a randomly positioned shadow misses a short or off-center word, its center
is moved onto actual glyph ink while retaining its shape, strength and random
stream. Metadata records this adjustment and audits require positive ink overlap.

`source-balance.json` reports article/topic allocation and source isolation.
Every sample records its target article, if applicable, and context article.
`vocabulary-report.json` and `word-frequencies.csv` measure the actual labels,
normalizing Turkish capitalization and apostrophes, excluding numeric/ID values
and artificial confusable exercises. Counts describe surface forms, including
inflections, names and abbreviations, not dictionary lemmas.
The vocabulary report also identifies any held-out characters absent from the
training labels; topic balance does not imply equal character frequencies.

The dataset audit checks all image hashes/dimensions, UTF-8/NFC labels, font-run
glyph coverage, category counts, source allocation, split isolation, clipping,
camera/lighting metadata and CTC capacity. Paddle checks exercise pinned upstream
preprocessing and label encoders; they do not train or evaluate a neural network.
Severe crops remain marked potentially ambiguous, with optional filtered lists.
