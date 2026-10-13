10,000 Turkish synthetic word and line images with matching OCR transcriptions, for recognition dataset experiments.

**[Download all 10,000 images and labels — ZIP, 58.4 MB](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-ocr-10k-v1/turkish-recognition-10000.zip)**

The archive extracts into one directory. The dedicated dataset tag stores the ZIP; use the direct link above to download it.

The ZIP includes:

- 10,000 RGB JPEG crops, each 48 pixels high with variable width.
- `labels.txt`: relative image path, TAB, exact UTF-8 transcription.
- `categories.csv`, `category-counts.json` and `metadata.json`: per-image categories, counts, rendering parameters and hashes.
- `turkish_dict.txt`, previews, validation results, generation scripts and reproduction instructions.
- Wikipedia revision attribution and font license notices. Font binaries are not included.

There are 20 font families (500 images each), 62 actual font files, regular/bold/italic styles, uppercase headings, small/body text, underlines, curved baselines, skew, perspective, ten procedural backgrounds, faint printing, low contrast and several blur/degradation levels. Content includes 6,000 Wikipedia selections, 1,500 dates, 1,500 prices and 1,000 synthetic document IDs.

All 10,000 images were decoded and checked against labels, hashes, font glyph coverage and category records. All 33 tests passed. Visual checks sampled the dataset; this release contains data, not a trained recognizer or an OCR accuracy result.

Wikipedia-derived text is attributed to Turkish Wikipedia contributors under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Exact revision URLs and transformations are recorded in `trwiki.sources.json`. Retain the attribution and accompanying notices when sharing derived samples. Dates, prices and document IDs are generated from templates. The included generator code retains the repository MIT license; font notices retain their respective licenses.

Source: [PR #1](https://github.com/turgutguvercin/donut/pull/1), commit `6f91b51b18e22f6cd42971790e3edd4271e01b02`.

ZIP SHA-256:

```text
fae1c6f7daa60db3f6ab2a275968de99c1925bc33428acfda66735b76e848918
```
