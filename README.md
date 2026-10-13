100 Turkish OCR recognition images with stronger clipping and longer sentences.

**[Download 100 images and labels — ZIP, 5.5 MB](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-severe-long-100-v1/turkish-severe-long-100.zip)**

[Compare missing dots and cedillas](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-severe-long-100-v1/marks-preview.jpg) · [View long sentences](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-severe-long-100-v1/sentences-preview.jpg)

![Mild versus severe clipping](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-severe-long-100-v1/marks-preview.jpg)

The batch contains 60 complete Turkish Wikipedia sentences: 30 in the 91–160
character range and 30 in the 161–240 range. Actual selected lengths are 95–238
characters. The other targets are 15 dates, 15 prices and 10 synthetic document
IDs. There are 20 font families and 62 actual font files, with varied styles,
typography, backgrounds, curves, capture geometry and degradation.

20 crops use severe clipping, including 11 long sentences. Severe top/bottom
cuts can remove whole dots, accents or cedillas, so ü can resemble u and ç can
resemble c. These samples retain their original target labels and are flagged
potentially_ambiguous in metadata. The severe mode keeps at least 70% of target
glyph ink before resizing/degradation, while allowing entire connected marks
to disappear. Missing information can make exact recognition impossible.

The remaining crop categories are 60 intact and 20 with neighboring-line
fragments. Only the target line belongs in labels.txt. Context text, severity,
component losses, crop rectangles and rendering parameters are in metadata.

All images retain their full width at height 48; widths range from 92 to 6687
pixels. Configure recognition width and maximum label length accordingly.
The sentence preview wraps each original line into horizontal display sections;
the underlying training image remains a single line.

Validation: 47 tests passed, every image decoded and matched to UTF-8/NFC labels,
font coverage, image hashes and category exports. The full 100-image batch
reproduced exactly from its seed. A separate 200-image mixed-severity run passed.
Prior mild-crop output remains byte-identical, and the previous 100/10k datasets
still validate. This release contains data, not a trained recognizer or accuracy result.

Included: images, labels, dictionary, category CSV/JSON, metadata, previews,
validation report, source scripts, reproduction commands and license notices.
Font binaries and the raw corpus are excluded. Wikipedia-derived text is by
Turkish Wikipedia contributors under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
retain trwiki.sources.json and the notices when redistributing. Code retains
the repository MIT license; font notices retain their respective terms.

Source: [PR #1](https://github.com/turgutguvercin/donut/pull/1), commit `2505a59478dc797997c452a050ce077d12d33f13`.

ZIP SHA-256: `0de093a293084d15546f005b59ccb9fdd26609c7edfb3c48cea403f55d20a5d3`
