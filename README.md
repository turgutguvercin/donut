100 Turkish OCR recognition crops demonstrating imperfect text-detector boundaries.

**[Download 100 images and labels — ZIP, 3.85 MB](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-detector-crops-100-v1/turkish-detector-crops-100.zip)**

[View the crop comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-detector-crops-100-v1/preview.jpg)

![The same target phrase under eight crop settings](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-detector-crops-100-v1/preview.jpg)

The batch contains 60 intact crops, 5 with neighboring text above, 5 below,
10 on both sides, 5 tight at the top, 5 tight at the bottom, 5 tight at both
edges, and 5 shifted vertically. There are 20 font families and 62 actual font
files, along with the existing typography, backgrounds, curves, capture geometry,
structured content and degradation categories.

Labels transcribe only the target line. Neighbor text, visible ink fractions,
crop rectangles and clipping measurements are recorded in metadata. Tight crops
retain at least 97% of target glyph ink and half of each connected component
before resizing and degradation; underline strokes are excluded. This limits
partial damage to Turkish dots, accents and descenders but does not guarantee
readability after blur. Completely missing marks cannot be recovered reliably.

Validation: 42 tests passed, all 100 images decoded and matched to UTF-8/NFC
labels, font coverage and categories. A separate 1,000-image run also validated.
The no-crop profile reproduced 80 previous images/labels exactly; the existing
10,000-image release remains valid. No model training or OCR accuracy is claimed.

The archive includes labels, character dictionary, metadata, category CSV/JSON,
previews, validation report, generator code, reproduction commands and notices.
Font binaries and the raw corpus are excluded. Wikipedia-derived text is from
Turkish Wikipedia contributors under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/);
retain trwiki.sources.json and the notices when redistributing. Code retains the
repository MIT license; font notices retain their respective terms.

Source: [PR #1](https://github.com/turgutguvercin/donut/pull/1), commit `86fc3176d94a3be28b59cd4b0aa2730ffb5c06ba`.

This is a new 100-image preview. The earlier [10,000-image release](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-10k-v1)
uses the previous crop behavior.

ZIP SHA-256: `2cfa0b5e9a02b5ce2595535fdec042f398f65a78f71e7e33874289876cefab6b`
