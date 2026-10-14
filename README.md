[Download all 1,000 images and labels — 11.4 MB ZIP](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/turkish-camera-lighting-1000-v1.zip)

[Camera-angle comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/comparison-1.jpg) · [Exposure and shadows](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/comparison-2.jpg) · [Mixed dataset examples](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/dataset-preview.jpg)

[Checksums](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/SHA256SUMS) · [Dataset audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/audit.json) · [Paddle preprocessing audit](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/paddle-preprocessing-audit.json) · [Implementation PR](https://github.com/turgutguvercin/donut/pull/1)

# Turkish OCR: camera angles, lighting and shadows — 1,000 samples

1,000 RGB recognition JPEGs with exact UTF-8 labels: 900 train / 50 validation / 50 test.
The extra 14 PNG crops in comparisons/ repeat one authored Turkish sentence to show the effects side by side; they are outside the dataset splits.

New geometry counts: {"perspective_left_mild": 100, "none": 400, "skew": 200, "perspective_left_strong": 100, "perspective_right_strong": 100, "perspective_right_mild": 100}.
Exposure counts: {"overexposed": 124, "underexposed": 125, "normal": 500, "bright": 124, "low_light": 127}.
Lighting counts: {"partial_shadow": 124, "none": 500, "hard_shadow": 124, "soft_shadow": 125, "gradient": 127}.
20 font families and 62 licensed font files are represented. Marginal quotas use per-split rounding; see category-counts.json for all factors.
For 100,000 samples the recipe assigns 10,000 to each side/strength view and 12,500 to each non-normal exposure/shadow category.

From the right, the left end is farther away and smaller. Both glyph height and spacing taper.
Low light, underexposure, brighter exposure, washed-out highlights, gradients and soft/hard/partial shadows act on the whole crop.
Some samples combine perspective, low light and shadows. CAMERA-GUIDE.md documents the recipe and readability constraints.

Images remain 48 pixels high, with proportional widths 31–4096; maximum allowed width 4096.
Use the included PaddleOCR configurations with matching width buckets and max_text_length=256, rather than squeezing long lines to 320.
All 1,000 images passed decode/hash, UTF-8/NFC, font coverage, split isolation, quota, camera-direction and shadow-overlap checks.
Pinned PaddleOCR resize and CTC/NRTR encoding also passed for all 1,000, without horizontal squeezing or lost label characters.
70 repository tests passed; 232 earlier samples replayed byte-for-byte across all 62 font files.
No neural-network training or real-world accuracy was tested. Severe detector crops retain source labels and have ambiguity flags/filter lists.

Generator source commit: 9163ad6c3e3552a86b9117f8e1276bbbb0029b68. Seed: 20261014.
Reproduction source snapshot is in reproduce/. Clone the repository at that commit to rerun; prepare the attributed corpus,
acquire licensed fonts matching font-manifest.json, update their local paths, and use:

```bash
cd synthdog
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/generate_tr_dataset.py \
  --config config_tr_recognition_camera.json --font-manifest /path/to/fonts.json \
  --output outputs/turkish-camera-1000 --count 1000 --seed 20261014 --workers 4
```

Generated images and labels are CC BY-SA 4.0. Keep LICENSE-DATASET.txt, trwiki.sources.json and licenses/.
The comparison sentence is authored for this repository and licensed CC BY-SA 4.0.
No raw Wikipedia text or font binaries are included. Code: MIT. Paddle-derived configurations: Apache-2.0.
This sample set is separate from the previously published 100,000-image v1 dataset.
