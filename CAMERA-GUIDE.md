# Turkish recognition: camera angle, exposure and shadows

`config_tr_recognition_camera.json` extends the [100k recipe](turkish-recognition-100k.md)
with explicit views from the left/right, exposure and shadows. Use this config
when generating a new dataset. The published `turkish-ocr-100k-v1` and its default
recipe retain their original images and categories.

**Full 100,000-image v2 dataset:** [release, downloads and audits](https://github.com/turgutguvercin/donut/releases/tag/turkish-ocr-100k-v2).
The camera, exposure and shadow categories below are integrated across the full
90,000 training / 5,000 validation / 5,000 test samples, with the exact listed quotas.
From `synthdog/`, download and verify all ZIP parts automatically:

```bash
python scripts/download_tr_dataset.py --output turkish-ocr-v2
```

The resulting directory is `turkish-ocr-v2/turkish-recognition-100k-v2/`.
The release also offers a browser ZIP containing all parts. After extracting it,
run its included `download_dataset.py --manifest download-manifest.json --parts-dir . --output .`
to verify and extract the parts locally. Full v2 retains all the other typography,
content, confusable-character, crop and degradation categories from the base recipe.
Use one version's grouped splits for an experiment; cross-version split isolation
is not guaranteed.

**Smaller preview set:** [download 1,000 images and labels (11.4 MB ZIP)](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/turkish-camera-lighting-1000-v1.zip).
[Camera comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/comparison-1.jpg) ·
[Lighting comparison](https://raw.githubusercontent.com/turgutguvercin/donut/refs/tags/turkish-camera-lighting-1000-v1/comparison-2.jpg) ·
[Release and audits](https://github.com/turgutguvercin/donut/releases/tag/turkish-camera-lighting-1000-v1).
This run includes 900 training / 50 validation / 50 test samples and 14 separate
controlled comparison crops. All 1,000 samples passed dataset and pinned Paddle
preprocessing checks. Source commit: `9163ad6c3e3552a86b9117f8e1276bbbb0029b68`.

## Categories and balance

| Factor | Quota for 100,000 images |
|---|---|
| Geometry | 40,000 frontal; 20,000 skew; 10,000 each left/mild, right/mild, left/strong, right/strong |
| Exposure | 50,000 normal; 12,500 each low light, underexposed, bright, overexposed |
| Lighting pattern | 50,000 none; 12,500 each gradient, soft shadow, hard shadow, partial shadow |

The other font, content, Turkish confusable-character, curve, crop, blur,
printing and typography quotas remain as documented in the original guide.
Balance means weighted marginal counts, with largest-remainder rounding per
split. Combinations are sampled, subject to the constraints below; every possible
combination is not equally frequent. The output contains category and pair counts.

The named camera side is the **near** side. From the right, the left end is
smaller and the right end is larger. A planar pinhole projection changes both
character height and horizontal spacing, and includes horizontal foreshortening.
The far/near height ratio is 0.70–0.85 for mild views and 0.40–0.60 for strong
views. Sampled yaw ranges are 20–35° and 40–55°, respectively. These are synthetic
camera parameters, not estimates calibrated to a particular phone or lens.
The same mapping is applied to the glyph mask and neighboring context lines.

Exposure and lighting patterns are separate factors, so low light can include a
shadow. Lighting changes the complete crop in linear light before conversion back
to sRGB, including paper, text, and neighboring lines. This differs from fading
only the printed ink. Low light uses −3.8 to −2.5 EV and signal-dependent sensor
noise; underexposure uses −1.8 to −0.8 EV. Bright exposure uses +0.35 to +0.75 EV;
overexposure uses +1.0 to +1.8 EV with veiling light that washes out the ink.
Shadows include a broad soft region, a sharp diagonal boundary, and a band across
part of the line. Uneven illumination uses a brightness gradient. Shallow depth
of field is not part of this extension.

Clean controls, severe clipped crops and faint printing use frontal geometry,
normal exposure and no additional lighting effect. Low-light and overexposed
examples use normal ink contrast, no print damage or occlusion, and no strong
blur. These constraints reduce combinations that would erase all useful text.
Hard examples still require visual review and evaluation on real detector crops.

## Generate and inspect

From `synthdog/`, using the prepared environment and licensed font catalog:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/generate_tr_dataset.py \
  --config config_tr_recognition_camera.json \
  --font-manifest /path/to/recognition-fonts.json \
  --output outputs/turkish-camera-100k \
  --count 100000 --seed 20261014 --workers 4

python scripts/prepare_paddle_tr.py \
  --dataset outputs/turkish-camera-100k \
  --cache /tmp/paddleocr-reference \
  --data-dir ./train_data/turkish-camera-100k

python scripts/preview_tr_capture.py \
  --font-manifest /path/to/recognition-fonts.json \
  --output outputs/turkish-camera-comparison
```

For a smaller complete audit run, use `--count 1000` and a fresh output path.
The separate comparison command renders the same Turkish sentence and font under
14 conditions, including a combined right-side view, low light and hard shadow.
Its PNG crops have exact UTF-8 labels; `comparison-1.jpg` shows geometry and
`comparison-2.jpg` shows lighting. Those repeated comparison labels are for visual
inspection only, outside the training/validation/test splits.

Recognition crops stay **48 pixels high**, with proportional width up to 4,096
and the original PaddleOCR width buckets/CTC capacity checks. Do not squeeze every
crop into 320 pixels. Metadata records camera corners, inverse homography,
far/near ratio, exposure EV, shadow parameters, noise levels and measured shadow
overlap with target ink. Dataset audits check the assigned camera direction and
lighting metadata, image hashes, fonts, UTF-8 labels and quotas. Paddle preprocessing
checks do not measure neural-network accuracy.

Corpus-derived images and labels retain CC BY-SA 4.0 attribution; font binaries
and raw Wikipedia text are excluded from distribution. The controlled comparison
sentence is authored for this repository and distributed under CC BY-SA 4.0.
Generator code uses the repository MIT license.
