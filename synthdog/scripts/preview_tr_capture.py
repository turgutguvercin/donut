#!/usr/bin/env python3
"""Render a controlled Turkish comparison of camera angles, exposure and shadows."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import generate_tr_dataset as dataset
import generate_tr_recognition as base

CASES = [
    ("Frontal / normal light", "none", "normal", "none"),
    ("Camera on LEFT / mild: right end is smaller", "perspective_left_mild", "normal", "none"),
    ("Camera on RIGHT / mild: left end is smaller", "perspective_right_mild", "normal", "none"),
    ("Camera on LEFT / strong: right end is smaller", "perspective_left_strong", "normal", "none"),
    ("Camera on RIGHT / strong: left end is smaller", "perspective_right_strong", "normal", "none"),
    ("Low light / sensor noise", "none", "low_light", "none"),
    ("Underexposed", "none", "underexposed", "none"),
    ("Bright exposure", "none", "bright", "none"),
    ("Overexposed / washed-out ink", "none", "overexposed", "none"),
    ("Soft shadow", "none", "normal", "soft_shadow"),
    ("Hard shadow", "none", "normal", "hard_shadow"),
    ("Partial shadow across the line", "none", "normal", "partial_shadow"),
    ("Uneven illumination", "none", "normal", "gradient"),
    ("RIGHT camera + low light + hard shadow", "perspective_right_strong", "low_light", "hard_shadow"),
]
TEXT = "Çiğdem, ılık ışıkta üzüm ve şeker gördü. İŞÇİ ÇĞİÖŞÜ"


def generate(output, manifest, seed=20261014):
    fonts = base.load_font_manifest(manifest)
    font = next((f for f in fonts if f.family == "Times New Roman" and f.style == "regular"), fonts[0])
    if not set(map(ord, TEXT)) <= font.codepoints:
        raise ValueError("Comparison font lacks required Turkish glyphs")
    output.mkdir(parents=True, exist_ok=False)
    config = {"max_width": 4096}
    rows, images = [], []
    for index, (caption, geometry, exposure, lighting) in enumerate(CASES):
        row = {"mixed_typography": "uniform", "spacing": "normal", "printing": "none",
               "occlusion": "none", "lighting_effect": lighting, "exposure": exposure}
        rng = np.random.default_rng(seed)
        def renderer(label, face, pixels, ink, underline):
            return dataset.rich_patch(label, face, pixels, ink, underline, row, rng, fonts)
        def processor(image, mask, generator):
            return dataset.postprocess(image, mask, generator, row, config)
        image, params = base.render_line(TEXT, font, 42, "white", "clean", rng,
                                         geometry=geometry, patch_renderer=renderer,
                                         image_processor=processor, rotation_jitter=False)
        name = f"comparison-{index:02d}.png"
        image.save(output / name)
        rows.append({"file_name": name, "text": TEXT, "caption": caption, "seed": seed,
                     "font_family": font.family, "font_style": font.style, "font_sha256": font.sha256,
                     "image_sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), **params})
        images.append(image)
    display = ImageFont.truetype(str(font.path), 22)
    width = max(1300, max(im.width for im in images) * 2 + 24)
    for page, start in enumerate((0, 5), 1):
        end = 5 if start == 0 else len(rows)
        sheet = Image.new("RGB", (width, (end - start) * 134 + 45), "#eeeeea")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 8), "Same Turkish text and font; crops shown at 2x. Labels retain the original text.", font=display, fill="black")
        for slot, i in enumerate(range(start, end)):
            y = 45 + slot * 134
            draw.text((12, y), rows[i]["caption"], font=display, fill="black")
            sheet.paste(images[i].resize((images[i].width * 2, 96), Image.Resampling.NEAREST), (12, y + 28))
        sheet.save(output / f"comparison-{page}.jpg", quality=94)
    dataset.write_json(output / "metadata.json", {"samples": rows, "comparison_only": True,
                                                 "license": "CC BY-SA 4.0; authored Turkish test sentence"})
    (output / "labels.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in rows), encoding="utf-8")
    if font.notice:
        shutil.copy2(font.notice, output / ("font-license-" + font.notice.name))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    print(f"Generated {len(generate(args.output, args.font_manifest, args.seed))} comparison crops")


if __name__ == "__main__":
    main()
