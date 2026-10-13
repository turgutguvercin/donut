#!/usr/bin/env python3
"""Generate varied Turkish line crops and PaddleOCR recognition labels."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

from fontTools.ttLib import TTFont
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from synthtiger.layers import TextLayer

TURKISH = "çğışöüÇĞİŞÖÜıI"
STYLES = ("regular", "bold", "italic", "bold_italic")
BACKGROUNDS = (
    "white", "aged_paper", "blue_form", "receipt", "ruled_paper",
    "graph_paper", "pink_form", "photocopy", "uneven_light", "dark_sign",
)
DEGRADATIONS = ("clean", "defocus", "motion_blur", "low_resolution", "noisy_scan")
# Real italic/oblique files, never simulated by tilting an upright font.
FAMILIES = (
    ("DejaVu Sans", "DejaVuSans", ("", "-Bold", "-Oblique", "-BoldOblique"), "dejavu"),
    ("DejaVu Serif", "DejaVuSerif", ("", "-Bold", "-Italic", "-BoldItalic"), "dejavu"),
    ("DejaVu Sans Mono", "DejaVuSansMono", ("", "-Bold", "-Oblique", "-BoldOblique"), "dejavu"),
    ("Liberation Sans", "LiberationSans", ("-Regular", "-Bold", "-Italic", "-BoldItalic"), "liberation"),
    ("Liberation Serif", "LiberationSerif", ("-Regular", "-Bold", "-Italic", "-BoldItalic"), "liberation"),
)


@dataclass
class Font:
    path: Path
    family: str
    style: str
    license: str
    codepoints: set
    sha256: str


def discover_fonts(root):
    paths = {p.name: p for p in sorted(root.rglob("*.ttf"))}
    fonts = []
    for style_idx, style in enumerate(STYLES):
        for family, prefix, suffixes, license_name in FAMILIES:
            filename = prefix + suffixes[style_idx] + ".ttf"
            if filename not in paths:
                raise FileNotFoundError(
                    f"Missing {filename}. Install fonts-dejavu-core, fonts-dejavu-extra "
                    "and fonts-liberation, or pass --font-root with these licensed fonts."
                )
            with TTFont(paths[filename]) as face:
                cmap = face.getBestCmap()
                codepoints = {key for key, value in cmap.items() if value != ".notdef"}
            if not set(map(ord, TURKISH)) <= codepoints:
                raise ValueError(f"Incomplete Turkish glyph coverage: {filename}")
            fonts.append(Font(paths[filename], family, style, license_name, codepoints,
                              hashlib.sha256(paths[filename].read_bytes()).hexdigest()))
    return fonts


def sample_text(words, font, rng):
    for _ in range(1000):
        start = int(rng.integers(len(words)))
        target = int(rng.integers(25, 57))
        selected = []
        for word in words[start:start + 20]:
            if len(" ".join(selected + [word])) > target:
                break
            selected.append(word)
        text = " ".join(selected)
        if len(text) < 15 or not set(text) & set("çğıöşüİ"):
            continue
        # SynthTiger replaces backslashes; exclude those and invisible controls.
        if any(c == "\\" or unicodedata.category(c).startswith("C") for c in text):
            continue
        if any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in text):
            continue
        if set(map(ord, text)) <= font.codepoints:
            return text
    raise ValueError("Corpus has insufficient suitable Turkish text for the selected font")


def background(size, kind, rng):
    width, height = size
    colors = {
        "white": (250, 250, 248), "aged_paper": (222, 204, 162),
        "blue_form": (211, 230, 244), "receipt": (236, 236, 227),
        "ruled_paper": (242, 240, 219), "graph_paper": (226, 239, 233),
        "pink_form": (243, 217, 224), "photocopy": (219, 219, 214),
        "uneven_light": (239, 230, 210), "dark_sign": (36, 57, 62),
    }
    field = np.full((height, width, 3), colors[kind], dtype=float)
    noise = 0.5 if kind == "white" else float(rng.uniform(2, 6))
    field += rng.normal(0, noise, (height, width, 1))
    if kind in ("aged_paper", "photocopy", "receipt", "dark_sign"):
        texture_sigma = {"aged_paper": 12, "photocopy": 10, "receipt": 5, "dark_sign": 6}[kind]
        texture = rng.normal(0, texture_sigma, (max(2, height // 8), max(2, width // 8)))
        texture = Image.fromarray(np.clip(texture + 128, 0, 255).astype("uint8"))
        field += (np.asarray(texture.resize(size, Image.Resampling.BICUBIC), dtype=float) - 128)[..., None]
    if kind in ("uneven_light", "photocopy"):
        # Broad shadows, stronger at one edge, like an uneven scan or photograph.
        x = np.linspace(0, 1, width)
        field *= (0.68 + 0.32 * np.sin(x * np.pi / 2))[None, :, None]
    image = Image.fromarray(np.clip(field, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(image)
    if kind in ("ruled_paper", "graph_paper"):
        for y in range(9, height, 18):
            draw.line((0, y, width, y), fill=(177, 197, 197), width=1)
        if kind == "graph_paper":
            for x in range(4, width, 18):
                draw.line((x, 0, x, height), fill=(192, 211, 202), width=1)
    if kind == "aged_paper":
        for _ in range(width // 8):
            x, y = int(rng.integers(width)), int(rng.integers(height))
            draw.line((x, y, x + int(rng.integers(2, 9)), y + 1), fill=(191, 175, 140))
    if kind == "receipt":
        for x in range(0, width, 31):
            draw.line((x, 0, x, height), fill=(216, 216, 210))
    return image


def render_line(text, font, size, kind, degradation, rng):
    ink = (241, 236, 216) if kind == "dark_sign" else (
        (24, 45, 80) if kind in ("blue_form", "pink_form") else (35, 32, 29)
    )
    layer = TextLayer(text, path=str(font.path), size=size, color=ink)
    rgba = Image.fromarray(layer.image.astype("uint8"), "RGBA")
    bounds = rgba.getbbox()
    if bounds is None:
        raise ValueError(f"Empty rendered text: {text}")
    rgba = rgba.crop(bounds)
    padding_x, padding_y = int(rng.integers(8, 18)), int(rng.integers(7, 16))
    # Larger padding on some lines deliberately varies relative text height.
    crop_size = (rgba.width + padding_x * 2, max(48, rgba.height + padding_y * 2))
    image = background(crop_size, kind, rng)
    image.paste(rgba, (padding_x, (image.height - rgba.height) // 2), rgba)
    angle = float(rng.uniform(-0.7, 0.7))
    image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=image.getpixel((0, 0)))
    image = image.resize((round(image.width * 48 / image.height), 48), Image.Resampling.LANCZOS)
    params = {"rotation_degrees": angle, "source_font_size": size,
              "background": kind, "degradation": degradation}
    if degradation == "defocus":
        params["blur_radius"] = float(rng.uniform(0.75, 1.25))
        image = image.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    elif degradation == "motion_blur":
        image = image.filter(ImageFilter.Kernel((3, 3), (0, 0, 0, 1, 1, 1, 0, 0, 0), scale=3))
        params["motion_kernel_width"] = 3
    elif degradation == "low_resolution":
        params["downsample_scale"] = float(rng.uniform(0.5, 0.68))
        low_size = tuple(max(1, round(v * params["downsample_scale"])) for v in image.size)
        image = image.resize(low_size, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BICUBIC)
    elif degradation == "noisy_scan":
        params["noise_sigma"] = float(rng.uniform(6, 11))
        array = np.asarray(image, dtype=float)
        array += rng.normal(0, params["noise_sigma"], array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype("uint8"))
        image = image.filter(ImageFilter.GaussianBlur(0.25))
    params["jpeg_quality"] = int(rng.integers(48, 76)) if degradation != "clean" else 96
    return image, params


def contact_sheet(root, samples):
    # Keep preview memory bounded when generating thousands of training lines.
    samples = samples[:40]
    rows = (len(samples) + 1) // 2
    cell_width, cell_height = 880, 116
    sheet = Image.new("RGB", (cell_width * 2, rows * cell_height), (239, 242, 246))
    draw = ImageDraw.Draw(sheet)
    ui_font = ImageFont.truetype(samples[0]["font_path"], 15)
    for idx, sample in enumerate(samples):
        x, y = (idx // rows) * cell_width + 12, (idx % rows) * cell_height + 8
        title = f"{idx + 1:02d}  {sample['family']} / {sample['style']} / {sample['background']} / {sample['degradation']}"
        draw.text((x, y), title, font=ui_font, fill=(42, 47, 57))
        with Image.open(root / sample["file_name"]) as image:
            if image.width > cell_width - 28:
                image = image.resize((cell_width - 28, round(image.height * (cell_width - 28) / image.width)))
            sheet.paste(image, (x, y + 29))
    sheet.save(root / "contact-sheet.jpg", quality=95)


def generate(corpus, output, font_root, count=20, seed=42):
    if count < 1:
        raise ValueError("count must be positive")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    text = unicodedata.normalize("NFC", corpus.read_text(encoding="utf-8"))
    words = text.split()
    if not words:
        raise ValueError("Corpus is empty")
    fonts = discover_fonts(font_root)
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    samples, labels = [], []
    for idx in range(count):
        font = fonts[idx % len(fonts)]
        label = sample_text(words, font, rng)
        # Balance the first 20; rotate assignments on later cycles so larger
        # datasets do not associate one font/style with just one background.
        cycle = idx // len(fonts)
        kind = BACKGROUNDS[(idx + cycle * 3) % len(BACKGROUNDS)]
        degradation = DEGRADATIONS[(idx + idx // 5 + cycle) % len(DEGRADATIONS)]
        font_size = (22, 30, 34, 18, 28, 24, 36, 20)[idx % 8]
        image, params = render_line(label, font, font_size, kind, degradation, rng)
        relative = f"images/line_{idx:05d}.jpg"
        image.save(output / relative, quality=params["jpeg_quality"])
        labels.append(f"{relative}\t{label}")
        samples.append({"file_name": relative, "text": label, "family": font.family,
                        "style": font.style, "font_path": str(font.path),
                        "font_sha256": font.sha256,
                        "font_license": font.license, "width": image.width, "height": image.height,
                        "image_sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(), **params})
    (output / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    characters = sorted((set("".join(s["text"] for s in samples)) | set(TURKISH)) - {" "})
    (output / "turkish_dict.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")
    metadata = {"seed": seed, "count": count, "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "families": dict(Counter(s["family"] for s in samples)),
                "styles": dict(Counter(s["style"] for s in samples)),
                "backgrounds": dict(Counter(s["background"] for s in samples)),
                "degradations": dict(Counter(s["degradation"] for s in samples)), "samples": samples}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = corpus.with_suffix(".sources.json")
    if manifest.exists():
        shutil.copy2(manifest, output / "trwiki.sources.json")
    license_dir = output / "licenses"
    license_dir.mkdir()
    for family, package in (("dejavu", "fonts-dejavu-core"), ("liberation", "fonts-liberation")):
        notice = Path("/usr/share/doc") / package / "copyright"
        if notice.exists():
            shutil.copy2(notice, license_dir / (family + ".txt"))
    (output / "README.txt").write_text(
        "PaddleOCR recognition line crops, height 48, variable width.\n"
        "labels.txt has relative image path, TAB, exact UTF-8 text. Set data_dir to this directory.\n"
        "Use turkish_dict.txt with use_space_char: true; this dictionary covers the samples, not all future corpora.\n"
        "Use a compatible recognition head/dictionary when fine-tuning pretrained weights.\n"
        "Five real font families with regular, bold, italic/oblique and bold italic styles; "
        "ten procedural backgrounds and five degradation types. See metadata.json for all parameters and hashes.\n"
        "Fonts are system DejaVu (Bitstream Vera license; DejaVu changes public domain) and Liberation (SIL OFL 1.1). "
        "Font binaries are not included. See licenses/ and system package copyright notices.\n"
        "For trwiki.txt: Turkish Wikipedia contributors, CC BY-SA 4.0, "
        "https://creativecommons.org/licenses/by-sa/4.0/. Revision attribution: trwiki.sources.json.\n"
        "Transformations: whole-word selection, text rendering, procedural backgrounds, rotation, scaling, "
        "blur/downsampling/noise and JPEG compression. This sample set is not evidence of OCR accuracy.\n",
        encoding="utf-8",
    )
    contact_sheet(output, samples)
    validate(output)
    return metadata


def validate(root):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    records = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
    characters = set((root / "turkish_dict.txt").read_text(encoding="utf-8").splitlines()) | {" "}
    if len(records) != metadata["count"] or len(metadata["samples"]) != len(records):
        raise ValueError("Wrong sample count")
    paths = set()
    for record, sample in zip(records, metadata["samples"]):
        name, label = record.split("\t", 1)
        path = root / name
        if path in paths or name != sample["file_name"] or label != sample["text"]:
            raise ValueError("Duplicate or mismatched image/label")
        if not label or "\ufffd" in label or not set(label) <= characters:
            raise ValueError("Invalid label or dictionary coverage")
        paths.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["image_sha256"]:
            raise ValueError("Image checksum mismatch")
        with Image.open(path) as image:
            if image.size != (sample["width"], 48):
                raise ValueError("Wrong image dimensions")
            image.verify()
    if paths != set((root / "images").glob("*.jpg")):
        raise ValueError("Missing or unlabelled images")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("resources/corpus/trwiki.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/PaddleOCR_tr_recognition"))
    parser.add_argument("--font-root", type=Path, default=Path("/usr/share/fonts/truetype"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = generate(args.corpus, args.output, args.font_root, args.count, args.seed)
    print(f"Validated {result['count']} UTF-8 image/label pairs in {args.output}")
    for key in ("families", "styles", "backgrounds", "degradations"):
        print(key + ": " + json.dumps(result[key], ensure_ascii=False))


if __name__ == "__main__":
    main()
