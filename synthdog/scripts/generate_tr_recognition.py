#!/usr/bin/env python3
"""Generate varied Turkish line crops and PaddleOCR recognition labels."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time
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
CURVES = ("straight", "arch", "bowl", "wave")
STRONG_DEGRADATIONS = (
    "strong_defocus", "strong_motion", "defocus_motion",
    "strong_defocus", "strong_motion", "defocus_motion",
    "defocus", "low_resolution", "noisy_scan", "clean",
)
BALANCED_DEGRADATIONS = (
    *("clean",) * 6, *("defocus",) * 4, *("motion_blur",) * 2,
    *("low_resolution",) * 2, *("noisy_scan",) * 2,
    *("strong_defocus",) * 2, "strong_motion", "defocus_motion",
)
CAPTURE_GEOMETRIES = ("none", "none", "skew", "perspective")
CONTRASTS = (*("normal",) * 6, *("low_contrast",) * 3, "faint_print")
CONTENT_TYPES = (*("wikipedia",) * 12, *("date",) * 3, *("price",) * 3,
                 *("synthetic_id",) * 2)
TEXT_LENGTHS = {"single_word": (3, 17), "short_line": (15, 31),
                "medium_line": (31, 56), "long_line": (56, 91)}
TYPOGRAPHIES = (
    ("heading", "uppercase", False, (48, 64), (16, 31)),
    ("heading", "uppercase", True, (48, 64), (16, 31)),
    ("body", "original", True, (26, 38), (25, 57)),
    ("small", "original", False, (18, 22), (35, 72)),
)
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
    notice: Path = None
    source: str = None


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


def load_font_manifest(path):
    """Load additional licensed fonts without requiring or copying binaries."""
    fonts, seen = [], set()
    for entry in json.loads(path.read_text(encoding="utf-8"))["fonts"]:
        font_path = (path.parent / entry["path"]).resolve()
        notice = (path.parent / entry["license_file"]).resolve()
        if not notice.is_file() or not entry.get("license"):
            raise ValueError(f"Missing font license notice: {font_path}")
        with TTFont(font_path) as face:
            family = face["name"].getBestFamilyName()
            subfamily = face["name"].getBestSubFamilyName().lower()
            italic = "italic" in subfamily or "oblique" in subfamily
            bold = "bold" in subfamily or "demi" in subfamily
            style = "bold_italic" if bold and italic else "italic" if italic else "bold" if bold else "regular"
            codepoints = {key for key, value in face.getBestCmap().items() if value != ".notdef"}
        if not set(map(ord, TURKISH)) <= codepoints:
            raise ValueError(f"Incomplete Turkish glyph coverage: {font_path}")
        digest = hashlib.sha256(font_path.read_bytes()).hexdigest()
        if entry.get("sha256", digest) != digest:
            raise ValueError(f"Font checksum mismatch: {font_path}")
        if digest in seen:
            raise ValueError(f"Duplicate font in manifest: {font_path}")
        seen.add(digest)
        fonts.append(Font(font_path, family, style, entry["license"], codepoints,
                          digest, notice, entry.get("source")))
    if not fonts:
        raise ValueError("Font manifest is empty")
    return fonts


def turkish_upper(text):
    # Python's default casing maps i to I; Turkish requires i -> İ and ı -> I.
    return unicodedata.normalize("NFC", text.translate(str.maketrans({"i": "İ", "ı": "I"})).upper())


def sample_text(words, font, rng, text_case="original", length_range=(25, 57),
                minimum_length=15, single_word=False):
    for _ in range(1000):
        start = int(rng.integers(len(words)))
        target = int(rng.integers(*length_range))
        selected = []
        for word in words[start:start + (1 if single_word else 30)]:
            if len(" ".join(selected + [word])) > target:
                break
            selected.append(word)
        text = " ".join(selected)
        if len(text) < minimum_length or not set(text) & set("çğıöşüİ"):
            continue
        # SynthTiger replaces backslashes; exclude those and invisible controls.
        if any(c == "\\" or unicodedata.category(c).startswith("C") for c in text):
            continue
        if any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in text):
            continue
        if text_case == "uppercase":
            text = turkish_upper(text)
        # Case conversion can introduce different glyphs; validate the final label.
        if set(map(ord, text)) <= font.codepoints:
            return text
    raise ValueError("Corpus has insufficient suitable Turkish text for the selected font")


def structured_text(kind, font, rng, text_case="original"):
    """Invent document fields; identifiers never come from personal records."""
    months = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
              "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
    for _ in range(100):
        if kind == "date":
            day = date(2000, 1, 1) + timedelta(days=int(rng.integers(0, 11323)))
            form = int(rng.integers(4))
            values = (day.strftime("%d.%m.%Y"), day.strftime("%d/%m/%Y"),
                      f"{day.day} {months[day.month - 1]} {day.year}", day.isoformat())
            text = str(rng.choice(("", "Tarih: ", "İşlem tarihi: "))) + values[form]
        elif kind == "price":
            whole, cents = int(rng.integers(1, 100000)), int(rng.integers(100))
            amount = f"{whole:,}".replace(",", ".") + f",{cents:02d}"
            currency = str(rng.choice(("TL", "₺"))) if ord("₺") in font.codepoints else "TL"
            text = str(rng.choice(("", "Tutar: ", "Ücret: ", "Ödenecek: "))) + amount + " " + currency
        elif kind == "synthetic_id":
            prefix = str(rng.choice(("Belge No: ", "Sipariş: ", "Ürün Kodu: ", "Kayıt: ")))
            text = f"{prefix}TR-{int(rng.integers(2000, 2031))}-{int(rng.integers(1000000)):06d}"
        else:
            raise ValueError(f"Unknown structured content type: {kind}")
        if text_case == "uppercase":
            text = turkish_upper(text)
        if set(map(ord, text)) <= font.codepoints:
            return text
    raise ValueError(f"Font cannot render structured content: {font.family}")


def balanced_choices(values, count, rng):
    choices = np.tile(values, (count + len(values) - 1) // len(values))[:count]
    rng.shuffle(choices)
    return choices


def balance_capture_effects(effects, contrasts):
    """Retain exact preset counts while avoiding unreadable effect combinations."""
    strong = {"strong_defocus", "strong_motion", "defocus_motion"}
    for idx, contrast in enumerate(contrasts):
        if contrast == "faint_print" and effects[idx] != "clean":
            candidates = np.flatnonzero((contrasts == "normal") & (effects == "clean"))
        elif contrast == "low_contrast" and effects[idx] in strong:
            candidates = np.flatnonzero((contrasts == "normal") & ~np.isin(effects, list(strong)))
        else:
            continue
        if not len(candidates):
            raise ValueError("Insufficient readable combinations for the selected presets")
        other = int(candidates[0])
        effects[idx], effects[other] = effects[other], effects[idx]
    return effects


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


def text_patch(text, font, size, ink, underline=False):
    layer = TextLayer(text, path=str(font.path), size=size, color=ink)
    rgba = Image.fromarray(layer.image.astype("uint8"), "RGBA")
    bounds = rgba.getbbox()
    if bounds is None:
        raise ValueError(f"Empty rendered text: {text}")
    rgba = rgba.crop(bounds)
    text_height = rgba.height
    gap = max(2, round(size * 0.06)) if underline else 0
    thickness = max(1, round(size * 0.04)) if underline else 0
    if underline:
        patch = Image.new("RGBA", (rgba.width, rgba.height + gap + thickness))
        patch.paste(rgba, (0, 0))
        # Place the line below the full glyph bounds, including ç/ğ descenders.
        ImageDraw.Draw(patch).rectangle(
            (0, rgba.height + gap, rgba.width - 1, patch.height - 1), fill=(*ink, 255)
        )
        rgba = patch
    return rgba, {"underline": underline, "underline_gap_native": gap,
                  "underline_thickness_native": thickness, "text_height_native": text_height}


def curve_patch(patch, curve, amplitude):
    """Shift complete RGBA columns onto a curved baseline, without clipping."""
    if curve not in CURVES:
        raise ValueError(f"Unknown curve: {curve}")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Curve amplitude must be finite and nonnegative")
    if curve == "straight" or amplitude == 0:
        return patch
    pixels = np.asarray(patch)
    height, width = pixels.shape[:2]
    x = np.linspace(-1, 1, width)
    profile = x ** 2 if curve == "arch" else 1 - x ** 2 if curve == "bowl" else (
        1 + np.sin(x * np.pi)
    ) / 2
    offsets = np.rint(amplitude * profile).astype(int)
    output = np.zeros((height + int(offsets.max()), width, 4), dtype="uint8")
    # Preserve character order, accents and underline pixels column by column.
    output[np.arange(height)[:, None] + offsets[None, :], np.arange(width)[None, :]] = pixels
    return Image.fromarray(output, "RGBA")


def perspective_patch(patch, destination):
    """Map all four corners onto a padded canvas using an inverse homography."""
    source = np.array(((0, 0), (patch.width - 1, 0),
                       (patch.width - 1, patch.height - 1), (0, patch.height - 1)), dtype=float)
    target = np.asarray(destination, dtype=float)
    target = target - np.floor(target.min(axis=0)) + 4
    width, height = np.ceil(target.max(axis=0)).astype(int) + 5
    equations, values = [], []
    for (x, y), (u, v) in zip(target, source):
        equations.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y),
                          (0, 0, 0, x, y, 1, -v * x, -v * y)))
        values.extend((u, v))
    coefficients = np.linalg.solve(np.asarray(equations), np.asarray(values))
    warped = patch.transform((int(width), int(height)), Image.Transform.PERSPECTIVE,
                             coefficients, Image.Resampling.BICUBIC)
    return warped, {"perspective_source_corners": source.tolist(),
                    "perspective_target_corners": target.tolist(),
                    "perspective_inverse_coefficients": coefficients.tolist()}


def motion_blur(image, width, direction="horizontal"):
    kernel = np.zeros((width, width), dtype=float)
    if direction == "horizontal":
        kernel[width // 2, :] = 1
    elif direction == "vertical":
        kernel[:, width // 2] = 1
    else:
        np.fill_diagonal(kernel, 1)
    return image.filter(ImageFilter.Kernel((width, width), tuple(kernel.ravel()), scale=width))


def render_line(text, font, size, kind, degradation, rng, underline=False, role="body",
                curve="straight", curve_strength=0.5, geometry="none", contrast="normal"):
    if geometry not in CAPTURE_GEOMETRIES or contrast not in CONTRASTS:
        raise ValueError("Unknown capture geometry or contrast")
    ink = (241, 236, 216) if kind == "dark_sign" else (
        (24, 45, 80) if kind in ("blue_form", "pink_form") else (35, 32, 29)
    )
    rgba, decoration = text_patch(text, font, size, ink, underline)
    amplitude = round(decoration["text_height_native"] * curve_strength) if curve != "straight" else 0
    rgba = curve_patch(rgba, curve, amplitude)
    geometry_params = {}
    perspective_scale = 1.0
    if geometry == "perspective":
        width, height = rgba.size
        taper = (0.04, 0.10) if role == "small" else (0.10, 0.25)
        inset = float(rng.uniform(*taper) * (height - 1))
        shear = float(rng.uniform(-0.12, 0.12) * height)
        left = bool(rng.integers(2))
        destination = ((shear, inset if left else 0),
                       (width - 1 + shear, 0 if left else inset),
                       (width - 1, height - 1 if left else height - 1 - inset),
                       (0, height - 1 - inset if left else height - 1))
        rgba, geometry_params = perspective_patch(rgba, destination)
        perspective_scale = 1 - 2 * inset / (height - 1)
    opacity_limits = (0.50, 0.68) if contrast == "low_contrast" else (0.28, 0.42)
    opacity = 1.0 if contrast == "normal" else float(rng.uniform(*opacity_limits))
    if opacity < 1:
        rgba.putalpha(rgba.getchannel("A").point(lambda value: round(value * opacity)))
    padding_x = int(rng.integers(8, 18))
    padding_y = int(rng.integers(4, 9) if role == "heading" else rng.integers(7, 16))
    if geometry == "perspective" and role == "small":
        # Perspective already adds transparent padding. Avoid normalizing a
        # mostly empty crop while the narrow end contains tiny characters.
        padding_y = min(padding_y, 5)
    # Larger padding on some lines deliberately varies relative text height.
    minimum_height = 0 if geometry == "perspective" and role == "small" else 48
    crop_size = (rgba.width + padding_x * 2, max(minimum_height, rgba.height + padding_y * 2))
    image = background(crop_size, kind, rng)
    image.paste(rgba, (padding_x, (image.height - rgba.height) // 2), rgba)
    if geometry == "skew":
        # A fixed angle on a long line can shrink every character to a few
        # pixels after height normalization. Bound the added vertical extent.
        maximum = min(4.0, float(np.degrees(np.arctan(0.6 * rgba.height / rgba.width))))
        angle = float(rng.uniform(0.6, 1.0) * maximum * rng.choice((-1, 1)))
    elif geometry == "perspective":
        maximum = min(0.7, float(np.degrees(np.arctan(0.15 * rgba.height / rgba.width))))
        angle = float(rng.uniform(-maximum, maximum))
    else:
        angle = float(rng.uniform(-0.7, 0.7))
    image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=image.getpixel((0, 0)))
    text_height_output = decoration["text_height_native"] * perspective_scale * 48 / image.height
    curve_amplitude_output = amplitude * 48 / image.height
    image = image.resize((round(image.width * 48 / image.height), 48), Image.Resampling.LANCZOS)
    params = {"rotation_degrees": angle, "source_font_size": size,
              "background": kind, "degradation": degradation,
              "curve": curve, "curve_strength": curve_strength if curve != "straight" else 0,
              "curve_amplitude_native": amplitude,
              "curve_amplitude_output": curve_amplitude_output,
              "text_height_output": text_height_output,
              "geometry": geometry, "contrast": contrast, "ink_opacity": opacity,
              **geometry_params, **decoration}
    if degradation == "defocus":
        params["blur_radius"] = float(rng.uniform(0.35, 0.65) if role == "small" else rng.uniform(0.75, 1.25))
        image = image.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
    elif degradation == "motion_blur":
        image = motion_blur(image, 3)
        params["motion_kernel_width"] = 3
        params["motion_direction"] = "horizontal"
    elif degradation in ("strong_defocus", "strong_motion", "defocus_motion"):
        # Keep tiny glyphs from disappearing completely under a fixed-size blur.
        scale = float(np.clip(text_height_output / 30, 0.35, 1.0))
        if degradation in ("strong_defocus", "defocus_motion"):
            limits = (1.3, 2.3) if degradation == "strong_defocus" else (0.9, 1.6)
            params["blur_radius"] = float(rng.uniform(*limits) * scale)
            image = image.filter(ImageFilter.GaussianBlur(params["blur_radius"]))
        if degradation in ("strong_motion", "defocus_motion"):
            params["motion_kernel_width"] = 5 if text_height_output >= 18 else 3
            params["motion_direction"] = str(rng.choice(("horizontal", "vertical", "diagonal")))
            image = motion_blur(image, params["motion_kernel_width"], params["motion_direction"])
    elif degradation == "low_resolution":
        params["downsample_scale"] = float(rng.uniform(0.65, 0.8) if role == "small" else rng.uniform(0.5, 0.68))
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


def categorize(sample):
    text = sample["text"]
    length = ("single_word" if len(text.split()) == 1 else "short_line" if len(text) <= 30
              else "medium_line" if len(text) <= 55 else "long_line")
    effect = sample["degradation"]
    blur = ("combined" if effect == "defocus_motion" else "defocus" if "defocus" in effect
            else "motion" if "motion" in effect else "none")
    has_digits, has_letters = any(c.isdigit() for c in text), any(c.isalpha() for c in text)
    return {
        "font_family": sample["family"], "font_style": sample["style"],
        "text_size": {"heading": "large", "body": "medium", "small": "small"}[sample["role"]],
        "text_role": sample["role"], "capitalization": sample["text_case"],
        "decoration": "underlined" if sample["underline"] else "plain",
        "baseline": sample["curve"], "geometry": sample["geometry"],
        "background": sample["background"],
        "lighting": "uneven" if sample["background"] in ("uneven_light", "photocopy") else "even",
        "contrast": sample["contrast"], "blur": blur,
        "blur_strength": "strong" if effect in ("strong_defocus", "strong_motion", "defocus_motion")
                         else "standard" if blur != "none" else "none",
        "degradation": effect, "content_type": sample["content_type"], "text_length": length,
        "character_content": "mixed" if has_digits and has_letters else "numeric" if has_digits else "letters",
    }


def category_counts(samples):
    return {key: dict(Counter(sample["categories"][key] for sample in samples))
            for key in samples[0]["categories"]}


def write_categories(output, samples):
    with (output / "categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file_name", "text", "character_count", "word_count",
                                                   *samples[0]["categories"]))
        writer.writeheader()
        for sample in samples:
            writer.writerow({key: sample[key] for key in ("file_name", "text", "character_count", "word_count")}
                            | sample["categories"])
    (output / "category-counts.json").write_text(
        json.dumps(category_counts(samples), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contact_sheet(root, samples, filename="contact-sheet.jpg", index_offset=0):
    # Keep preview memory bounded when generating thousands of training lines.
    samples = samples[:40]
    rows = (len(samples) + 1) // 2
    cell_width, cell_height = 880, 154
    sheet = Image.new("RGB", (cell_width * 2, rows * cell_height), (239, 242, 246))
    draw = ImageDraw.Draw(sheet)
    ui_font = ImageFont.truetype(samples[0]["font_path"], 15)
    for idx, sample in enumerate(samples):
        x, y = (idx // rows) * cell_width + 12, (idx % rows) * cell_height + 8
        decoration = " / underline" if sample["underline"] else ""
        title = (f"{idx + index_offset + 1:03d}  {sample['family']} / {sample['style']} / "
                 f"{sample['role']} {sample['source_font_size']}px{decoration}")
        draw.text((x, y), title, font=ui_font, fill=(42, 47, 57))
        effects = f"{sample['curve']} / {sample['degradation'].replace('_', ' ')} / {sample['background'].replace('_', ' ')}"
        draw.text((x, y + 20), effects, font=ui_font, fill=(75, 80, 90))
        capture = f"{sample['geometry']} / {sample['contrast']} / {sample['content_type']} / {sample['categories']['text_length']}"
        draw.text((x, y + 39), capture, font=ui_font, fill=(75, 80, 90))
        with Image.open(root / sample["file_name"]) as image:
            if image.width > cell_width - 28:
                image = image.resize((cell_width - 28, round(image.height * (cell_width - 28) / image.width)))
            sheet.paste(image, (x, y + 62))
    sheet.save(root / filename, quality=95)


def previews(root, samples):
    if len(samples) <= 40:
        contact_sheet(root, samples)
    else:
        # Show up to 100 examples in readable 20-image pages; stay bounded for
        # full training datasets with tens of thousands of images.
        for start in range(0, min(len(samples), 100), 20):
            filename = f"contact-sheet-{start // 20 + 1:02d}.jpg"
            contact_sheet(root, samples[start:start + 20], filename, start)
        shutil.copy2(root / "contact-sheet-01.jpg", root / "contact-sheet.jpg")


def generate(corpus, output, font_root, count=20, seed=42, font_manifest=None,
             curve_mode="mixed", blur_profile="standard", capture_profile="none",
             content_profile="wikipedia", progress_every=0):
    if count < 1:
        raise ValueError("count must be positive")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    if (curve_mode not in ("mixed", "straight") or blur_profile not in ("standard", "strong", "balanced")
            or capture_profile not in ("none", "mixed") or content_profile not in ("wikipedia", "mixed")):
        raise ValueError("Unknown generation profile")
    text = unicodedata.normalize("NFC", corpus.read_text(encoding="utf-8"))
    words = text.split()
    if not words:
        raise ValueError("Corpus is empty")
    fonts = load_font_manifest(font_manifest) if font_manifest else discover_fonts(font_root)
    families = {}
    for font in fonts:
        families.setdefault(font.family, []).append(font)
    family_groups = list(families.values())
    rng = np.random.default_rng(seed)
    curve_types = CURVES if curve_mode == "mixed" else ("straight",)
    curves = balanced_choices(curve_types, count, rng)
    if blur_profile in ("strong", "balanced"):
        effects = balanced_choices(STRONG_DEGRADATIONS if blur_profile == "strong" else BALANCED_DEGRADATIONS, count, rng)
    geometries, contrasts = ["none"] * count, ["normal"] * count
    if capture_profile == "mixed":
        geometries = balanced_choices(CAPTURE_GEOMETRIES, count, rng)
        contrasts = balanced_choices(CONTRASTS, count, rng)
        if blur_profile == "balanced":
            effects = balance_capture_effects(effects, contrasts)
    contents = ["wikipedia"] * count
    if content_profile == "mixed":
        contents = balanced_choices(CONTENT_TYPES, count, rng)
        wiki_lengths = iter(balanced_choices(tuple(TEXT_LENGTHS), sum(contents == "wikipedia"), rng))
        # Independent assignment lets every actual font style appear with each
        # size/case, instead of a fixed association between two cyclic schedules.
        typography_profiles = balanced_choices(tuple(range(len(TYPOGRAPHIES))), count, rng)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    samples, labels = [], []
    started = time.perf_counter()
    for idx in range(count):
        if font_manifest:
            # Equal family representation; a four-style family must not swamp
            # distinctive families such as Impact or Comic Sans with fewer files.
            group = family_groups[idx % len(family_groups)]
            font = group[(idx // len(family_groups)) % len(group)]
        else:
            font = fonts[idx % len(fonts)]
        profile = (idx % len(family_groups) + idx // len(family_groups)) % len(TYPOGRAPHIES)
        if content_profile == "mixed":
            profile = int(typography_profiles[idx])
        role, text_case, underline, sizes, lengths = TYPOGRAPHIES[profile]
        content = str(contents[idx])
        if content != "wikipedia":
            label = structured_text(content, font, rng, text_case)
        elif content_profile == "mixed":
            length_kind = str(next(wiki_lengths))
            limits = TEXT_LENGTHS[length_kind]
            label = sample_text(words, font, rng, text_case, limits, limits[0], length_kind == "single_word")
        else:
            label = sample_text(words, font, rng, text_case, lengths)
        # Balance the first 20; rotate assignments on later cycles so larger
        # datasets do not associate one font/style with just one background.
        cycle = idx // len(fonts)
        kind = BACKGROUNDS[(idx + cycle * 3) % len(BACKGROUNDS)]
        degradation = DEGRADATIONS[(idx + idx // 5 + cycle) % len(DEGRADATIONS)]
        if blur_profile in ("strong", "balanced"):
            degradation = str(effects[idx])
        font_size = int(rng.integers(sizes[0], sizes[1] + 1))
        curve = str(curves[idx])
        strength = float(rng.uniform(0.35, 0.7)) if curve != "straight" else 0
        image, params = render_line(label, font, font_size, kind, degradation, rng, underline, role, curve, strength,
                                    str(geometries[idx]), str(contrasts[idx]))
        relative = f"images/line_{idx:05d}.jpg"
        image.save(output / relative, quality=params["jpeg_quality"])
        labels.append(f"{relative}\t{label}")
        sample = {"file_name": relative, "text": label, "role": role, "text_case": text_case,
                        "content_type": content, "character_count": len(label), "word_count": len(label.split()),
                        "family": font.family,
                        "style": font.style, "font_path": str(font.path),
                        "font_sha256": font.sha256,
                        "font_license": font.license, "font_source": font.source,
                        "width": image.width, "height": image.height,
                        "image_sha256": hashlib.sha256((output / relative).read_bytes()).hexdigest(), **params}
        sample["categories"] = categorize(sample)
        samples.append(sample)
        if progress_every and (idx + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"Generated {idx + 1}/{count} ({(idx + 1) / elapsed:.1f} images/s)", flush=True)
    (output / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    characters = sorted((set("".join(s["text"] for s in samples)) | set(TURKISH)) - {" "})
    (output / "turkish_dict.txt").write_text("\n".join(characters) + "\n", encoding="utf-8")
    metadata = {"schema_version": 2, "seed": seed, "count": count,
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "curve_mode": curve_mode, "blur_profile": blur_profile,
                "capture_profile": capture_profile, "content_profile": content_profile,
                "categories": category_counts(samples),
                "curves": dict(Counter(s["curve"] for s in samples)),
                "families": dict(Counter(s["family"] for s in samples)),
                "styles": dict(Counter(s["style"] for s in samples)),
                "backgrounds": dict(Counter(s["background"] for s in samples)),
                "degradations": dict(Counter(s["degradation"] for s in samples)), "samples": samples}
    metadata["typography"] = {
        "roles": dict(Counter(s["role"] for s in samples)),
        "text_cases": dict(Counter(s["text_case"] for s in samples)),
        "underlined": sum(s["underline"] for s in samples),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_categories(output, samples)
    manifest = corpus.with_suffix(".sources.json")
    if manifest.exists():
        shutil.copy2(manifest, output / "trwiki.sources.json")
    license_dir = output / "licenses"
    license_dir.mkdir()
    for family, package in (("dejavu", "fonts-dejavu-core"), ("liberation", "fonts-liberation")):
        notice = Path("/usr/share/doc") / package / "copyright"
        if notice.exists():
            shutil.copy2(notice, license_dir / (family + ".txt"))
    for notice in {font.notice for font in fonts if font.notice}:
        # Preserve notices even when different packages use the same basename.
        digest = hashlib.sha256(notice.read_bytes()).hexdigest()[:12]
        shutil.copy2(notice, license_dir / (digest + "-" + notice.name))
    if font_manifest:
        shutil.copy2(font_manifest, output / "font-manifest.json")
    (output / "README.txt").write_text(
        "PaddleOCR recognition line crops, height 48, variable width.\n"
        "labels.txt has relative image path, TAB, exact UTF-8 text. Set data_dir to this directory.\n"
        "Use turkish_dict.txt with use_space_char: true; this dictionary covers the samples, not all future corpora.\n"
        "Use a compatible recognition head/dictionary when fine-tuning pretrained weights.\n"
        f"{len(families)} real font families, {len(fonts)} font files, ten procedural backgrounds "
        "and configurable degradation types. See metadata.json for all parameters and hashes.\n"
        f"Curve mode: {curve_mode}; blur profile: {blur_profile}. Arches, bowls and waves warp the complete "
        "text/underline patch on an expanded canvas. Strong blur adds defocus, directional motion and combined blur, "
        "scaled to text height. All labels preserve the unwarped reading order.\n"
        f"Capture profile: {capture_profile}; content profile: {content_profile}. Mixed capture adds "
        "skew, perspective, low contrast and faint printing. Mixed content includes Wikipedia words/lines "
        "of different lengths, invented dates/prices and synthetic document IDs (not personal IDs).\n"
        "The balanced blur preset reserves 20% for strong effects. With mixed capture, faint printing "
        "uses clean rendering and low-contrast text avoids strong blur; preset counts stay unchanged.\n"
        "categories.csv provides one row per image with its transcription and category labels; "
        "category-counts.json summarizes coverage. Geometry parameters and ink opacity are in metadata.json.\n"
        "Typography includes large uppercase headings (48-64px source fonts), body text (26-38px), "
        "small text (18-22px), and underlines below glyph descenders. All crops are resized to height 48; "
        "relative text size is preserved and recorded. Turkish casing maps i to İ and ı to I. "
        "Labels include the rendered case but omit decorative underline strokes.\n"
        "Font families and styles come from actual font metadata. License notices: licenses/. "
        "Font binaries are not included. See licenses/ and system package copyright notices.\n"
        "For trwiki.txt: Turkish Wikipedia contributors, CC BY-SA 4.0, "
        "https://creativecommons.org/licenses/by-sa/4.0/. Revision attribution: trwiki.sources.json.\n"
        "Transformations: whole-word selection or authored field templates, text rendering, procedural backgrounds, "
        "curved baselines, perspective, skew, opacity, rotation, scaling, "
        "blur/downsampling/noise and JPEG compression. This sample set is not evidence of OCR accuracy.\n",
        encoding="utf-8",
    )
    previews(output, samples)
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
        if sample.get("text_case") == "uppercase" and label != turkish_upper(label):
            raise ValueError("Uppercase label does not match heading typography")
        if metadata.get("schema_version", 1) >= 2:
            if (sample["character_count"] != len(label) or sample["word_count"] != len(label.split())
                    or sample["categories"] != categorize(sample)):
                raise ValueError("Incorrect sample categories or text length")
            if metadata["blur_profile"] == "balanced" and metadata["capture_profile"] == "mixed":
                if ((sample["contrast"] == "faint_print" and sample["degradation"] != "clean")
                        or (sample["contrast"] == "low_contrast" and sample["categories"]["blur_strength"] == "strong")):
                    raise ValueError("Unreadable capture/degradation combination")
        paths.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sample["image_sha256"]:
            raise ValueError("Image checksum mismatch")
        with Image.open(path) as image:
            if image.size != (sample["width"], 48):
                raise ValueError("Wrong image dimensions")
            image.verify()
    if paths != set((root / "images").glob("*.jpg")):
        raise ValueError("Missing or unlabelled images")
    if metadata.get("schema_version", 1) >= 2:
        counts = category_counts(metadata["samples"])
        if counts != metadata["categories"] or counts != json.loads((root / "category-counts.json").read_text(encoding="utf-8")):
            raise ValueError("Incorrect category counts")
        with (root / "categories.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != metadata["count"]:
            raise ValueError("Wrong category row count")
        for row, sample in zip(rows, metadata["samples"]):
            expected = {key: str(sample[key]) for key in ("file_name", "text", "character_count", "word_count")}
            if row != expected | sample["categories"]:
                raise ValueError("Incorrect CSV categories")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("resources/corpus/trwiki.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/PaddleOCR_tr_recognition"))
    parser.add_argument("--font-root", type=Path, default=Path("/usr/share/fonts/truetype"))
    parser.add_argument("--font-manifest", type=Path, help="JSON catalog of additional licensed fonts and notices")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curve-mode", choices=("mixed", "straight"), default="mixed")
    parser.add_argument("--blur-profile", choices=("standard", "strong", "balanced"), default="standard")
    parser.add_argument("--capture-profile", choices=("none", "mixed"), default="none")
    parser.add_argument("--content-profile", choices=("wikipedia", "mixed"), default="wikipedia")
    parser.add_argument("--progress-every", type=int, default=250, help="Log progress every N images; 0 disables it")
    args = parser.parse_args()
    result = generate(args.corpus, args.output, args.font_root, args.count, args.seed, args.font_manifest,
                      args.curve_mode, args.blur_profile, args.capture_profile, args.content_profile, args.progress_every)
    print(f"Validated {result['count']} UTF-8 image/label pairs in {args.output}")
    for key in ("families", "styles", "backgrounds", "degradations", "curves", "typography"):
        print(key + ": " + json.dumps(result[key], ensure_ascii=False))
    for key in ("geometry", "contrast", "content_type", "text_length"):
        print(key + ": " + json.dumps(result["categories"][key], ensure_ascii=False))


if __name__ == "__main__":
    main()
