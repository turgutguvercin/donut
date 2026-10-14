#!/usr/bin/env python3
"""Deterministic, quota-controlled Turkish recognition dataset, with audit.

See config_tr_recognition_100k.json for explicit weighted marginal quotas.
Fonts are balanced by family, then by real available styles within each family.
The older document and recognition generators remain usable independently.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import generate_tr_recognition as base

ROOT = Path(__file__).resolve().parents[1]
CLIPS = (*base.CLIPPING_ERRORS, *base.HORIZONTAL_ERRORS)
WORKER = {}


def canonical(text):
    return unicodedata.normalize("NFC", " ".join(text.translate(str.maketrans({"İ": "i", "I": "ı"})).lower().split()))


def text_split(text):
    # Stable across process counts, sample seeds, case and visual augmentations.
    number = int(hashlib.sha256(("tr-ocr-group-v1:" + canonical(text)).encode()).hexdigest()[:12], 16) % 100
    return "train" if number < 90 else "val" if number < 95 else "test"


def quotas(weights, count):
    total = sum(weights.values())
    raw = {key: value * count / total for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda k: -(raw[k] - result[k]))[:count - sum(result.values())]:
        result[key] += 1
    return result


def choices(weights, count, rng):
    values = [key for key, amount in quotas(weights, count).items() for _ in range(amount)]
    rng.shuffle(values)
    return values


def require_values(rows, axis, required, rng):
    """Satisfy explicit constraints by permutation; never change a marginal quota."""
    donors = defaultdict(list)
    for i, row in enumerate(rows):
        if i not in required:
            donors[row[axis]].append(i)
    for indices in donors.values():
        rng.shuffle(indices)
    for i, value in required.items():
        if rows[i][axis] == value:
            continue
        if not donors[value]:
            raise ValueError(f"Infeasible constraint for {axis}={value}")
        j = donors[value].pop()
        previous = rows[i][axis]
        rows[i][axis], rows[j][axis] = value, previous
        donors[previous].append(j)


def plan_samples(config, fonts, count, seed):
    rng = np.random.default_rng(seed)
    family_fonts = defaultdict(list)
    for i, font in enumerate(fonts):
        family_fonts[font.family].append(i)
    style_offset = Counter()
    rows = []
    for split, split_count in quotas(config["splits"], count).items():
        family_schedule = choices(dict.fromkeys(family_fonts, 1), split_count, rng)
        block = [{"split": split, "family": family} for family in family_schedule]
        for row in block:
            family = row["family"]
            indices = family_fonts[family]
            row["font_index"] = indices[style_offset[family] % len(indices)]
            row["style"] = fonts[row["font_index"]].style
            style_offset[family] += 1
        for axis, weights in config["axes"].items():
            for row, value in zip(block, choices(weights, split_count, rng)):
                row[axis] = value
        # Case and confusable groups have their own exact conditional quotas.
        regular = [r for r in block if r["content_type"] != "confusables"]
        for row, case in zip(regular, choices({"original": 1, "uppercase": 1}, len(regular), rng)):
            row["text_case"] = case
            row["confusable_group"] = "none"
        confusables = [r for r in block if r["content_type"] == "confusables"]
        # Continue the cycle across splits so the GLOBAL count is exactly equal.
        names = list(config["confusable_groups"])
        start = sum(r["content_type"] == "confusables" for r in rows)
        groups = [names[(start + i) % len(names)] for i in range(len(confusables))]
        rng.shuffle(groups)
        for row, group in zip(confusables, groups):
            row["text_case"], row["confusable_group"] = "mixed", group
        # Five percent uncorrupted reference crops, selected across content types.
        controls = set()
        for kind in config["axes"]["content_type"]:
            indices = [i for i, row in enumerate(block) if row["content_type"] == kind]
            rng.shuffle(indices)
            controls.update(indices[:round(len(indices) * config["clean_control_percent"] / 100)])
        clean_axes = {"curve": "straight", "geometry": "none", "background": "white",
                      "lighting_effect": "none", "contrast": "normal", "degradation": "clean",
                      "crop_error": "none", "spacing": "normal", "printing": "none",
                      "occlusion": "none", "mixed_typography": "uniform", "underline": "plain"}
        for axis, value in clean_axes.items():
            require_values(block, axis, dict.fromkeys(controls, value), rng)
        severe = set()
        for error in CLIPS:
            indices = [i for i, row in enumerate(block) if row["crop_error"] == error]
            rng.shuffle(indices)
            severe.update(indices[:round(len(indices) * config["severe_fraction_of_clipping"])])
        # Avoid stacking other severe corruptions on already ambiguous crops.
        for axis, value in {"contrast": "normal", "degradation": "clean", "occlusion": "none",
                            "printing": "none", "spacing": "normal"}.items():
            require_values(block, axis, dict.fromkeys(controls | severe, value), rng)
        faint = {i for i, row in enumerate(block) if row["contrast"] == "faint_print"}
        require_values(block, "degradation", dict.fromkeys(controls | severe | faint, "clean"), rng)
        require_values(block, "occlusion", dict.fromkeys(controls | severe | faint, "none"), rng)
        require_values(block, "printing", dict.fromkeys(controls | severe | faint, "none"), rng)
        # Low contrast uses ordinary, rather than strong, blur.
        protected = controls | severe | faint
        donors = [i for i, r in enumerate(block) if i not in protected and r["contrast"] == "normal"
                  and r["degradation"] not in base.STRONG_BLURS]
        rng.shuffle(donors)
        for i, row in enumerate(block):
            if row["contrast"] == "low_contrast" and row["degradation"] in base.STRONG_BLURS:
                if not donors:
                    raise ValueError("Insufficient normal-contrast blur donors")
                j = donors.pop()
                row["degradation"], block[j]["degradation"] = block[j]["degradation"], row["degradation"]
        for i, row in enumerate(block):
            row["clean_control"] = i in controls
            row["clipping_severity"] = "severe" if i in severe else "mild" if row["crop_error"] in CLIPS else "none"
        rows.extend(block)
    # Keep a stable index and independent RNG for each image; worker count is irrelevant.
    for i, row in enumerate(rows):
        row["index"] = i
        row["seed"] = int(np.random.SeedSequence([seed, i]).generate_state(1)[0])
    return rows


def safe_text(text):
    return (text and len(text) <= 254 and "\\" not in text and "\ufffd" not in text
            and all(not unicodedata.category(c).startswith("C") for c in text)
            and all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in text))


def make_pools(corpus):
    corpus = unicodedata.normalize("NFC", " ".join(corpus.split()))
    words = sorted(set(re.findall(r"\b[A-Za-zÇĞİÖŞÜçğıöşü]{3,20}\b", corpus)))
    pools = {"ascii_word": [w for w in words if w.isascii()],
             "turkish_word": [w for w in words if not w.isascii()]}
    words_in_order = corpus.split()
    for kind, low, high in (("short_line", 15, 40), ("medium_line", 41, 90)):
        phrases = set()
        for start in range(len(words_in_order)):
            phrase = ""
            for word in words_in_order[start:start + 20]:
                phrase = (phrase + " " + word).strip()
                if len(phrase) > high:
                    break
                if len(phrase) >= low and safe_text(phrase):
                    phrases.add(phrase)
                    break
        pools[kind] = sorted(phrases)
    pools.update(base.sentence_pools(corpus))
    result = {}
    for kind, texts in pools.items():
        for split in ("train", "val", "test"):
            selected = [t for t in texts if safe_text(t) and text_split(t) == split]
            if not selected:
                raise ValueError(f"Empty {split} pool for {kind}")
            result[kind, split] = selected
    return result


@lru_cache(maxsize=512)
def pil_font(path, size):
    # NFC Turkish uses Latin glyphs; BASIC avoids repeated complex-script
    # shaping on each run while retaining the real glyphs and kerning.
    return ImageFont.truetype(path, size=size, layout_engine=ImageFont.Layout.BASIC)


def rich_patch(text, primary, size, ink, underline, row, rng, fonts):
    """Render actual font files, preserving glyph bounds and per-run provenance."""
    mode = row["mixed_typography"]
    chunks = re.findall(r"\S+|\s+", text)
    # A one-word sample still needs genuine within-word typographic changes.
    if len(chunks) == 1 and mode in ("mixed_fonts", "mixed_styles", "mixed_sizes"):
        middle = max(1, len(text) // 2)
        chunks = [text[:middle], text[middle:]]
    family_candidates = [f for f in fonts if f.family != primary.family and set(map(ord, text)) <= f.codepoints]
    style_candidates = [f for f in fonts if f.family == primary.family and f.style != primary.style
                        and set(map(ord, text)) <= f.codepoints]
    if not style_candidates:
        style_candidates = [f for f in family_candidates if f.style != primary.style]
    secondary = primary
    if mode == "mixed_fonts":
        secondary = family_candidates[int(rng.integers(len(family_candidates)))]
    elif mode == "mixed_styles":
        secondary = style_candidates[int(rng.integers(len(style_candidates)))]
    operations, runs = [], []
    x, offset, word_number = 0.0, 0, 0
    tracking = {"normal": 0, "tight": -0.045, "wide": 0.13,
                "irregular_word_gaps": 0, "touching": -0.14}[row["spacing"]] * size
    for chunk in chunks:
        chosen = secondary if word_number % 2 else primary
        run_size = max(10, round(size * (0.78 if word_number % 2 else 1.12))) if mode == "mixed_sizes" else size
        baseline = 0
        if mode in ("superscript", "subscript") and offset + len(chunk) == len(text):
            # Last token is an explicit, transcribed footnote/index digit.
            run_size = max(10, round(size * 0.65))
            baseline = round(size * (-0.38 if mode == "superscript" else 0.22))
        face = pil_font(str(chosen.path), run_size)
        start_x = x
        if chunk.isspace():
            gap = float(rng.uniform(0.4, 2.4)) if row["spacing"] == "irregular_word_gaps" else 1
            x += float(face.getlength(chunk)) * gap
        elif tracking:
            for char in chunk:
                operations.append((x, baseline, char, face))
                x += max(1.0, float(face.getlength(char)) + tracking)
        else:
            operations.append((x, baseline, chunk, face))
            x += float(face.getlength(chunk))
        if not chunk.isspace():
            runs.append({"start": offset, "end": offset + len(chunk), "family": chosen.family,
                         "style": chosen.style, "font_sha256": chosen.sha256,
                         "size": run_size, "baseline_offset": baseline, "advance": round(x - start_x, 3)})
            word_number += 1
        offset += len(chunk)
    boxes = [(ox + b[0], oy + b[1], ox + b[2], oy + b[3])
             for ox, oy, string, face in operations for b in [face.getbbox(string, anchor="ls")]]
    left, top = math.floor(min(b[0] for b in boxes)), math.floor(min(b[1] for b in boxes))
    right, bottom = math.ceil(max(b[2] for b in boxes)), math.ceil(max(b[3] for b in boxes))
    alpha = Image.new("L", (right - left + 4, bottom - top + 4))
    draw = ImageDraw.Draw(alpha)
    for ox, oy, string, face in operations:
        draw.text((ox - left + 2, oy - top + 2), string, font=face, anchor="ls", fill=255)
    alpha = alpha.crop(alpha.getbbox())
    original_ink = int((np.asarray(alpha) >= 64).sum())
    print_mode = row["printing"]
    if print_mode == "ink_bleed":
        alpha = alpha.filter(ImageFilter.MaxFilter(3))
    elif print_mode in ("missing_ink", "broken_strokes", "dot_matrix"):
        pixels = np.array(alpha)
        if print_mode == "missing_ink":
            pixels[rng.random(pixels.shape) < 0.06] = 0
        elif print_mode == "broken_strokes":
            for _ in range(max(1, alpha.width // 18)):
                px, py = int(rng.integers(alpha.width)), int(rng.integers(alpha.height))
                pixels[py:py + 1, px:px + max(2, size // 8)] = 0
        else:
            # A 3x3 dot grid, softened rather than deleting entire thin letters.
            yy, xx = np.indices(pixels.shape)
            gaps = (xx % 3 == 2) | (yy % 3 == 2)
            pixels[gaps] = np.rint(pixels[gaps] * 0.28).astype("uint8")
        alpha = Image.fromarray(pixels)
    print_ink = int((np.asarray(alpha) >= 64).sum())
    height = alpha.height
    gap, thickness = (max(2, round(size * .06)), max(1, round(size * .04))) if underline else (0, 0)
    patch = Image.new("RGBA", (alpha.width, alpha.height + gap + thickness), (*ink, 0))
    full_alpha = Image.new("L", patch.size)
    full_alpha.paste(alpha, (0, 0))
    if underline:
        ImageDraw.Draw(full_alpha).rectangle((0, height + gap, alpha.width - 1, patch.height - 1), fill=255)
    patch.putalpha(full_alpha)
    return patch, {"underline": underline, "underline_gap_native": gap,
                   "underline_thickness_native": thickness, "text_height_native": height,
                   "font_runs": runs, "tracking_native": tracking,
                   "printing_ink_ratio": print_ink / max(1, original_ink)}


def postprocess(image, glyph_mask, rng, row, config):
    """Final-size, measured occlusions and illumination, before blur/compression."""
    width, height = image.size
    scale = min(1, config["max_width"] / width)
    if scale < 1:
        scaled = (config["max_width"], max(1, round(height * scale)))
        canvas = Image.new("RGB", (scaled[0], height), image.getpixel((0, 0)))
        y = (height - scaled[1]) // 2
        canvas.paste(image.resize(scaled, Image.Resampling.LANCZOS), (0, y))
        mask_canvas = Image.new("L", canvas.size)
        mask_canvas.paste(glyph_mask.resize(scaled, Image.Resampling.LANCZOS), (0, y))
        image, glyph_mask = canvas, mask_canvas
    width, height = image.size
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    kind = row["occlusion"]
    center = int(rng.integers(max(1, width // 5), max(2, 4 * width // 5)))
    extent = max(8, min(90, width // 3))
    if kind == "stamp":
        draw.ellipse((center - extent // 2, 5, center + extent // 2, 42), outline=(173, 35, 47, 155), width=2)
        draw.line((center - extent // 3, 24, center + extent // 3, 24), fill=(173, 35, 47, 135), width=2)
    elif kind == "signature":
        points = [(center - extent // 2 + k * extent // 12, int(rng.integers(14, 38))) for k in range(13)]
        draw.line(points, fill=(25, 61, 116, 145), width=1)
    elif kind == "fold":
        draw.line((center - 5, 0, center + 5, height), fill=(80, 75, 70, 75), width=3)
        draw.line((center - 2, 0, center + 8, height), fill=(255, 255, 255, 125), width=2)
    elif kind == "strikethrough":
        draw.line((2, height // 2, width - 3, height // 2 + int(rng.integers(-3, 4))),
                  fill=(65, 50, 52, 170), width=1)
    mask = np.asarray(glyph_mask) >= 64
    coverage = float(((np.asarray(overlay.getchannel("A")) > 0) & mask).sum() / max(1, mask.sum()))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    lighting = row["lighting_effect"]
    if lighting != "none":
        xx = np.linspace(0, 1, width)
        if lighting == "shadow":
            profile = 1 - .25 * np.exp(-((xx - float(rng.uniform(.2, .8))) / .18) ** 2)
        else:
            profile = np.linspace(.73, 1, width)
            if rng.integers(2):
                profile = profile[::-1]
        image = Image.fromarray(np.clip(np.asarray(image) * profile[None, :, None], 0, 255).astype("uint8"))
    bbox = glyph_mask.getbbox()
    return image, {"width_cap_scale": scale, "glyph_bounds_output": list(bbox) if bbox else None,
                   "occluded_ink_fraction": coverage, "occlusion_is_transcribed": False}


def initialize(corpus, font_manifest, config, output):
    # Native BLAS/OpenCV threading is set in the launch command as well.
    import cv2
    cv2.setNumThreads(1)
    WORKER.update(fonts=base.load_font_manifest(Path(font_manifest)), config=config,
                  pools=make_pools(Path(corpus).read_text(encoding="utf-8")), output=Path(output))


def select_label(row, font, rng):
    config, pools = WORKER["config"], WORKER["pools"]
    kind, split = row["content_type"], row["split"]
    confusable_span = None
    for _ in range(3000):
        if kind in ("date", "price", "synthetic_id"):
            source = base.structured_text(kind, font, rng, "original")
        elif kind == "confusables":
            group = config["confusable_groups"][row["confusable_group"]]
            chars = list(group)
            rng.shuffle(chars)
            span = " ".join(chars)
            # Neutral synthetic suffix ensures many labels without privileging a group member.
            alphabet = [c for c in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" if c not in group]
            suffix = "".join(rng.choice(alphabet, size=8))
            source = span + " / " + suffix
            confusable_span = [0, len(span)]
        else:
            pool = pools[kind, split]
            source = pool[int(rng.integers(len(pool)))]
        text = base.turkish_upper(source) if row["text_case"] == "uppercase" else source
        if kind == "ascii_word" and not text.isascii():
            continue
        # Require both source and final case to belong to this group split.
        if text_split(source) != split or text_split(text) != split:
            continue
        if row["mixed_typography"] in ("superscript", "subscript"):
            text += " " + str(int(rng.integers(2, 10)))
        if safe_text(text) and set(map(ord, text)) <= font.codepoints:
            return text, source, confusable_span
    raise ValueError(f"No label for {row}")


def generate_one(row):
    rng = np.random.default_rng(row["seed"])
    fonts, config, output = WORKER["fonts"], WORKER["config"], WORKER["output"]
    font = fonts[row["font_index"]]
    text, source, span = select_label(row, font, rng)
    # Context is selected from the SAME split and remains excluded from the target label.
    context_pool = WORKER["pools"]["medium_line", row["split"]]
    context = []
    while len(context) < 2:
        candidate = context_pool[int(rng.integers(len(context_pool)))]
        if set(map(ord, candidate)) <= font.codepoints:
            context.append(candidate)
    sizes = {"small": (18, 23), "body": (27, 39), "heading": (48, 65)}
    size = int(rng.integers(*sizes[row["role"]]))
    def renderer(label, face, pixels, ink, underline):
        return rich_patch(label, face, pixels, ink, underline, row, rng, fonts)
    def processor(image, mask, generator):
        return postprocess(image, mask, generator, row, config)
    image, params = base.render_line(
        text, font, size, row["background"], row["degradation"], rng,
        underline=row["underline"] == "underlined", role=row["role"],
        curve=row["curve"], curve_strength=float(rng.uniform(.25, .6)),
        geometry=row["geometry"], contrast=row["contrast"], crop_error=row["crop_error"],
        context_texts=context, clipping_severity="severe" if row["clipping_severity"] == "severe" else "mild",
        patch_renderer=renderer, image_processor=processor,
        rotation_jitter=False, compact_scene=True)
    # A role called heading always retains its assigned case in metadata; headings
    # are not all uppercase, allowing the factors to vary independently.
    relative = f"images/{row['index'] // 1000:03d}/line_{row['index']:06d}.jpg"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=params["jpeg_quality"], subsampling=0)
    bucket = next(w for w in config["width_buckets"] if w >= image.width)
    axes = {key: row[key] for key in config["axes"]}
    axes.update(font_family=font.family, font_style=font.style, text_case=row["text_case"],
                confusable_group=row["confusable_group"], clipping_severity=row["clipping_severity"])
    return {"file_name": relative, "text": text, "source_text": source,
            "source_group_sha256": hashlib.sha256(canonical(source).encode()).hexdigest(),
            "split": row["split"], "index": row["index"], "seed": row["seed"],
            "width": image.width, "height": image.height, "width_bucket": bucket,
            "character_count": len(text), "word_count": len(text.split()),
            "font_sha256": font.sha256, "confusable_span": span,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clean_control": row["clean_control"], "categories": axes, **params}


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def counts(rows, config):
    keys = [*config["axes"], "font_family", "font_style", "text_case", "confusable_group", "clipping_severity"]
    return {key: dict(Counter(row["categories"][key] for row in rows)) for key in keys}


def expected_counts(plan, config):
    categories = [{"categories": {**r, "font_family": r["family"], "font_style": r["style"]}} for r in plan]
    return counts(categories, config)


def save_notices(output, corpus, fonts, manifest):
    shutil.copy2(corpus.with_suffix(".sources.json"), output / "trwiki.sources.json")
    (output / "licenses").mkdir()
    for notice in sorted({f.notice for f in fonts}, key=str):
        if notice:
            digest = hashlib.sha256(notice.read_bytes()).hexdigest()[:12]
            shutil.copy2(notice, output / "licenses" / (digest + "-" + notice.name))
    shutil.copy2(manifest, output / "font-manifest.json")
    write_json(output / "font-provenance.json", [{"family": f.family, "style": f.style,
               "sha256": f.sha256, "license": f.license, "source": f.source,
               "codepoints": sorted(f.codepoints)} for f in fonts])


def make_previews(root, rows):
    # Include detail crops for long lines; a whole 4096-wide sentence is too tiny in a grid.
    display_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    targets = [("spacing", value) for value in ("normal", "tight", "wide", "irregular_word_gaps", "touching")]
    targets += [("printing", v) for v in ("broken_strokes", "missing_ink", "ink_bleed", "dot_matrix")]
    targets += [("occlusion", v) for v in ("stamp", "signature", "fold", "strikethrough")]
    targets += [("mixed_typography", v) for v in ("mixed_fonts", "mixed_styles", "mixed_sizes", "superscript", "subscript")]
    targets += [("crop_error", v) for v in CLIPS]
    targets += [("confusable_group", v) for v in WORKER.get("config", {}).get("confusable_groups", {})]
    targets += [("content_type", v) for v in ("long_sentence", "extra_long_sentence")]
    selected = []
    for key, value in targets:
        candidates = [r for r in rows if r["categories"][key] == value]
        if candidates:
            candidates.sort(key=lambda r: (r["categories"]["degradation"] != "clean", r["width"] > 1000))
            selected.append((key, value, candidates[0]))
    for page in range(math.ceil(len(selected) / 12)):
        sheet = Image.new("RGB", (1600, 12 * 138), "#eeeeea")
        draw = ImageDraw.Draw(sheet)
        for i, (key, value, row) in enumerate(selected[page * 12:(page + 1) * 12]):
            y = i * 138
            draw.text((12, y + 4), f"{key}={value} | {row['categories']['font_family']} | {row['file_name']}", font=display_font, fill="black")
            with Image.open(root / row["file_name"]) as image:
                detail = image.crop((0, 0, min(image.width, 1560), 48))
                sheet.paste(detail, (12, y + 29))
            label = row["text"][:135] + ("…" if len(row["text"]) > 135 else "")
            draw.text((12, y + 83), label, font=display_font, fill="black")
            if row["width"] > 1560:
                draw.text((12, y + 108), "Preview shows the LEFT detail; the saved image contains the full line.", font=display_font, fill="#555555")
        sheet.save(root / f"preview-{page + 1:02d}.jpg", quality=91)


def finish(root, config, plan, rows, corpus, fonts, manifest, seed):
    actual, expected = counts(rows, config), expected_counts(plan, config)
    if actual != expected:
        raise ValueError("Actual category totals differ from the planned quotas")
    write_json(root / "category-counts.json", actual)
    write_json(root / "planned-counts.json", expected)
    chars = sorted((set("".join(r["text"] for r in rows)) | set(base.TURKISH)) - {" "})
    (root / "turkish_dict.txt").write_text("\n".join(chars) + "\n", encoding="utf-8")
    with (root / "categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "split", "text", *actual])
        writer.writeheader()
        for row in rows:
            writer.writerow({"file_name": row["file_name"], "split": row["split"], "text": row["text"], **row["categories"]})
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        (root / f"{split}.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in selected), encoding="utf-8")
        for width in config["width_buckets"]:
            (root / f"{split}-w{width}.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in selected if r["width_bucket"] == width), encoding="utf-8")
        (root / f"{split}-without-severe-clips.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in selected if not r["potentially_ambiguous"]), encoding="utf-8")
    save_notices(root, corpus, fonts, manifest)
    pair_axes = [("font_family", "content_type"), ("font_family", "crop_error"), ("font_family", "confusable_group"),
                 ("content_type", "crop_error"), ("content_type", "degradation"), ("crop_error", "clipping_severity"),
                 ("contrast", "degradation"), ("printing", "occlusion"), ("curve", "geometry")]
    write_json(root / "pair-counts.json", {a + " × " + b: dict(Counter(r["categories"][a] + " | " + r["categories"][b] for r in rows)) for a, b in pair_axes})
    metadata = {"schema_version": 5, "count": len(rows), "seed": seed,
                "config": config, "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "generator_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(base.__file__))},
                "split_counts": dict(Counter(r["split"] for r in rows)), "font_family_count": len(set(f.family for f in fonts)),
                "font_file_count": len(fonts), "balance": "explicit weighted marginal quotas; styles equal within each family; not a full Cartesian product",
                "clean_control_count": sum(r["clean_control"] for r in rows),
                "constraints": ["5% clean controls have white background, plain decoration and no geometric/capture defects",
                                "severe clips use normal contrast, clean rendering, normal spacing and no print defect or occlusion",
                                "faint printing uses clean rendering without print defects or occlusion",
                                "low contrast excludes strong blur"],
                "provenance": "Turkish Wikipedia contributors, CC BY-SA 4.0; exact revisions in trwiki.sources.json. Synthetic fields and confusable templates are generated.",
                "license": "CC BY-SA 4.0 for this generated dataset; font binaries are excluded and font notices are in licenses/"}
    write_json(root / "metadata.json", metadata)
    WORKER["config"] = config
    make_previews(root, rows)


def audit(root):
    config_meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = config_meta["config"]
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
    dictionary = set((root / "turkish_dict.txt").read_text(encoding="utf-8").splitlines()) | {" "}
    expected = json.loads((root / "planned-counts.json").read_text(encoding="utf-8"))
    if len(rows) != config_meta["count"] or len(labels) != len(rows) or counts(rows, config) != expected:
        raise ValueError("Count/quota mismatch")
    font_points = {f["sha256"]: set(f["codepoints"]) for f in json.loads((root / "font-provenance.json").read_text(encoding="utf-8"))}
    paths, hashes, text_groups, source_groups = set(), set(), {}, {}
    duplicate_images, duplicate_texts = 0, 0
    confusable_characters = defaultdict(Counter)
    buckets = defaultdict(Counter)
    for row, record in zip(rows, labels):
        name, label = record.split("\t", 1)
        if name != row["file_name"] or label != row["text"] or name in paths:
            raise ValueError("Mismatched/duplicate path or label")
        if not safe_text(label) or unicodedata.normalize("NFC", label) != label or not set(label) <= dictionary:
            raise ValueError("Invalid UTF-8/NFC/dictionary text")
        if len(label) > config["max_text_length"]:
            raise ValueError("Label length exceeds PaddleOCR configuration")
        if row["categories"]["text_case"] == "uppercase" and base.turkish_upper(label) != label:
            raise ValueError("Wrong Turkish uppercase label")
        for run in row["font_runs"]:
            if not set(map(ord, label[run["start"]:run["end"]])) <= font_points[run["font_sha256"]]:
                raise ValueError("Missing glyph in actual rendered run")
        paths.add(name)
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != row["image_sha256"]:
            raise ValueError("Image checksum mismatch")
        duplicate_images += digest in hashes
        hashes.add(digest)
        with Image.open(root / name) as image:
            image.load()
            if image.mode != "RGB" or image.size != (row["width"], 48) or image.width > config["max_width"]:
                raise ValueError("Invalid recognition image dimensions/mode")
        if not row["width"] <= row["width_bucket"] <= config["max_width"]:
            raise ValueError("Bucket would squeeze the image")
        # CTC needs an extra timestep between repeated adjacent labels.
        timesteps_needed = len(label) + sum(a == b for a, b in zip(label, label[1:]))
        if row["width_bucket"] // 4 < timesteps_needed:
            raise ValueError("Insufficient CTC time steps at width stride four")
        for mapping, key in ((text_groups, canonical(label)), (source_groups, row["source_group_sha256"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Text group leaked between dataset splits")
        duplicate_texts += canonical(label) in text_groups
        text_groups[canonical(label)], source_groups[row["source_group_sha256"]] = row["split"], row["split"]
        severity = row["categories"]["clipping_severity"]
        ink_floor, component_floor = (.7, 0) if severity == "severe" else (.97, .5)
        if row["target_ink_retention"] < ink_floor - 1e-9 or row["minimum_component_retention"] < component_floor - 1e-9:
            raise ValueError("Excessive detector clipping")
        if row["potentially_ambiguous"] != (severity == "severe"):
            raise ValueError("Incorrect severe clipping flag")
        group = row["categories"]["confusable_group"]
        if group != "none":
            first, last = row["confusable_span"]
            observed = Counter(c for c in label[first:last] if c != " ")
            if observed != Counter(config["confusable_groups"][group]):
                raise ValueError("Confusable group members are not equally represented")
            confusable_characters[group].update(observed)
        buckets[row["split"]][str(row["width_bucket"])] += 1
    if len(list((root / "images").glob("*/*.jpg"))) != len(rows):
        raise ValueError("Unexpected image files")
    for split in ("train", "val", "test"):
        expected_lines = [r["file_name"] + "\t" + r["text"] for r in rows if r["split"] == split]
        if (root / f"{split}.txt").read_text(encoding="utf-8").splitlines() != expected_lines:
            raise ValueError("Split labels differ from metadata")
        bucket_lines = []
        for width in config["width_buckets"]:
            bucket_lines.extend((root / f"{split}-w{width}.txt").read_text(encoding="utf-8").splitlines())
        if sorted(bucket_lines) != sorted(expected_lines):
            raise ValueError("Width buckets lose or duplicate samples")
    report = {"passed": True, "images_decoded_and_hashed": len(rows), "labels_utf8_nfc_checked": len(rows),
              "planned_quotas_match": True, "source_and_label_groups_disjoint_between_splits": True,
              "glyph_coverage_checked_per_actual_font_run": True, "all_images_height": 48,
              "width_range": [min(r["width"] for r in rows), max(r["width"] for r in rows)],
              "character_count_range": [min(len(r["text"]) for r in rows), max(len(r["text"]) for r in rows)],
              "width_buckets_by_split": dict(buckets), "targeted_confusable_character_counts": dict(confusable_characters),
              "duplicate_image_hashes": duplicate_images, "repeated_canonical_labels_within_split": duplicate_texts,
              "unique_source_text_groups": len(source_groups), "unique_canonical_labels": len(text_groups),
              "severe_clips": sum(r["potentially_ambiguous"] for r in rows),
              "crops_with_completely_removed_components": sum(r["removed_components"] > 0 for r in rows),
              "width_capped_by_uniform_scaling": sum(r["width_cap_scale"] < 1 for r in rows),
              "maximum_occluded_ink_fraction": max(r["occluded_ink_fraction"] for r in rows),
              "limitations": ["Coverage and label checks do not measure OCR accuracy or guarantee human readability.",
                              "Severe crops retain source labels even when marks/letters are missing; use the filtered split files if desired.",
                              "Synthetic validation/test images are not a substitute for real detector crops."]}
    write_json(root / "audit.json", report)
    return report


def generate(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fonts = base.load_font_manifest(args.font_manifest)
    plan = plan_samples(config, fonts, args.count, args.seed)
    if args.output.exists():
        raise ValueError("Output already exists; choose a fresh directory")
    args.output.mkdir(parents=True)
    write_json(args.output / "recipe.json", config)
    write_json(args.output / "planned-counts.json", expected_counts(plan, config))
    save_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / "generation-arguments.json", save_args)
    rows, started = [], time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.corpus), str(args.font_manifest), config, str(args.output))) as pool:
        with (args.output / "metadata.jsonl").open("w", encoding="utf-8") as meta, (args.output / "labels.txt").open("w", encoding="utf-8") as labels:
            for i, row in enumerate(pool.map(generate_one, plan, chunksize=16), 1):
                meta.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                labels.write(row["file_name"] + "\t" + row["text"] + "\n")
                rows.append(row)
                if i % args.progress_every == 0 or i == args.count:
                    meta.flush()
                    labels.flush()
                    print(f"Generated {i}/{args.count}; {i / (time.monotonic() - started):.1f} images/s", flush=True)
    finish(args.output, config, plan, rows, args.corpus, fonts, args.font_manifest, args.seed)
    print(json.dumps(audit(args.output), ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config_tr_recognition_100k.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "resources/corpus/trwiki.txt")
    parser.add_argument("--font-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(audit(args.output), ensure_ascii=False, indent=2))
    elif args.count < 1000 or args.workers < 1 or args.progress_every < 1:
        parser.error("Use at least 1,000 samples for this quota recipe, and positive worker/progress counts")
    else:
        generate(args)


if __name__ == "__main__":
    main()
