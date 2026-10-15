#!/usr/bin/env python3
"""Expand a verified recognition release with original official-style crops.

Existing images/splits are retained. New images use a separate deterministic
seed and explicit quotas; ambiguous crops have separate recommended lists.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import generate_tr_dataset as dataset
import generate_tr_recognition as base
import official_tr_text as official

ROOT = Path(__file__).resolve().parents[1]
WORK = {}
EXTRA_AXES = ("official_kind", "scan_profile", "targeted_group")
SCAN_PROFILES = {
    "small_clear": ((14, 19), (0.0, 0.15), (88, 96)),
    "small_scan": ((10, 15), (0.15, 0.35), (65, 85)),
    "small_jpeg": ((9, 13), (0.1, 0.3), (42, 66)),
    "photocopy_scan": ((10, 15), (0.2, 0.4), (55, 78)),
}


def plan_samples(config, fonts, count, seed, offset=0):
    plan = dataset.plan_samples(config, fonts, count, seed)
    rng = np.random.default_rng([seed, 9017])
    by_family = defaultdict(list)
    for i, font in enumerate(fonts):
        by_family[font.family].append(i)
    weights = {family: config["font_family_weights"].get(family, 1) for family in by_family}
    style_offsets = Counter()
    for split in config["splits"]:
        block = [row for row in plan if row["split"] == split]
        for row, family in zip(block, dataset.choices(weights, len(block), rng)):
            indices = by_family[family]
            row["font_index"] = indices[style_offsets[family] % len(indices)]
            row["family"], row["style"] = family, fonts[row["font_index"]].style
            style_offsets[family] += 1
        controls = {i for i, row in enumerate(block) if row["clean_control"]}
        # Physical scan blur is applied once; do not stack it with independent
        # strong blur, faint ink, print damage or aggressive camera exposure.
        dataset.require_values(block, "scan_profile", dict.fromkeys(controls, "none"), rng)
        scans = {i for i, row in enumerate(block) if row["scan_profile"] != "none"}
        protected = controls | scans
        for axis, value in {"degradation": "clean", "contrast": "normal", "printing": "none",
                            "occlusion": "none", "exposure": "normal"}.items():
            dataset.require_values(block, axis, dict.fromkeys(protected, value), rng)
        difficult_light = {i for i, row in enumerate(block) if row["exposure"] in ("low_light", "overexposed")}
        dataset.require_values(block, "contrast", dict.fromkeys(protected | difficult_light, "normal"), rng)
        # Swapping the axes above can move a strong blur onto faint text. Keep
        # all difficult-contrast/camera samples protected, preserving quotas.
        protected |= {i for i, row in enumerate(block) if row["contrast"] != "normal"
                      or row["exposure"] in ("low_light", "overexposed")}
        for axis, value in {"degradation": "clean", "printing": "none", "occlusion": "none"}.items():
            dataset.require_values(block, axis, dict.fromkeys(protected, value), rng)
        for old_kind in ("confusables", "synthetic_id"):
            selected = [r for r in block if r["content_type"] == old_kind]
            groups = dataset.choices(dict.fromkeys(official.GROUPS, 1), len(selected), rng)
            for row, group in zip(selected, groups):
                row["targeted_group"] = group
                row["text_case"] = "mixed"
        for row in block:
            row["official_kind"] = official.KINDS[row["content_type"]]
            row["content_type"] = row["official_kind"]
            row.setdefault("targeted_group", "none")
            row["confusable_group"] = "none"
            row["index"] += offset
    return plan


def scan_image(image, mask, profile, rng, glyph_height=None):
    if profile == "none":
        return image, {"physical_scan": {"profile": "none"}}
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Empty target glyph mask")
    # This is the measured total ink height, not a typographic point size.
    ink_height = bounds[3] - bounds[1]
    heights, blurs, qualities = SCAN_PROFILES[profile]
    requested_height = int(rng.integers(*heights))
    # Curves/perspective can make the whole ink bounding box much taller than
    # individual letters. Use the renderer's actual transformed glyph height.
    glyph_height = float(glyph_height or ink_height)
    scale = min(1.0, requested_height / glyph_height)
    low_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    low = image.resize(low_size, Image.Resampling.BOX)
    low_mask = mask.resize(low_size, Image.Resampling.BOX)
    low_bounds = low_mask.getbbox()
    radius = float(rng.uniform(*blurs))
    low = low.filter(ImageFilter.GaussianBlur(radius))
    noise = 0.0
    if profile == "photocopy_scan":
        noise = float(rng.uniform(1.5, 4.0))
        values = np.asarray(low, dtype=float)
        values += rng.normal(0, noise, values.shape[:2])[..., None]
        low = Image.fromarray(np.clip(values, 0, 255).astype("uint8"))
    quality = int(rng.integers(*qualities))
    buffer = io.BytesIO()
    low.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        result = compressed.convert("RGB").resize(image.size, Image.Resampling.BICUBIC)
    return result, {"physical_scan": {
        "profile": profile, "order": ["area_downsample", "optical_blur", "scan_noise", "jpeg", "bicubic_upsample"],
        "ink_height_before_px": ink_height, "requested_ink_height_px": requested_height,
        "ink_height_at_capture_px": low_bounds[3] - low_bounds[1], "capture_size": list(low_size),
        "glyph_height_before_px": glyph_height, "glyph_height_at_capture_px": glyph_height * scale,
        "downsample_scale": scale, "blur_radius_at_capture_px": radius,
        "noise_sigma_at_capture": noise, "jpeg_quality_at_capture": quality,
        "output_size": list(result.size), "height_measurement": "renderer glyph-height estimate plus independently measured ink bounds; not point size",
    }}


def initialize(font_manifest, config, output):
    cv2.setNumThreads(1)
    WORK.clear()
    WORK.update(fonts=base.load_font_manifest(Path(font_manifest)), config=config, output=Path(output))


def generate_one(row):
    rng = np.random.default_rng(row["seed"])
    config, fonts = WORK["config"], WORK["fonts"]
    font = fonts[row["font_index"]]
    text, source = official.select(row, font, rng)
    context = []
    context_row = {**row, "official_kind": "official_body_short", "text_case": "original",
                   "mixed_typography": "uniform", "targeted_group": "none"}
    for _ in range(2):
        context.append(official.select(context_row, font, rng)[0])
    sizes = {"small": (16, 23), "body": (27, 39), "heading": (48, 65)}
    size = int(rng.integers(*sizes[row["role"]]))
    retained = {}

    def renderer(label, face, pixels, ink, underline):
        return dataset.rich_patch(label, face, pixels, ink, underline, row, rng, fonts)

    def processor(image, mask, generator):
        image, params = dataset.postprocess(image, mask, generator, row, config)
        # postprocess may uniformly shrink exceptionally wide images. Mirror
        # its mask geometry before measuring the physical scan stage.
        scale = params["width_cap_scale"]
        if scale < 1:
            scaled = (config["max_width"], max(1, round(mask.height * scale)))
            new_mask = Image.new("L", image.size)
            new_mask.paste(mask.resize(scaled, Image.Resampling.LANCZOS), (0, (48 - scaled[1]) // 2))
            mask = new_mask
        retained["mask"] = mask
        return image, params

    image, params = base.render_line(
        text, font, size, row["background"], row["degradation"], rng,
        underline=row["underline"] == "underlined", role=row["role"],
        curve=row["curve"], curve_strength=float(rng.uniform(.2, .45)),
        geometry=row["geometry"], contrast=row["contrast"], crop_error=row["crop_error"],
        context_texts=context, clipping_severity="mild", patch_renderer=renderer,
        image_processor=processor, rotation_jitter=False, compact_scene=True)
    image, scan = scan_image(image, retained["mask"], row["scan_profile"], rng, params["text_height_output"])
    params.update(scan)
    relative = f"images/{row['index'] // 1000:03d}/line_{row['index']:06d}.jpg"
    path = WORK["output"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=params["jpeg_quality"], subsampling=0)
    categories = {key: row[key] for key in config["axes"]}
    categories.update(official_kind=row["official_kind"], targeted_group=row["targeted_group"],
                      font_family=font.family, font_style=font.style, text_case=row["text_case"],
                      confusable_group="none", clipping_severity=row["clipping_severity"])
    return {"file_name": relative, "text": text, "source_text": source, "source_article": None,
            "context_article": None, "source_group_sha256": official.source_id(source),
            "dataset_origin": "official_synthetic_v4", "source_type": "original_generated_template",
            "synthetic_values": True, "split": row["split"], "index": row["index"], "seed": row["seed"],
            "width": image.width, "height": image.height,
            "width_bucket": dataset.recognition_bucket(image.width, text, config),
            "character_count": len(text), "word_count": len(text.split()),
            "font_sha256": font.sha256, "confusable_span": None,
            "targeted_members": list(official.GROUPS.get(row["targeted_group"], ())),
            "context_source_texts": context,
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clean_control": row["clean_control"], "categories": categories, **params}


def quality_reasons(row):
    reasons = []
    if row["potentially_ambiguous"] or row["removed_components"]:
        reasons.append("clipped_character_components")
    if row["text_height_output"] < 10:
        reasons.append("very_small_output_glyphs")
    scan = row.get("physical_scan", {})
    if scan.get("glyph_height_at_capture_px", 48) < 11:
        reasons.append("very_small_capture_glyphs")
    if row["text_height_output"] < 14 and row["categories"]["degradation"] in base.STRONG_BLURS:
        reasons.append("small_text_with_strong_blur")
    if row["text_height_output"] < 14 and row.get("ink_opacity", 1) < .5:
        reasons.append("small_faint_text")
    if row.get("occluded_ink_fraction", 0) > .15:
        reasons.append("substantial_ink_occlusion")
    if row.get("printing_ink_ratio", 1) < .75:
        reasons.append("substantial_print_ink_loss")
    return reasons


def add_quality(row):
    row["quality_flags"] = quality_reasons(row)
    row["recommended_for_standard_training"] = not row["quality_flags"]
    return row


def extra_reports(root, rows):
    official_rows = [r for r in rows if r["dataset_origin"] == "official_synthetic_v4"]
    report = {"count": len(official_rows), "synthetic_only": True,
              "kinds": dict(Counter(r["categories"]["official_kind"] for r in official_rows)),
              "fonts": dict(Counter(r["categories"]["font_family"] for r in official_rows)),
              "font_styles": {}, "scan_profiles": dict(Counter(r["categories"]["scan_profile"] for r in official_rows)),
              "targeted_groups": dict(Counter(r["categories"]["targeted_group"] for r in official_rows)),
              "long_lines_over_150_characters": sum(len(r["text"]) > 150 for r in official_rows),
              "capture_ink_heights": dict(Counter(r["physical_scan"].get("ink_height_at_capture_px", "none") for r in official_rows)),
              "complete_component_loss": sum(r["removed_components"] > 0 for r in official_rows),
              "real_document_training_crops": 0,
              "limitations": ["Official references guide typography and field structure; their pages/text are not training samples.",
                              "Quality flags are measurable conservative filters, not human readability certification.",
                              "Generated template families recur across splits; real-document validation is still needed."]}
    for family in report["fonts"]:
        report["font_styles"][family] = dict(Counter(r["categories"]["font_style"] for r in official_rows if r["categories"]["font_family"] == family))
    dataset.write_json(root / "official-balance.json", report)
    quality = {"policy": "exclude clipped components, output glyph height <10, capture glyph height <11, small text with strong blur/faint ink, occlusion >15%, print ink loss >25%",
               "flags": dict(Counter(flag for r in rows for flag in r["quality_flags"])), "splits": {}}
    for split in ("train", "val", "test"):
        selected = [r for r in rows if r["split"] == split]
        quality["splits"][split] = {}
        for name, keep in (("recommended", True), ("challenge", False)):
            subset = [r for r in selected if r["recommended_for_standard_training"] == keep]
            quality["splits"][split][name] = len(subset)
            quality["splits"][split][name + "_categories"] = {
                axis: dict(Counter(r["categories"][axis] for r in subset))
                for axis in ("content_type", "font_family", "degradation", "scan_profile", "targeted_group")}
            (root / f"{split}-{name}.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in subset), encoding="utf-8")
            for width in (320, 640, 960, 1280, 1920, 2560, 4096):
                (root / f"{split}-{name}-w{width}.txt").write_text("".join(r["file_name"] + "\t" + r["text"] + "\n" for r in subset if r["width_bucket"] == width), encoding="utf-8")
    dataset.write_json(root / "quality-report.json", quality)
    return report, quality


def official_previews(root, rows):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    selected = []
    for kind in official.KINDS.values():
        candidates = [r for r in rows if r["categories"]["official_kind"] == kind]
        for category in ("none", "small_scan", "small_jpeg"):
            sample = next((r for r in candidates if r["categories"]["scan_profile"] == category
                           and r["categories"]["font_family"] in ("Times New Roman", "Arial")
                           and r["recommended_for_standard_training"]), None)
            if sample:
                selected.append(sample)
    for start in range(0, len(selected), 10):
        block = selected[start:start + 10]
        width = max(1800, max(r["width"] for r in block) + 24)
        sheet = Image.new("RGB", (width, 150 * len(block)), "#eeeeea")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(block):
            y = 150 * i
            c = row["categories"]
            draw.text((12, y + 3), f"{row['index']} | {c['official_kind']} | {c['font_family']} {c['font_style']} | {c['scan_profile']}", font=font, fill="black")
            with Image.open(root / row["file_name"]) as image:
                sheet.paste(image, (12, y + 30))
            label = row["text"]
            # Display in two lines when needed; original training label is unchanged.
            for j in range(0, len(label), 150):
                draw.text((12, y + 85 + j // 150 * 24), label[j:j + 150], font=font, fill="black")
        sheet.save(root / f"official-preview-{start // 10 + 1:02d}.jpg", quality=94)


def audit_extras(root):
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    count = 0
    for row in rows:
        if row["quality_flags"] != quality_reasons(row) or row["recommended_for_standard_training"] != (not row["quality_flags"]):
            raise ValueError("Quality flags inconsistent")
        if row["dataset_origin"] != "official_synthetic_v4":
            continue
        count += 1
        if row["removed_components"] or row["potentially_ambiguous"]:
            raise ValueError("New official crop completely removes a component")
        if row["categories"]["official_kind"] == "official_body_long" and not 150 <= len(row["text"]) <= 252:
            raise ValueError("Official long line outside planned length")
        for context in row["context_source_texts"]:
            if dataset.text_split(context) != row["split"]:
                raise ValueError("Neighbor context split leakage")
        scan = row["physical_scan"]
        if scan["profile"] != row["categories"]["scan_profile"]:
            raise ValueError("Scan category mismatch")
        if scan["profile"] != "none":
            if scan["order"] != ["area_downsample", "optical_blur", "scan_noise", "jpeg", "bicubic_upsample"]:
                raise ValueError("Scan order mismatch")
            if not 0 < scan["downsample_scale"] <= 1 or scan["output_size"] != [row["width"], 48]:
                raise ValueError("Invalid physical scan geometry")
        if row["clean_control"] and scan["profile"] != "none":
            raise ValueError("Clean control received scan degradation")
    for split in ("train", "val", "test"):
        expected = {r["file_name"] + "\t" + r["text"] for r in rows if r["split"] == split}
        recommended = set((root / f"{split}-recommended.txt").read_text().splitlines())
        challenge = set((root / f"{split}-challenge.txt").read_text().splitlines())
        if recommended & challenge or recommended | challenge != expected:
            raise ValueError("Quality subsets must partition each complete split")
    result = {"passed": True, "official_images": count, "quality_flags_checked": len(rows),
              "official_complete_component_loss": 0, "neighbor_context_split_checked": True,
              "recommended_and_challenge_lists_partition_splits": True}
    dataset.write_json(root / "official-audit.json", result)
    return result


def generate(args):
    config = json.loads(args.config.read_text())
    fonts = base.load_font_manifest(args.font_manifest)
    if args.output.exists():
        raise ValueError("Choose a fresh output directory")
    args.output.mkdir(parents=True)
    base_rows = []
    if args.base_dataset:
        old_audit = json.loads((args.base_dataset / "audit.json").read_text())
        if not old_audit["passed"]:
            raise ValueError("Base release audit must pass")
        for line in (args.base_dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["dataset_origin"] = "wikipedia_balanced_v3"
            row["categories"].update(official_kind="none", scan_profile="legacy", targeted_group="none")
            target = args.output / row["file_name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(args.base_dataset / row["file_name"], target)
            except OSError:
                shutil.copy2(args.base_dataset / row["file_name"], target)
            base_rows.append(add_quality(row))
        if len(base_rows) != old_audit["images_decoded_and_hashed"]:
            raise ValueError("Base image count mismatch")
    plan = plan_samples(config, fonts, args.additional_count, args.seed, offset=len(base_rows))
    expected_official = {axis: dict(Counter(r[axis] for r in plan)) for axis in (*config["axes"], "official_kind", "targeted_group")}
    dataset.write_json(args.output / "official-planned-counts.json", expected_official)
    dataset.write_json(args.output / "official-recipe.json", config)
    dataset.write_json(args.output / "generation-arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    rows = list(base_rows)
    started = time.monotonic()
    with (args.output / "metadata.jsonl").open("w", encoding="utf-8") as meta, (args.output / "labels.txt").open("w", encoding="utf-8") as labels:
        for row in base_rows:
            meta.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            labels.write(row["file_name"] + "\t" + row["text"] + "\n")
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(str(args.font_manifest), config, str(args.output))) as pool:
            for i, row in enumerate(pool.map(generate_one, plan, chunksize=16), 1):
                add_quality(row)
                rows.append(row)
                meta.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                labels.write(row["file_name"] + "\t" + row["text"] + "\n")
                if i % 1000 == 0 or i == len(plan):
                    meta.flush(); labels.flush()
                    print(f"Generated official {i}/{len(plan)}; {i / (time.monotonic() - started):.1f} images/s", flush=True)
    for axis, expected in expected_official.items():
        actual = Counter(r["categories"][axis] for r in rows[len(base_rows):])
        if actual != expected:
            raise ValueError(f"Official quota changed: {axis}")
    combined_config = copy.deepcopy(config)
    combined_config["source_sampling"] = "v3_plus_official_v1"
    for axis in EXTRA_AXES:
        combined_config["axes"].setdefault(axis, {})
    combined_plan = [{**r["categories"], "family": r["categories"]["font_family"], "style": r["categories"]["font_style"]} for r in base_rows] + plan
    # Counts are the sum of immutable v3 quotas and independently planned new quotas.
    for axis in combined_config["axes"]:
        combined_config["axes"][axis] = dict(Counter(r[axis] for r in combined_plan))
    dataset.write_json(args.output / "recipe.json", combined_config)
    dataset.finish(args.output, combined_config, combined_plan, rows, args.corpus, fonts, args.font_manifest, args.seed)
    meta = json.loads((args.output / "metadata.json").read_text())
    meta.update(schema_version=8, origins=dict(Counter(r["dataset_origin"] for r in rows)),
                provenance="Preserved v3 Turkish Wikipedia crops plus original generated official-style text. Public official references guide structure only.",
                balance="Explicit per-origin marginal quotas; official text kinds equal; Times New Roman/Arial weighted; styles equal within each family.")
    meta["constraints"][0] = "Clean controls have white background and no geometric/capture defects; v3 quota 5%, official quota 10%."
    meta["constraints"].append("Physical scan profiles exclude additional blur, faint ink, print damage, occlusion and abnormal exposure.")
    for name in ("expand_tr_official.py", "official_tr_text.py"):
        meta["generator_sha256"][name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    if args.base_dataset:
        meta["base_release"] = {"directory": args.base_dataset.name,
                                "metadata_sha256": hashlib.sha256((args.base_dataset / "metadata.jsonl").read_bytes()).hexdigest(),
                                "images_and_splits_preserved": len(base_rows)}
        shutil.copy2(args.base_dataset / "source-balance.json", args.output / "base-source-balance.json")
    dataset.write_json(args.output / "metadata.json", meta)
    extra_reports(args.output, rows)
    official_previews(args.output, rows)
    print(json.dumps(dataset.audit(args.output), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(audit_extras(args.output), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config_tr_recognition_official.json")
    parser.add_argument("--font-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="Used only to retain Wikipedia attribution")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.additional_count < 1000 or args.workers < 1:
        parser.error("Use at least 1,000 additional samples and a positive worker count")
    generate(args)


if __name__ == "__main__":
    main()
