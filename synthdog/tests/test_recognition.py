from pathlib import Path
from dataclasses import replace
from datetime import datetime
import json
import re
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pytest
from synthtiger.layers import TextLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_tr_recognition import (
    BACKGROUNDS, CURVES, DEGRADATIONS, TURKISH, curve_patch, discover_fonts, generate,
    load_font_manifest, perspective_patch, render_line, structured_text, text_patch, turkish_upper, validate,
)

FONT_ROOT = Path("/usr/share/fonts/truetype")


@pytest.fixture(scope="module")
def fonts():
    try:
        return discover_fonts(FONT_ROOT)
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def batch(tmp_path_factory, fonts):
    root = tmp_path_factory.mktemp("recognition")
    corpus = root / "corpus.txt"
    corpus.write_text(
        "İstanbul'da öğleden sonra güzel bir gün. Çocuklar üzüm ve şeker alıyor. "
        "Güneşli sokakta kitap okuyan öğrenciler. ÇĞİÖŞÜ çğıöşü Iı. " * 30,
        encoding="utf-8",
    )
    output = root / "samples"
    metadata = generate(corpus, output, FONT_ROOT)
    return corpus, output, metadata


def test_real_styles_and_turkish_glyphs(fonts):
    assert len({font.path for font in fonts}) == 20
    for font in fonts:
        actual_style = ImageFont.truetype(str(font.path), 32).getname()[1].lower()
        if "italic" in font.style:
            assert "italic" in actual_style or "oblique" in actual_style
        if "bold" in font.style:
            assert "bold" in actual_style
        for char in TURKISH:
            layer = TextLayer(char, path=str(font.path), size=32)
            assert layer.image[..., 3].max() > 0


def test_batch_diversity_and_labels(batch):
    corpus, output, metadata = batch
    assert validate(output) == metadata
    assert set(metadata["families"].values()) == {4}
    assert set(metadata["styles"].values()) == {5}
    assert set(metadata["backgrounds"]) == set(BACKGROUNDS)
    assert set(metadata["degradations"]) == set(DEGRADATIONS)
    assert len({s["image_sha256"] for s in metadata["samples"]}) == 20
    assert len({s["source_font_size"] for s in metadata["samples"]}) > 4
    assert "İ".encode("utf-8") in (output / "labels.txt").read_bytes()


def test_seed_reproduces_images_and_labels(batch):
    corpus, output, metadata = batch
    replay = output.parent / "replay"
    assert generate(corpus, replay, FONT_ROOT) == metadata
    assert (output / "labels.txt").read_bytes() == (replay / "labels.txt").read_bytes()
    for sample in metadata["samples"]:
        assert (output / sample["file_name"]).read_bytes() == (replay / sample["file_name"]).read_bytes()


def test_existing_output_is_preserved(batch):
    corpus, output, metadata = batch
    before = (output / "metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate(corpus, output, FONT_ROOT)
    assert (output / "metadata.json").read_bytes() == before


def test_validator_rejects_incorrect_labels(batch):
    corpus, output, metadata = batch
    labels = output / "labels.txt"
    before = labels.read_bytes()
    try:
        rows = labels.read_text(encoding="utf-8").splitlines()
        rows[0] = rows[0].split("\t")[0] + "\tYanlış etiket"
        labels.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mismatched"):
            validate(output)
    finally:
        labels.write_bytes(before)


def test_custom_fonts_use_actual_names_and_check_hashes(tmp_path, fonts):
    font = fonts[0]
    notice = tmp_path / "notice.txt"
    notice.write_text("Test license notice", encoding="utf-8")
    catalog = tmp_path / "fonts.json"
    entry = {"path": str(font.path), "license": font.license,
             "license_file": str(notice), "sha256": font.sha256}
    catalog.write_text(json.dumps({"fonts": [entry]}), encoding="utf-8")
    loaded = load_font_manifest(catalog)[0]
    assert loaded.family == ImageFont.truetype(str(font.path), 32).getname()[0]
    assert loaded.style == "regular"
    catalog.write_text(json.dumps({"fonts": [{**entry, "sha256": "wrong"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_font_manifest(catalog)
    notice.unlink()
    with pytest.raises(ValueError, match="license"):
        load_font_manifest(catalog)


def test_custom_catalog_balances_families_with_different_style_counts(batch, fonts):
    corpus, output, metadata = batch
    catalog = output.parent / "fonts.json"
    # Four Sans variants, four Serif variants, but just one Mono variant.
    chosen = [font for font in fonts if font.family != "DejaVu Sans Mono" or font.style == "regular"]
    chosen = [font for font in chosen if font.family.startswith("DejaVu")]
    notice = output.parent / "notice.txt"
    notice.write_text("Test license notice", encoding="utf-8")
    catalog.write_text(json.dumps({"fonts": [
        {"path": str(font.path), "license": font.license, "license_file": str(notice), "sha256": font.sha256}
        for font in chosen
    ]}), encoding="utf-8")
    result = generate(corpus, output.parent / "custom", FONT_ROOT, count=24, font_manifest=catalog)
    assert len(result["families"]) == 3
    assert set(result["families"].values()) == {8}
    notices = list((output.parent / "custom/licenses").glob("*-notice.txt"))
    assert len(notices) == 1
    assert notices[0].read_text(encoding="utf-8") == "Test license notice"


def test_turkish_uppercase():
    assert turkish_upper("istanbul ısparta İZMİR ışık şeker çığ öykü üzüm") == (
        "İSTANBUL ISPARTA İZMİR IŞIK ŞEKER ÇIĞ ÖYKÜ ÜZÜM"
    )


@pytest.mark.parametrize("style", ["regular", "bold", "italic", "bold_italic"])
def test_underline_preserves_glyphs_and_clears_descenders(fonts, style):
    font = next(font for font in fonts if font.style == style)
    plain, _ = text_patch("İĞçğışöü", font, 48, (0, 0, 0))
    underlined, params = text_patch("İĞçğışöü", font, 48, (0, 0, 0), underline=True)
    pixels = np.asarray(underlined)
    assert np.array_equal(pixels[:plain.height], np.asarray(plain))
    assert not pixels[plain.height:plain.height + params["underline_gap_native"], :, 3].any()
    assert (pixels[-params["underline_thickness_native"]:, :, 3] == 255).all()
    assert underlined.height > plain.height


def test_typography_batch_and_rendered_case(batch):
    corpus, output, metadata = batch
    assert metadata["typography"]["roles"] == {"heading": 10, "body": 5, "small": 5}
    assert metadata["typography"]["underlined"] == 10
    for sample in metadata["samples"]:
        if sample["role"] == "heading":
            assert sample["text"].isupper()
            assert sample["text"] == turkish_upper(sample["text"])
            assert 48 <= sample["source_font_size"] <= 64
        elif sample["role"] == "body":
            assert 26 <= sample["source_font_size"] <= 38
        else:
            assert 18 <= sample["source_font_size"] <= 22


def test_heading_pixels_are_larger_than_small_text(fonts):
    def ink_height(image):
        pixels = np.asarray(image)
        ys = np.where(pixels.mean(axis=2) < 96)[0]
        assert len(ys)
        return ys.max() - ys.min() + 1
    heading, _ = render_line("İSTANBUL IŞIK ÇIĞ", fonts[0], 56, "white", "clean", np.random.default_rng(42), role="heading")
    small, _ = render_line("İSTANBUL IŞIK ÇIĞ", fonts[0], 20, "white", "clean", np.random.default_rng(42), role="small")
    assert heading.height == small.height == 48
    assert ink_height(heading) >= ink_height(small) + 8


@pytest.mark.parametrize("curve", ["arch", "bowl", "wave"])
def test_curves_keep_every_glyph_and_underline_pixel(fonts, curve):
    patch, _ = text_patch("İĞçğışöü", fonts[0], 48, (12, 45, 67), underline=True)
    warped = curve_patch(patch, curve, 20)
    source, result = np.asarray(patch), np.asarray(warped)
    assert warped.width == patch.width
    assert warped.height > patch.height
    # Every colored/antialiased pixel survives in its original reading column,
    # including Turkish dots, accents, descenders and the underline.
    for x in range(patch.width):
        before = source[:, x][source[:, x, 3] > 0]
        after = result[:, x][result[:, x, 3] > 0]
        assert np.array_equal(before, after)
    bottom = ((result[..., 3] > 0) * np.arange(result.shape[0])[:, None]).max(axis=0)
    assert bottom.max() - bottom.min() == 20


def test_strong_defocus_reduces_edges_more_than_standard(fonts):
    args = ("ÇĞİŞÖÜ İSTANBUL", fonts[0], 56, "white")
    standard, standard_params = render_line(*args, "defocus", np.random.default_rng(42), role="heading", curve="arch")
    strong, strong_params = render_line(*args, "strong_defocus", np.random.default_rng(42), role="heading", curve="arch")
    def edge_energy(image):
        gray = np.asarray(image, dtype=float).mean(axis=2)
        return np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()
    assert standard.size == strong.size
    assert strong_params["blur_radius"] > standard_params["blur_radius"]
    assert edge_energy(strong) < edge_energy(standard)


def test_curved_strong_batch_and_straight_option(batch):
    corpus, output, _ = batch
    strong = generate(corpus, output.parent / "strong", FONT_ROOT, curve_mode="mixed", blur_profile="strong")
    assert validate(output.parent / "strong") == strong
    assert set(strong["curves"]) == set(CURVES)
    assert set(strong["curves"].values()) == {5}
    assert sum(strong["degradations"][key] for key in ("strong_defocus", "strong_motion", "defocus_motion")) == 12
    for sample in strong["samples"]:
        assert (sample["curve_amplitude_native"] > 0) == (sample["curve"] != "straight")
        if sample["degradation"] == "defocus_motion":
            assert sample["blur_radius"] > 0 and sample["motion_kernel_width"] in (3, 5)
    straight = generate(corpus, output.parent / "straight", FONT_ROOT, curve_mode="straight")
    assert straight["curves"] == {"straight": 20}


def test_perspective_keeps_all_corners_inside_canvas():
    patch = Image.new("RGBA", (180, 52))
    draw = ImageDraw.Draw(patch)
    colors = ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255))
    for (x, y), color in zip(((0, 0), (171, 0), (171, 43), (0, 43)), colors):
        draw.rectangle((x, y, x + 8, y + 8), fill=color)
    warped, params = perspective_patch(patch, ((-7, 10), (172, 0), (179, 51), (0, 41)))
    pixels = np.asarray(warped)
    for color in colors:
        assert np.count_nonzero(np.all(pixels == color, axis=2)) >= 5
    assert not pixels[[0, -1], :, 3].any()
    assert not pixels[:, [0, -1], 3].any()
    inverse = np.append(params["perspective_inverse_coefficients"], 1).reshape(3, 3)
    for source, target in zip(params["perspective_source_corners"], params["perspective_target_corners"]):
        mapped = inverse @ np.array([*target, 1])
        assert np.allclose(mapped[:2] / mapped[2], source)


@pytest.mark.parametrize("background_kind", ["white", "dark_sign"])
def test_contrast_reduces_ink_visibility_in_both_polarities(fonts, background_kind):
    levels = []
    for contrast in ("normal", "low_contrast", "faint_print"):
        image, params = render_line("İĞçğışöü", fonts[0], 52, background_kind, "clean",
                                    np.random.default_rng(42), contrast=contrast)
        pixels = np.asarray(image, dtype=float).mean(axis=2)
        paper = np.median(pixels)
        levels.append(np.abs(pixels - paper).max())
    assert levels[0] > levels[1] > levels[2] > 10


def test_skew_is_bounded_for_long_lines(fonts):
    image, params = render_line("İSTANBUL ÇIĞ GÜZEL " * 5, fonts[0], 52, "white", "clean",
                                np.random.default_rng(42), geometry="skew")
    assert 0 < abs(params["rotation_degrees"]) <= 4
    assert params["text_height_output"] >= 15
    assert image.height == 48


@pytest.mark.parametrize("text", ["güçlü", "güçlü bir görüş " * 5])
def test_small_text_survives_combined_perspective_and_curvature(fonts, text):
    for font in fonts:
        image, params = render_line(text, font, 18, "white", "clean", np.random.default_rng(42),
                                    role="small", curve="wave", curve_strength=0.7,
                                    geometry="perspective", contrast="faint_print")
        assert params["text_height_output"] >= 9
        assert image.height == 48


def test_structured_fields_are_valid_and_use_supported_glyphs(fonts):
    rng = np.random.default_rng(9)
    font = replace(fonts[0], codepoints=fonts[0].codepoints - {ord("₺")})
    months = "Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık".split()
    for _ in range(50):
        value = structured_text("date", font, rng).split(": ")[-1]
        if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", value):
            datetime.strptime(value, "%d.%m.%Y")
        elif re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
            datetime.strptime(value, "%d/%m/%Y")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            datetime.strptime(value, "%Y-%m-%d")
        else:
            day, month, year = value.split()
            datetime(int(year), months.index(month) + 1, int(day))
        price = structured_text("price", font, rng, "uppercase")
        assert re.search(r"(?:^|: )\d{1,3}(?:\.\d{3})*,\d{2} TL$", price)
        identifier = structured_text("synthetic_id", font, rng, "uppercase")
        assert re.search(r"TR-20\d{2}-\d{6}$", identifier)
        assert identifier == turkish_upper(identifier)
        assert set(map(ord, price + identifier)) <= font.codepoints


def test_full_category_batch_and_reproducible_labels(batch):
    corpus, output, _ = batch
    options = dict(count=40, capture_profile="mixed", content_profile="mixed", blur_profile="balanced")
    root = output.parent / "categories"
    metadata = generate(corpus, root, FONT_ROOT, **options)
    assert validate(root) == metadata
    categories = metadata["categories"]
    assert categories["geometry"] == {"none": 20, "skew": 10, "perspective": 10}
    assert categories["contrast"] == {"normal": 24, "low_contrast": 12, "faint_print": 4}
    assert categories["content_type"] == {"wikipedia": 24, "date": 6, "price": 6, "synthetic_id": 4}
    assert set(categories["text_length"]) == {"single_word", "short_line", "medium_line", "long_line"}
    assert categories["blur_strength"]["strong"] == 8
    for sample in metadata["samples"]:
        if sample["contrast"] == "faint_print":
            assert sample["degradation"] == "clean"
        elif sample["contrast"] == "low_contrast":
            assert sample["categories"]["blur_strength"] != "strong"
        if sample["geometry"] == "perspective":
            assert len(sample["perspective_target_corners"]) == 4
    replay = generate(corpus, output.parent / "categories-replay", FONT_ROOT, **options)
    assert metadata == replay
    csv_path = root / "categories.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("wikipedia", "incorrect", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="CSV categories"):
        validate(root)
