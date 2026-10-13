from pathlib import Path
import json
import sys

from PIL import ImageFont
import numpy as np
import pytest
from synthtiger.layers import TextLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_tr_recognition import (
    BACKGROUNDS, DEGRADATIONS, TURKISH, discover_fonts, generate,
    load_font_manifest, render_line, text_patch, turkish_upper, validate,
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
