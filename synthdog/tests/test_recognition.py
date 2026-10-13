from pathlib import Path
import sys

from PIL import ImageFont
import pytest
from synthtiger.layers import TextLayer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_tr_recognition import BACKGROUNDS, DEGRADATIONS, TURKISH, discover_fonts, generate, validate

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
