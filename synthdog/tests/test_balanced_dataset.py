from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_tr_dataset as dataset
import generate_tr_recognition as base


@pytest.fixture(scope="module")
def config():
    return json.loads((dataset.ROOT / "config_tr_recognition_100k.json").read_text())


@pytest.fixture(scope="module")
def fonts():
    try:
        return base.discover_fonts(Path("/usr/share/fonts/truetype"))
    except FileNotFoundError as error:
        pytest.skip(str(error))


def test_full_quota_plan_and_readability_constraints(config):
    # Match a catalog with uneven style availability, including italic-only faces.
    fonts = [SimpleNamespace(family=f"family{i}", style=s)
             for i in range(20) for s in (base.STYLES if i < 13 else
                                         ("regular", "bold") if i < 16 else
                                         ("regular",) if i < 19 else ("italic",))]
    rows = dataset.plan_samples(config, fonts, 100000, 20261014)
    assert Counter(r["family"] for r in rows) == dict.fromkeys([f"family{i}" for i in range(20)], 5000)
    assert Counter(r["split"] for r in rows) == {"train": 90000, "val": 5000, "test": 5000}
    assert Counter(r["content_type"] for r in rows) == dict.fromkeys(config["axes"]["content_type"], 10000)
    assert sum(r["clean_control"] for r in rows) == 5000
    assert sum(r["clipping_severity"] == "severe" for r in rows) == 7000
    assert Counter(r["confusable_group"] for r in rows if r["confusable_group"] != "none") == dict.fromkeys(config["confusable_groups"], 1250)
    for axis, weights in config["axes"].items():
        planned = Counter(r[axis] for r in rows)
        # Thirds have at most two samples of split-rounding discrepancy.
        target = dataset.quotas(weights, 100000)
        assert all(abs(planned[key] - target[key]) <= 2 for key in target)
    per_family = defaultdict(Counter)
    for row in rows:
        per_family[row["family"]][row["style"]] += 1
        if row["clipping_severity"] == "severe":
            assert row["degradation"] == "clean" and row["contrast"] == "normal"
            assert row["printing"] == row["occlusion"] == "none"
        if row["contrast"] == "faint_print":
            assert row["degradation"] == "clean"
        if row["contrast"] == "low_contrast":
            assert row["degradation"] not in base.STRONG_BLURS
    assert all(max(c.values()) - min(c.values()) <= 1 for c in per_family.values())


def test_turkish_case_split_identity():
    for label in ("İstanbul ılık ışık", "Çiğdem şeker üzüm", "kalem", "göl", "çığ"):
        assert dataset.canonical(label) == dataset.canonical(base.turkish_upper(label))
        assert dataset.text_split(label) == dataset.text_split(base.turkish_upper(label))


def test_real_font_runs_spacing_and_script_baselines(fonts):
    font = fonts[0]
    label = "Çiğdem ışığı gördü 2"
    row = {"mixed_typography": "uniform", "spacing": "normal", "printing": "none"}
    widths = {}
    for spacing in ("touching", "tight", "normal", "wide"):
        image, params = dataset.rich_patch(label, font, 36, (0, 0, 0), True,
                                           {**row, "spacing": spacing}, np.random.default_rng(7), fonts)
        widths[spacing] = image.width
        assert image.getbbox() and params["underline_gap_native"] >= 2
    assert widths["touching"] < widths["tight"] < widths["normal"] < widths["wide"]
    for mode in ("mixed_fonts", "mixed_styles", "mixed_sizes", "superscript", "subscript"):
        _, params = dataset.rich_patch(label, font, 36, (0, 0, 0), False,
                                       {**row, "mixed_typography": mode}, np.random.default_rng(7), fonts)
        runs = params["font_runs"]
        assert "".join(label[r["start"]:r["end"]] for r in runs) == label.replace(" ", "")
        if mode == "mixed_fonts":
            assert len({r["family"] for r in runs}) == 2
        elif mode == "mixed_styles":
            assert len({r["style"] for r in runs}) == 2
        elif mode == "mixed_sizes":
            assert len({r["size"] for r in runs}) == 2
        else:
            assert runs[-1]["baseline_offset"] < 0 if mode == "superscript" else runs[-1]["baseline_offset"] > 0


def test_width_cap_preserves_aspect_and_measures_occlusion(config):
    image, mask = Image.new("RGB", (8192, 48), "white"), Image.new("L", (8192, 48), 255)
    result, params = dataset.postprocess(image, mask, np.random.default_rng(4),
                                         {"occlusion": "none", "lighting_effect": "none"}, config)
    assert result.size == (4096, 48)
    assert params["width_cap_scale"] == .5
    assert params["glyph_bounds_output"] == [0, 12, 4096, 36]
    assert params["occluded_ink_fraction"] == 0


def test_occlusions_reach_even_off_center_glyphs(config):
    from PIL import ImageDraw
    image, mask = Image.new("RGB", (300, 48), "white"), Image.new("L", (300, 48))
    ImageDraw.Draw(mask).rectangle((3, 1, 8, 5), fill=255)
    for kind in ("stamp", "signature", "fold", "strikethrough"):
        _, params = dataset.postprocess(image, mask, np.random.default_rng(4),
                                         {"occlusion": kind, "lighting_effect": "none"}, config)
        assert params["occluded_ink_fraction"] > 0


@pytest.mark.parametrize("edge", base.HORIZONTAL_ERRORS)
def test_horizontal_crop_ink_bounds(fonts, edge):
    image, params = base.render_line("Çiğdem üzüm İŞÇİ", fonts[0], 40, "white", "clean",
                                     np.random.default_rng(7), crop_error=edge, clipping_severity="severe")
    assert image.height == 48 and params["potentially_ambiguous"]
    assert .7 <= params["target_ink_retention"] < 1
    assert params.get("clipped_left_native", 0) + params.get("clipped_right_native", 0) > 0


def test_rich_sample_replay_and_exact_confusable_members(tmp_path, fonts, config):
    rows = dataset.plan_samples(config, fonts, 1000, 42)
    selected = [r for r in rows if r["content_type"] in ("confusables", "date", "price", "synthetic_id")][:24]
    dataset.WORKER.update(fonts=fonts, config=config,
                          pools={("medium_line", split): ["Çiğdem İstanbul'da üzüm ve şeker aldı."]
                                 for split in ("train", "val", "test")})
    results = []
    for repeat in ("a", "b"):
        dataset.WORKER["output"] = tmp_path / repeat
        samples = [dataset.generate_one(row) for row in selected]
        results.append(samples)
        for sample in samples:
            assert sample["height"] == 48 and sample["width"] <= sample["width_bucket"]
            if sample["confusable_span"]:
                start, end = sample["confusable_span"]
                expected = config["confusable_groups"][sample["categories"]["confusable_group"]]
                assert Counter(sample["text"][start:end].replace(" ", "")) == Counter(expected)
    assert results[0] == results[1]
