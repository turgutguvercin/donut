from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expand_tr_official as official
import official_tr_text as text
import prepare_paddle_tr_v5 as paddle


@pytest.fixture(scope="module")
def config():
    return json.loads((official.ROOT / "config_tr_recognition_official.json").read_text())


def test_official_100k_plan_has_exact_quotas_and_protected_scans(config):
    families = ["Times New Roman", "Arial"] + [f"family{i}" for i in range(18)]
    fonts = [SimpleNamespace(family=family, style=style) for family in families for style in official.base.STYLES]
    plan = official.plan_samples(config, fonts, 100000, 20261016, 100000)
    assert Counter(r["split"] for r in plan) == {"train": 90000, "val": 5000, "test": 5000}
    assert Counter(r["official_kind"] for r in plan) == dict.fromkeys(text.KINDS.values(), 10000)
    assert Counter(r["scan_profile"] for r in plan) == {"none": 70000, "small_clear": 5000, "small_scan": 10000, "small_jpeg": 10000, "photocopy_scan": 5000}
    assert sum(r["family"] == "Times New Roman" for r in plan) == 20000
    assert sum(r["family"] == "Arial" for r in plan) == 20000
    assert len({r["index"] for r in plan}) == 100000
    assert min(r["index"] for r in plan) == 100000
    styles = defaultdict(Counter)
    for r in plan:
        styles[r["family"]][r["style"]] += 1
        assert r["clipping_severity"] != "severe"
        if r["clean_control"]:
            assert r["scan_profile"] == "none"
        if r["scan_profile"] != "none":
            assert r["degradation"] == "clean" and r["contrast"] == "normal"
            assert r["printing"] == r["occlusion"] == "none" and r["exposure"] == "normal"
    assert all(max(c.values()) - min(c.values()) <= 1 for c in styles.values())
    for kind in ("official_confusables_words", "official_confusables_codes"):
        groups = Counter(r["targeted_group"] for r in plan if r["official_kind"] == kind)
        assert set(groups) == set(text.GROUPS)
        assert max(groups.values()) - min(groups.values()) <= 3


def test_official_text_split_unicode_length_and_required_fields(config):
    font = SimpleNamespace(codepoints=set(range(0x10000)))
    for split in ("train", "val", "test"):
        for i, kind in enumerate(text.KINDS.values()):
            row = {"index": i, "split": split, "official_kind": kind, "text_case": "original",
                   "mixed_typography": "uniform", "targeted_group": "province_Il"}
            label, source = text.select(row, font, np.random.default_rng(42 + i))
            assert official.dataset.text_split(label) == official.dataset.text_split(source) == split
            assert official.dataset.safe_text(label) and len(label) <= 252
            if kind == "official_body_long":
                assert len(label) >= 150
            if kind == "official_confusables_codes":
                assert "İl" in label and "İ1" in label
    f = text.fields(np.random.default_rng(42))
    outputs = [text.raw_text("official_footer_contact", f, np.random.default_rng(i)) for i in range(100)]
    assert any("@" in s and "example.org" in s for s in outputs)
    assert any("Telefon:" in s and "Dahili:" in s for s in outputs)
    assert any("Adres:" in s and "/" in s for s in outputs)
    assert any("KEP:" in s for s in outputs)


def test_physical_scan_uses_glyph_height_not_curved_ink_extent():
    image, mask = Image.new("RGB", (600, 48), "white"), Image.new("L", (600, 48))
    draw, ink = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    for x in range(10, 580, 12):
        y = 5 if x < 300 else 25
        draw.rectangle((x, y, x + 4, y + 12), fill="black")
        ink.rectangle((x, y, x + 4, y + 12), fill=255)
    a, params = official.scan_image(image, mask, "small_scan", np.random.default_rng(3), glyph_height=13)
    b, again = official.scan_image(image, mask, "small_scan", np.random.default_rng(3), glyph_height=13)
    assert a.size == image.size and a.tobytes() == b.tobytes() and params == again
    scan = params["physical_scan"]
    assert scan["ink_height_before_px"] > 2 * scan["glyph_height_before_px"]
    assert scan["glyph_height_at_capture_px"] >= 10
    assert scan["downsample_scale"] == min(1, scan["requested_ink_height_px"] / 13)
    assert a.tobytes() != image.tobytes()


def test_quality_filter_keeps_ambiguous_and_small_capture_out_of_normal_training():
    row = {"potentially_ambiguous": False, "removed_components": 0, "text_height_output": 25,
           "occluded_ink_fraction": 0, "printing_ink_ratio": 1, "categories": {"degradation": "clean"}}
    assert official.quality_reasons(row) == []
    assert "clipped_character_components" in official.quality_reasons({**row, "removed_components": 1})
    assert "very_small_capture_glyphs" in official.quality_reasons({**row, "physical_scan": {"glyph_height_at_capture_px": 9}})
    assert "substantial_print_ink_loss" in official.quality_reasons({**row, "printing_ink_ratio": .7})


def test_v5_configs_replace_short_limits_and_use_explicit_filtered_lists():
    template = {"Global": {"max_text_length": 25, "d2s_train_image_shape": [3, 48, 320]},
                "Architecture": {"algorithm": "SVTR_HGNet", "Backbone": {"name": "PPHGNetV2_B4"},
                                 "Head": {"head_list": [{"CTCHead": {}}, {"NRTRHead": {"max_text_length": 25}}]}},
                "Train": {"sampler": {"name": "MultiScaleSampler"}, "loader": {}}, "Eval": {"loader": {}}}
    original = copy.deepcopy(template)
    cfg = paddle.training_config(template, "./data", 4096)
    assert template == original
    assert cfg["Global"]["max_text_length"] == cfg["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] == 256
    assert cfg["Global"]["d2s_train_image_shape"] == [3, 48, 4096]
    assert cfg["Architecture"]["Backbone"]["name"] == "PPHGNetV2_B4"
    assert "sampler" not in cfg["Train"]
    assert cfg["Train"]["dataset"]["label_file_list"] == ["./data/train-recommended.txt"]
    assert cfg["Eval"]["dataset"]["label_file_list"] == ["./data/val-recommended.txt"]
    all_samples = paddle.training_config(template, "./data", 4096, recommended=False)
    assert all_samples["Train"]["dataset"]["label_file_list"] == ["./data/train.txt"]
    subset = paddle.training_config(template, "./data", 640, bucket_only=True)
    assert subset["Train"]["dataset"]["label_file_list"] == ["./data/train-recommended-w640.txt"]
