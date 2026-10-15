"""Regressions for Turkish vocabulary transfer and complete long-line epochs."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paddle_v6_utils import (WIDTHS, bucket_batches, characters, choose_bucket,
                             remap_axis, tokens)
from prepare_paddle_tr_v6 import training_config


@pytest.mark.parametrize("head", ["ctc", "nrtr"])
@pytest.mark.parametrize("axis", [0, 1])
def test_transfer_uses_identity_even_when_vocabularies_have_same_size(head, axis):
    old = tokens(["i", "İ", "ı", "ç", "ü", " "], head)
    new = tokens(["ü", "i", "ğ", "İ", "ı", " "], head)
    original = np.arange(len(old) * 3).reshape(len(old), 3)
    initial = np.full((len(new), 3), -17)
    if axis == 1:
        original, initial = original.T, initial.T
    result = remap_axis(original, initial, old, new, axis)
    for i, token in enumerate(new):
        actual = np.take(result, i, axis=axis)
        expected = np.take(original, old.index(token), axis=axis) if token in old else np.full(3, -17)
        np.testing.assert_array_equal(actual, expected)
    assert np.all(initial == -17), "Caller initialization must not be mutated"
    if head == "nrtr":
        assert new[:4] == ["<blank>", "<unk>", "<s>", "</s>"]
        assert new[4] == "ü" and new[-1] == "<nrtr_unused_output>"


def test_checkpoint_shape_and_ambiguous_dictionary_fail_closed():
    with pytest.raises(ValueError, match="class count"):
        remap_axis(np.zeros((3, 2)), np.zeros((2, 2)), ["a"], ["b", "c"], 0)
    with pytest.raises(ValueError, match="Non-vocabulary"):
        remap_axis(np.zeros((2, 3)), np.zeros((2, 4)), ["a", "b"], ["b", "a"], 0)
    with pytest.raises(ValueError, match="Ambiguous"):
        remap_axis(np.zeros((2, 3)), np.zeros((2, 3)), ["a", "a"], ["b", "a"], 0)


def test_dictionary_keeps_real_unicode_space_and_turkish_case(tmp_path):
    path = tmp_path / "dict.txt"
    path.write_bytes("i\r\nİ\r\nı\r\nI\r\nç\r\nü\r\n\u3000\r\n".encode())
    assert characters(path) == ["i", "İ", "ı", "I", "ç", "ü", "\u3000", " "]
    for bad in ("i\ni\n", "i\n \n", "i\n\n", "ij\n", ""):
        path.write_text(bad)
        with pytest.raises(ValueError):
            characters(path)


def test_ctc_capacity_includes_repeats_and_preserves_image_width():
    measured = {w: w // 8 for w in WIDTHS}
    assert choose_bucket(320, "ab" * 20, measured) == 320
    assert choose_bucket(320, "a" * 21, measured) == 640  # 41 steps, not 21.
    assert choose_bucket(1900, "İl", measured) == 1920
    assert choose_bucket(4000, "ı" * 254, measured) == 4096
    with pytest.raises(ValueError):
        choose_bucket(4097, "Türkçe", measured)
    with pytest.raises(ValueError):
        choose_bucket(320, "ı" * 257, measured)


def test_sampler_retains_every_bucket_and_partial_batch_and_reshuffles():
    groups = {320: list(range(11)), 640: list(range(11, 18)), 4096: [18, 19, 20]}
    sizes = {320: 4, 640: 3, 4096: 2}
    first = bucket_batches(groups, sizes, 72, 0)
    second = bucket_batches(groups, sizes, 72, 1)
    assert first == bucket_batches(groups, sizes, 72, 0) and first != second
    membership = {i: width for width, ids in groups.items() for i in ids}
    for epoch in (first, second):
        assert sorted(i for batch in epoch for i in batch) == list(range(21))
        assert len(epoch) == 8
        assert all(len({membership[i] for i in batch}) == 1 for batch in epoch)
        assert any(len(batch) == 1 for batch in epoch)
    assert groups[320] == list(range(11))


def test_training_config_keeps_medium_architecture_and_all_samples():
    template = {
        "Global": {"max_text_length": 25},
        "Architecture": {"Backbone": {"name": "PPLCNetV4", "model_size": "medium"},
            "Head": {"head_list": [{"CTCHead": {"Neck": {"name": "lightsvtr"}}}, {"NRTRHead": {"nrtr_dim": 512}}]}},
        "Optimizer": {"lr": {"name": "Cosine"}}, "Metric": {"name": "RecMetric"},
        "Train": {"sampler": {"name": "MultiScaleSampler"}, "loader": {"drop_last": True}},
        "Eval": {"loader": {"drop_last": True}},
    }
    before = copy.deepcopy(template)
    config = training_config(template, "data", "data/paddle-v6-medium")
    assert template == before
    assert config["Architecture"]["Backbone"] == before["Architecture"]["Backbone"]
    assert config["Global"]["max_text_length"] == config["Architecture"]["Head"]["head_list"][1]["NRTRHead"]["max_text_length"] == 256
    assert config["TurkishRecognitionPooling"]["enabled"]
    assert set(config["TurkishWidthBuckets"]["widths"]) == set(WIDTHS)
    assert config["Metric"]["ignore_space"] is False and config["Metric"]["is_filter"] is False
    for mode in ("Train", "Eval"):
        assert "sampler" not in config[mode]
        assert config[mode]["dataset"]["ratio_list"] == [1.0]
        assert config[mode]["loader"]["drop_last"] is False
        ops = config[mode]["dataset"]["transforms"]
        assert [next(iter(op)) for op in ops] == ["DecodeImage", "MultiLabelEncode", "RecResizeImg", "KeepKeys"]
        assert ops[2]["RecResizeImg"]["image_shape"] == [3, 48, 4096]
    fixed = training_config(template, "data", "adapter", bucketed=False, recommended=False)
    assert "TurkishWidthBuckets" not in fixed
    assert fixed["TurkishRecognitionPooling"]["enabled"]
    assert fixed["Train"]["dataset"]["label_file_list"] == ["data/train.txt"]
    template["Architecture"]["Backbone"]["model_size"] = "small"
    with pytest.raises(ValueError, match="medium"):
        training_config(template, "data", "adapter")
