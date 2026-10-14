from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_effects as capture
import generate_tr_dataset as dataset
import generate_tr_recognition as base


@pytest.mark.parametrize("geometry", capture.SIDE_GEOMETRIES)
def test_camera_shrinks_height_and_spacing_at_far_end(geometry):
    corners, params = capture.side_view_corners((1001, 81), geometry, np.random.default_rng(12))
    # Repeated identical rectangles make the actual raster's taper measurable.
    patch = Image.new("RGBA", (1001, 81))
    draw = ImageDraw.Draw(patch)
    for x in range(20, 1000, 50):
        draw.rectangle((x, 10, x + 20, 70), fill="black")
    warped, mapping = base.perspective_patch(patch, corners)
    alpha = np.asarray(warped.getchannel("A")) > 128
    active = np.where(alpha.any(axis=0))[0]
    groups = np.split(active, np.where(np.diff(active) > 1)[0] + 1)
    assert len(groups) == 20
    heights = [alpha[:, g].any(axis=1).sum() for g in groups]
    widths = [len(g) for g in groups]
    centers = [g.mean() for g in groups]
    left_near = params["camera_side"] == "left"
    assert (heights[0] > heights[-1]) == left_near
    assert (widths[0] > widths[-1]) == left_near
    assert (centers[1] - centers[0] > centers[-1] - centers[-2]) == left_near
    assert warped.width < patch.width
    result, lighting = capture.apply_lighting(Image.new("RGB", (10, 10), "white"), np.random.default_rng(3))
    capture.audit_capture({"categories": {"geometry": geometry, "exposure": "normal", "lighting_effect": "none"},
                           **params, **mapping, **lighting})


def test_exposure_changes_whole_crop_and_is_deterministic():
    image = Image.new("RGB", (300, 48), (170, 170, 170))
    ImageDraw.Draw(image).rectangle((0, 20, 299, 30), fill=(40, 40, 40))
    outputs = {}
    for exposure in capture.EXPOSURES:
        a, params = capture.apply_lighting(image, np.random.default_rng(5), exposure=exposure)
        b, replay = capture.apply_lighting(image, np.random.default_rng(5), exposure=exposure)
        assert a.tobytes() == b.tobytes() and params == replay
        outputs[exposure] = np.asarray(a, dtype=float)
    assert np.array_equal(outputs["normal"], np.asarray(image))
    for y in (10, 25):  # Paper AND ink change, not just ink opacity.
        means = [outputs[k][y].mean() for k in ("low_light", "underexposed", "normal", "bright", "overexposed")]
        assert means == sorted(means) and len(set(means)) == 5
    assert outputs["low_light"][10].std() > 1
    assert outputs["underexposed"][10].std() == 0


@pytest.mark.parametrize("kind", ("soft_shadow", "hard_shadow", "partial_shadow"))
def test_shadows_cover_only_part_of_crop(kind):
    image = Image.new("RGB", (600, 48), (220, 220, 220))
    output, params = capture.apply_lighting(image, np.random.default_rng(31), lighting=kind)
    row = np.asarray(output)[24, :, 0].astype(float)
    assert row.max() > 205 and row.min() < 145
    assert .1 < params["capture_lighting"]["shadowed_ink_fraction"] < .95
    if kind == "hard_shadow":
        assert np.abs(np.diff(row)).max() > 35
    if kind == "soft_shadow":
        assert np.abs(np.diff(row)).max() < 5


def test_camera_recipe_exact_quotas_and_protected_controls():
    config = json.loads((dataset.ROOT / "config_tr_recognition_camera.json").read_text())
    fonts = [SimpleNamespace(family=f"family{i}", style="regular") for i in range(20)]
    rows = dataset.plan_samples(config, fonts, 100000, 20261014)
    for axis, weights in config["axes"].items():
        actual, expected = Counter(r[axis] for r in rows), dataset.quotas(weights, 100000)
        assert all(abs(actual[key] - count) <= 2 for key, count in expected.items())
    for row in rows:
        if row["contrast"] == "low_contrast":
            assert row["degradation"] not in base.STRONG_BLURS
        if row["clean_control"] or row["clipping_severity"] == "severe" or row["contrast"] == "faint_print":
            assert row["exposure"] == "normal"
            assert row["geometry"] == row["lighting_effect"] == "none"
        if row["exposure"] in ("low_light", "overexposed"):
            assert row["contrast"] == "normal"
            assert row["printing"] == row["occlusion"] == "none"
            assert row["degradation"] not in base.STRONG_BLURS


def test_lighting_none_preserves_rng():
    rng = np.random.default_rng(42)
    capture.apply_lighting(Image.new("RGB", (10, 48), "white"), rng)
    assert rng.random() == np.random.default_rng(42).random()
