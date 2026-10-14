"""Opt-in camera geometry and lighting for recognition crops.

Camera geometry projects a flat line rotated about its vertical axis. Lighting
acts on the complete RGB crop in linear light, including paper and context.
"""
import numpy as np
from PIL import Image

SIDE_GEOMETRIES = tuple(f"perspective_{side}_{strength}"
                        for side in ("left", "right") for strength in ("mild", "strong"))
EXPOSURES = ("normal", "low_light", "underexposed", "bright", "overexposed")
LIGHTING = ("none", "gradient", "soft_shadow", "hard_shadow", "partial_shadow")


def side_view_corners(size, geometry, rng):
    """Return a pinhole projection; the named side is nearer to the camera.

    Normalize scale so the near edge retains its source height. A plane's
    projected height varies as 1/depth; projected x coordinates have the same
    denominator, so distant glyphs also become horizontally smaller.
    """
    if geometry not in SIDE_GEOMETRIES or min(size) < 2:
        raise ValueError("Invalid side-view geometry or patch dimensions")
    _, side, strength = geometry.split("_")
    ratio = float(rng.uniform(*((.70, .85) if strength == "mild" else (.40, .60))))
    yaw = float(rng.uniform(*((20, 35) if strength == "mild" else (40, 55))))
    width, height = np.asarray(size, dtype=float) - 1
    cosine = float(np.cos(np.radians(yaw)))
    left_scale, right_scale = (1., ratio) if side == "left" else (ratio, 1.)
    destination = np.array([
        (-width / 2 * cosine * left_scale, -height / 2 * left_scale),
        (width / 2 * cosine * right_scale, -height / 2 * right_scale),
        (width / 2 * cosine * right_scale, height / 2 * right_scale),
        (-width / 2 * cosine * left_scale, height / 2 * left_scale),
    ])
    return destination, {
        "camera_side": side, "camera_strength": strength,
        "camera_yaw_degrees": yaw if side == "right" else -yaw,
        "far_to_near_height_ratio": ratio,
        "projected_width_ratio": cosine * (1 + ratio) / 2,
        "camera_model": "planar_pinhole_near_edge_scale_1",
    }


def to_linear(array):
    values = array / 255.
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def to_srgb(array):
    values = np.clip(array, 0, 1)
    return 255 * np.where(values <= .0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - .055)


def apply_lighting(image, rng, lighting="none", exposure="normal", glyph_mask=None):
    if lighting not in LIGHTING or exposure not in EXPOSURES:
        raise ValueError("Unknown camera lighting or exposure")
    width, height = image.size
    xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    profile = np.ones((height, width), dtype=float)
    params = {"lighting_effect": lighting, "exposure": exposure,
              "exposure_ev": 0., "veiling_light": 0.,
              "sensor_read_noise_sigma": 0., "sensor_shot_noise_sigma": 0.}
    if lighting == "gradient":
        minimum = float(rng.uniform(.38, .60))
        reverse = bool(rng.integers(2))
        profile = minimum + (1 - minimum) * (1 - xx if reverse else xx)
        params.update(gradient_minimum=minimum, gradient_reversed=reverse)
    elif lighting != "none":
        center = float(rng.uniform(.35, .65))
        slope = float(rng.uniform(-.22, .22))
        coordinate = xx + slope * (yy - .5) - center
        minimum = float(rng.uniform(.12, .30))
        if lighting == "soft_shadow":
            spread = float(rng.uniform(.13, .25))
            profile = 1 - (1 - minimum) * np.exp(-.5 * (coordinate / spread) ** 2)
            params["shadow_spread_fraction"] = spread
        else:
            edge_pixels = float(rng.uniform(.8, 1.8) if lighting == "hard_shadow" else rng.uniform(2, 4))
            if lighting == "partial_shadow":
                half_width = float(rng.uniform(.15, .25))
                signed_distance = np.abs(coordinate) - half_width
                params["shadow_half_width_fraction"] = half_width
            else:
                direction = int(rng.choice((-1, 1)))
                signed_distance = coordinate * direction
                params["shadow_direction"] = direction
            fraction = np.clip(signed_distance * max(1, width - 1) / edge_pixels + .5, 0, 1)
            fraction = fraction * fraction * (3 - 2 * fraction)
            profile = minimum + (1 - minimum) * fraction
            params["shadow_edge_pixels"] = edge_pixels
        params.update(shadow_center_fraction=center, shadow_slope=slope, shadow_minimum=minimum)
    ev_ranges = {"low_light": (-3.8, -2.5), "underexposed": (-1.8, -.8),
                 "bright": (.35, .75), "overexposed": (1.0, 1.8)}
    if exposure != "normal":
        params["exposure_ev"] = float(rng.uniform(*ev_ranges[exposure]))
    if exposure == "overexposed":
        # Veiling light models a washed-out capture, distinct from faint ink.
        params["veiling_light"] = float(rng.uniform(.02, .06))
    source = np.asarray(image, dtype=float)
    linear = to_linear(source) * profile[:, :, None] * 2 ** params["exposure_ev"]
    linear += params["veiling_light"]
    result = to_srgb(linear)
    if exposure == "low_light":
        params["sensor_read_noise_sigma"] = float(rng.uniform(1.0, 2.0))
        params["sensor_shot_noise_sigma"] = float(rng.uniform(2.0, 4.0))
        sigma = np.sqrt(params["sensor_read_noise_sigma"] ** 2
                        + params["sensor_shot_noise_sigma"] ** 2 * result / 255)
        result += rng.normal(0, 1, result.shape) * sigma
    # A no-op must stay byte-identical, without spending any random numbers.
    output = image.copy() if lighting == "none" and exposure == "normal" else Image.fromarray(
        np.clip(np.rint(result), 0, 255).astype("uint8"))
    mask = np.asarray(glyph_mask) >= 64 if glyph_mask is not None else np.ones(profile.shape, dtype=bool)
    luma = np.array((.2126, .7152, .0722))
    params.update(illumination_min=float(profile.min()), illumination_max=float(profile.max()),
                  shadowed_ink_fraction=float(((profile < .75) & mask).sum() / max(1, mask.sum())),
                  mean_luma_before=float((source @ luma).mean()),
                  mean_luma_after=float((np.asarray(output) @ luma).mean()))
    return output, {"capture_lighting": params}


def audit_capture(row):
    """Validate recorded camera mapping and lighting against assigned categories."""
    categories = row["categories"]
    geometry = categories["geometry"]
    if geometry in SIDE_GEOMETRIES:
        _, side, strength = geometry.split("_")
        corners = np.array(row["perspective_target_corners"])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        ratio = min(left_height, right_height) / max(left_height, right_height)
        low, high = (.70, .85) if strength == "mild" else (.40, .60)
        if (row["camera_side"] != side or row["camera_strength"] != strength
                or not low <= ratio <= high
                or not np.isclose(ratio, row["far_to_near_height_ratio"])
                or (left_height > right_height) != (side == "left")):
            raise ValueError("Camera direction/taper metadata differs from projected corners")
    lighting = row["capture_lighting"]
    if (lighting["exposure"] != categories["exposure"]
            or lighting["lighting_effect"] != categories["lighting_effect"]
            or not 0 < lighting["illumination_min"] <= lighting["illumination_max"] <= 1):
        raise ValueError("Incorrect camera lighting metadata")
    if categories["lighting_effect"].endswith("shadow") and lighting["shadowed_ink_fraction"] <= 0:
        raise ValueError("Shadow misses all target ink")
