"""Independent scalar checks against the published AMD FSR 1.0 equations."""

import math

import numpy as np

from sr_tool.fsr.easu import easu
from sr_tool.fsr.rcas import rcas

_TAPS = {
    "b": (0, -1),
    "c": (1, -1),
    "e": (-1, 0),
    "f": (0, 0),
    "g": (1, 0),
    "h": (2, 0),
    "i": (-1, 1),
    "j": (0, 1),
    "k": (1, 1),
    "l": (2, 1),
    "n": (0, 2),
    "o": (1, 2),
}


def _scalar_easu_pixel(
    image: np.ndarray,
    output_y: int,
    output_x: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    source_x = output_x * width / output_width + 0.5 * width / output_width - 0.5
    source_y = output_y * height / output_height + 0.5 * height / output_height - 0.5
    floor_x, floor_y = math.floor(source_x), math.floor(source_y)
    fraction_x, fraction_y = source_x - floor_x, source_y - floor_y

    colors: dict[str, np.ndarray] = {}
    lumas: dict[str, float] = {}
    for name, (offset_x, offset_y) in _TAPS.items():
        x = min(max(floor_x + offset_x, 0), width - 1)
        y = min(max(floor_y + offset_y, 0), height - 1)
        color = image[y, x].astype(np.float64)
        colors[name] = color
        lumas[name] = float(color[1] + 0.5 * (color[0] + color[2]))

    direction_x = direction_y = edge_length = 0.0

    def accumulate(weight: float, names: tuple[str, str, str, str, str]) -> None:
        nonlocal direction_x, direction_y, edge_length
        a, b, c, d, e = (lumas[name] for name in names)
        delta_x = d - b
        magnitude_x = max(abs(d - c), abs(c - b))
        direction_x += delta_x * weight
        normalized_x = min(abs(delta_x) / magnitude_x, 1.0) if magnitude_x else 0.0
        delta_y = e - a
        magnitude_y = max(abs(e - c), abs(c - a))
        direction_y += delta_y * weight
        normalized_y = min(abs(delta_y) / magnitude_y, 1.0) if magnitude_y else 0.0
        edge_length += (normalized_x**2 + normalized_y**2) * weight

    accumulate((1.0 - fraction_x) * (1.0 - fraction_y), ("b", "e", "f", "g", "j"))
    accumulate(fraction_x * (1.0 - fraction_y), ("c", "f", "g", "h", "k"))
    accumulate((1.0 - fraction_x) * fraction_y, ("f", "i", "j", "k", "n"))
    accumulate(fraction_x * fraction_y, ("g", "j", "k", "l", "o"))

    direction_squared = direction_x**2 + direction_y**2
    if direction_squared < 1.0 / 32768.0:
        direction_x, direction_y, edge_length = 1.0, 0.0, 0.0
    else:
        inverse = 1.0 / math.sqrt(direction_squared)
        direction_x *= inverse
        direction_y *= inverse
    edge_length = (edge_length * 0.5) ** 2
    stretch = 1.0 / max(abs(direction_x), abs(direction_y))
    stretch_x = 1.0 + (stretch - 1.0) * edge_length
    stretch_y = 1.0 - 0.5 * edge_length
    lobe = 0.5 - 0.29 * edge_length
    clip_distance = 1.0 / lobe

    color_sum = np.zeros(3, dtype=np.float64)
    weight_sum = 0.0
    for name, (tap_x, tap_y) in _TAPS.items():
        offset_x = tap_x - fraction_x
        offset_y = tap_y - fraction_y
        rotated_x = (offset_x * direction_x + offset_y * direction_y) * stretch_x
        rotated_y = (offset_x * -direction_y + offset_y * direction_x) * stretch_y
        distance = min(rotated_x**2 + rotated_y**2, clip_distance)
        base = 0.4 * distance - 1.0
        window = lobe * distance - 1.0
        weight = ((25.0 / 16.0) * base**2 - 9.0 / 16.0) * window**2
        color_sum += colors[name] * weight
        weight_sum += weight
    filtered = color_sum / weight_sum
    nearest = np.stack([colors[name] for name in ("f", "g", "j", "k")])
    return np.clip(filtered, nearest.min(axis=0), nearest.max(axis=0))


def test_easu_matches_independent_scalar_equations() -> None:
    image = np.random.default_rng(101).random((5, 7, 3), dtype=np.float32)
    actual = easu(image, 2, block_size=5)
    expected = np.empty_like(actual, dtype=np.float64)
    for y in range(expected.shape[0]):
        for x in range(expected.shape[1]):
            expected[y, x] = _scalar_easu_pixel(
                image, y, x, expected.shape[0], expected.shape[1]
            )
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=2e-5)


def _scalar_rcas(image: np.ndarray, sharpness: float) -> np.ndarray:
    height, width = image.shape[:2]
    output = np.empty_like(image)
    for y in range(height):
        for x in range(width):
            center = image[y, x].astype(np.float64)
            neighbors = np.stack(
                [
                    image[max(y - 1, 0), x],
                    image[y, max(x - 1, 0)],
                    image[y, min(x + 1, width - 1)],
                    image[min(y + 1, height - 1), x],
                ]
            ).astype(np.float64)
            ring_min = neighbors.min(axis=0)
            ring_max = neighbors.max(axis=0)
            hit_min = np.divide(
                np.minimum(ring_min, center),
                4.0 * ring_max,
                out=np.zeros(3),
                where=ring_max != 0.0,
            )
            hit_max = np.divide(
                1.0 - np.maximum(ring_max, center),
                4.0 * ring_min - 4.0,
                out=np.zeros(3),
                where=ring_min != 1.0,
            )
            lobe = np.clip(np.maximum(-hit_min, hit_max).max(), -0.1875, 0.0)
            lobe *= 2.0**-sharpness
            output[y, x] = np.clip(
                (lobe * neighbors.sum(axis=0) + center) / (4.0 * lobe + 1.0),
                0.0,
                1.0,
            )
    return output


def test_rcas_matches_independent_scalar_equations() -> None:
    image = np.random.default_rng(103).random((7, 9, 3), dtype=np.float32)
    np.testing.assert_allclose(rcas(image, 0.35), _scalar_rcas(image, 0.35), atol=2e-7)
