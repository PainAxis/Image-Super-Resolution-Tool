"""NVIDIA FXAA 3.11 quality path, adapted for NumPy image arrays.

The implementation follows ``FxaaPixelShader`` with quality preset 12. It
uses perceptual BT.601 luma (the reference RGBL integration path) and samples
in global image coordinates so block size does not affect output pixels.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sr_tool.fsr.common import (
    CancelCallback,
    check_cancelled,
    validate_rgb_image,
    validate_unit_interval,
)

# FXAA 3.11 reference defaults and quality preset 12.
_EDGE_THRESHOLD = 0.166
_EDGE_THRESHOLD_MIN = 0.0833
_SUBPIX_QUALITY = 0.75
_QUALITY_STEPS = (1.0, 1.5, 2.0, 4.0, 12.0)
_ITERATIONS = 12  # Compatibility cap; preset 12 contains five search steps.
_FXAA_BLOCK = 512


def _rgb2luma(rgb: np.ndarray) -> np.ndarray:
    """Return perceptual BT.601 luma as required by the RGBL FXAA path."""
    return (
        rgb[..., 0] * np.float32(0.299)
        + rgb[..., 1] * np.float32(0.587)
        + rgb[..., 2] * np.float32(0.114)
    )


def _gather(plane: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Gather integer coordinates with clamp-to-edge addressing."""
    height, width = plane.shape[:2]
    return plane[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]


def _sample_bilinear(
    plane: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Sample a 2-D or channel-last array at pixel-center coordinates."""
    height, width = plane.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, float(width - 1))
    y = np.clip(np.asarray(y, dtype=np.float32), 0.0, float(height - 1))
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fraction_x = x - x0.astype(np.float32)
    fraction_y = y - y0.astype(np.float32)
    if plane.ndim == 3:
        fraction_x = fraction_x[..., None]
        fraction_y = fraction_y[..., None]
    top = plane[y0, x0] * (1.0 - fraction_x) + plane[y0, x1] * fraction_x
    bottom = plane[y1, x0] * (1.0 - fraction_x) + plane[y1, x1] * fraction_x
    return top * (1.0 - fraction_y) + bottom * fraction_y


def _fxaa_region(
    img: np.ndarray,
    luma: np.ndarray,
    y_start: int,
    y_stop: int,
    x_start: int,
    x_stop: int,
    edge_threshold: float,
    edge_threshold_min: float,
    subpix_quality: float,
    quality_steps: tuple[float, ...],
) -> np.ndarray:
    """Apply the FXAA shader logic to a rectangular global output region."""
    y_int = np.arange(y_start, y_stop, dtype=np.intp)[:, None]
    x_int = np.arange(x_start, x_stop, dtype=np.intp)[None, :]
    tile_shape = (y_stop - y_start, x_stop - x_start)
    y = np.broadcast_to(y_int, tile_shape)
    x = np.broadcast_to(x_int, tile_shape)
    y_float = y.astype(np.float32)
    x_float = x.astype(np.float32)

    luma_m = _gather(luma, y, x)
    luma_n = _gather(luma, y - 1, x)
    luma_s = _gather(luma, y + 1, x)
    luma_w = _gather(luma, y, x - 1)
    luma_e = _gather(luma, y, x + 1)
    range_max = np.maximum.reduce((luma_m, luma_n, luma_s, luma_w, luma_e))
    range_min = np.minimum.reduce((luma_m, luma_n, luma_s, luma_w, luma_e))
    luma_range = range_max - range_min
    active = luma_range >= np.maximum(edge_threshold_min, range_max * edge_threshold)
    if not np.any(active):
        return img[y_start:y_stop, x_start:x_stop].copy()

    luma_nw = _gather(luma, y - 1, x - 1)
    luma_ne = _gather(luma, y - 1, x + 1)
    luma_sw = _gather(luma, y + 1, x - 1)
    luma_se = _gather(luma, y + 1, x + 1)

    luma_ns = luma_n + luma_s
    luma_we = luma_w + luma_e
    luma_ne_se = luma_ne + luma_se
    luma_nw_ne = luma_nw + luma_ne
    luma_nw_sw = luma_nw + luma_sw
    luma_sw_se = luma_sw + luma_se
    edge_horizontal = (
        np.abs((-2.0 * luma_w) + luma_nw_sw)
        + np.abs((-2.0 * luma_m) + luma_ns) * 2.0
        + np.abs((-2.0 * luma_e) + luma_ne_se)
    )
    edge_vertical = (
        np.abs((-2.0 * luma_n) + luma_nw_ne)
        + np.abs((-2.0 * luma_m) + luma_we) * 2.0
        + np.abs((-2.0 * luma_s) + luma_sw_se)
    )
    orientation_tolerance = np.float32(8.0 * np.finfo(np.float32).eps) * np.maximum(
        1.0, np.maximum(edge_horizontal, edge_vertical)
    )
    ambiguous_orientation = (
        np.abs(edge_horizontal - edge_vertical) <= orientation_tolerance
    )
    horizontal_span = edge_horizontal >= edge_vertical

    subpixel_average = (luma_ns + luma_we) * 2.0 + luma_nw_sw + luma_ne_se
    subpixel_delta = subpixel_average * np.float32(1.0 / 12.0) - luma_m
    reciprocal_range = np.zeros_like(luma_range)
    np.divide(1.0, luma_range, out=reciprocal_range, where=luma_range > 0.0)
    subpixel = np.clip(np.abs(subpixel_delta) * reciprocal_range, 0.0, 1.0)
    subpixel = ((-2.0 * subpixel) + 3.0) * subpixel * subpixel
    subpixel = subpixel * subpixel * subpix_quality

    negative_luma = np.where(horizontal_span, luma_n, luma_w)
    positive_luma = np.where(horizontal_span, luma_s, luma_e)
    negative_gradient = negative_luma - luma_m
    positive_gradient = positive_luma - luma_m
    absolute_negative_gradient = np.abs(negative_gradient)
    absolute_positive_gradient = np.abs(positive_gradient)
    pair_tolerance = np.float32(8.0 * np.finfo(np.float32).eps) * np.maximum(
        1.0,
        np.maximum(absolute_negative_gradient, absolute_positive_gradient),
    )
    ambiguous_pair = (
        np.abs(absolute_negative_gradient - absolute_positive_gradient)
        <= pair_tolerance
    )
    pair_negative = absolute_negative_gradient >= absolute_positive_gradient
    gradient = np.maximum(absolute_negative_gradient, absolute_positive_gradient)
    length_sign = np.where(pair_negative, -1.0, 1.0).astype(np.float32)
    paired_luma_sum = np.where(
        pair_negative,
        negative_luma + luma_m,
        positive_luma + luma_m,
    )
    local_luma_delta = luma_m - paired_luma_sum * np.float32(0.5)
    local_luma_negative = local_luma_delta < 0.0
    gradient_scaled = gradient * np.float32(0.25)

    tangent_x = horizontal_span.astype(np.float32)
    tangent_y = (~horizontal_span).astype(np.float32)
    base_x = x_float + np.where(horizontal_span, 0.0, length_sign * 0.5)
    base_y = y_float + np.where(horizontal_span, length_sign * 0.5, 0.0)
    first_step = np.float32(quality_steps[0])
    negative_x = base_x - tangent_x * first_step
    negative_y = base_y - tangent_y * first_step
    positive_x = base_x + tangent_x * first_step
    positive_y = base_y + tangent_y * first_step
    local_average = paired_luma_sum * np.float32(0.5)
    negative_end = _sample_bilinear(luma, negative_y, negative_x) - local_average
    positive_end = _sample_bilinear(luma, positive_y, positive_x) - local_average
    done_negative = (~active) | (np.abs(negative_end) >= gradient_scaled)
    done_positive = (~active) | (np.abs(positive_end) >= gradient_scaled)

    # The final preset distance is intentionally not sampled afterwards. This
    # mirrors Fxaa3_11.h, where it bounds an unfinished endpoint search.
    for index, distance in enumerate(quality_steps[1:], start=1):
        search_negative = ~done_negative
        search_positive = ~done_positive
        negative_x = np.where(
            search_negative, negative_x - tangent_x * distance, negative_x
        )
        negative_y = np.where(
            search_negative, negative_y - tangent_y * distance, negative_y
        )
        positive_x = np.where(
            search_positive, positive_x + tangent_x * distance, positive_x
        )
        positive_y = np.where(
            search_positive, positive_y + tangent_y * distance, positive_y
        )
        if index == len(quality_steps) - 1:
            break
        if np.any(search_negative):
            sampled = _sample_bilinear(luma, negative_y, negative_x) - local_average
            negative_end = np.where(search_negative, sampled, negative_end)
            done_negative |= np.abs(negative_end) >= gradient_scaled
        if np.any(search_positive):
            sampled = _sample_bilinear(luma, positive_y, positive_x) - local_average
            positive_end = np.where(search_positive, sampled, positive_end)
            done_positive |= np.abs(positive_end) >= gradient_scaled

    distance_negative = np.where(
        horizontal_span,
        x_float - negative_x,
        y_float - negative_y,
    )
    distance_positive = np.where(
        horizontal_span,
        positive_x - x_float,
        positive_y - y_float,
    )
    span_length = distance_negative + distance_positive
    pixel_offset = np.float32(0.5) - np.minimum(
        distance_negative, distance_positive
    ) / span_length
    good_negative = (negative_end < 0.0) != local_luma_negative
    good_positive = (positive_end < 0.0) != local_luma_negative
    direction_negative = distance_negative < distance_positive
    good_span = np.where(direction_negative, good_negative, good_positive)
    equal_distance = np.isclose(
        distance_negative, distance_positive, rtol=1e-6, atol=1e-6
    )
    good_span = np.where(equal_distance, good_negative & good_positive, good_span)
    edge_offset = np.where(good_span, pixel_offset, 0.0)
    final_offset = np.maximum(edge_offset, subpixel)
    # The shader resolves exact gradient ties toward one fixed neighbor. On a
    # CPU still-image filter that creates a directional impulse halo and makes
    # mirrored inputs differ. Conservatively retain the center in this truly
    # ambiguous case; non-tied FXAA behavior remains reference-equivalent.
    final_offset = np.where(ambiguous_pair | ambiguous_orientation, 0.0, final_offset)

    sample_x = x_float + np.where(horizontal_span, 0.0, final_offset * length_sign)
    sample_y = y_float + np.where(horizontal_span, final_offset * length_sign, 0.0)
    filtered = _sample_bilinear(img, sample_y, sample_x)
    center = img[y_start:y_stop, x_start:x_stop]
    return np.where(active[..., None], filtered, center).astype(np.float32, copy=False)


def _quality_distances(iterations: int) -> tuple[float, ...]:
    """Map the legacy iteration cap onto preset 12's fixed search schedule."""
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise TypeError("iterations must be an integer")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    return _QUALITY_STEPS[: min(iterations, len(_QUALITY_STEPS))]


def _fxaa_core(
    img: np.ndarray,
    edge_threshold: float = _EDGE_THRESHOLD,
    edge_threshold_min: float = _EDGE_THRESHOLD_MIN,
    subpix_quality: float = _SUBPIX_QUALITY,
    iterations: int = _ITERATIONS,
) -> np.ndarray:
    """Apply FXAA without public block iteration (used for conformance tests)."""
    source = validate_rgb_image(img)
    edge_threshold = validate_unit_interval("edge_threshold", edge_threshold)
    edge_threshold_min = validate_unit_interval(
        "edge_threshold_min", edge_threshold_min
    )
    subpix_quality = validate_unit_interval("subpix_quality", subpix_quality)
    return _fxaa_region(
        source,
        _rgb2luma(source),
        0,
        source.shape[0],
        0,
        source.shape[1],
        edge_threshold,
        edge_threshold_min,
        subpix_quality,
        _quality_distances(iterations),
    )


def fxaa(
    img: np.ndarray,
    edge_threshold: float = _EDGE_THRESHOLD,
    edge_threshold_min: float = _EDGE_THRESHOLD_MIN,
    subpix_quality: float = _SUBPIX_QUALITY,
    iterations: int = _ITERATIONS,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: CancelCallback | None = None,
    *,
    block_size: int = _FXAA_BLOCK,
) -> np.ndarray:
    """Apply FXAA 3.11 quality preset 12 with block-bounded temporaries."""
    source = validate_rgb_image(img)
    edge_threshold = validate_unit_interval("edge_threshold", edge_threshold)
    edge_threshold_min = validate_unit_interval(
        "edge_threshold_min", edge_threshold_min
    )
    subpix_quality = validate_unit_interval("subpix_quality", subpix_quality)
    quality_steps = _quality_distances(iterations)
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    check_cancelled(cancel_callback)
    height, width = source.shape[:2]
    luma = _rgb2luma(source)
    output = np.empty_like(source)
    vertical_blocks = (height + block_size - 1) // block_size
    horizontal_blocks = (width + block_size - 1) // block_size
    total_blocks = vertical_blocks * horizontal_blocks
    completed = 0
    for y_start in range(0, height, block_size):
        check_cancelled(cancel_callback)
        y_stop = min(y_start + block_size, height)
        for x_start in range(0, width, block_size):
            check_cancelled(cancel_callback)
            x_stop = min(x_start + block_size, width)
            output[y_start:y_stop, x_start:x_stop] = _fxaa_region(
                source,
                luma,
                y_start,
                y_stop,
                x_start,
                x_stop,
                edge_threshold,
                edge_threshold_min,
                subpix_quality,
                quality_steps,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed / total_blocks)
    return np.clip(output, 0.0, 1.0)
