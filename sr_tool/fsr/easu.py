"""AMD FidelityFX Super Resolution 1.0 EASU, ported to NumPy.

This is a CPU/vectorized translation of the 32-bit ``FsrEasuF`` path in
AMD's public FSR 1.0 reference implementation. Work is split into output
tiles, while every tap is addressed in global source coordinates; therefore
tile boundaries cannot change the result.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sr_tool.fsr.common import CancelCallback, check_cancelled, validate_rgb_image

# AMD reference: GPUOpen-Effects/FidelityFX-FSR, ffx-fsr/ffx_fsr1.h.
BLOCK_SIZE = 128
_DIRECTION_EPSILON = np.float32(1.0 / 32768.0)

# Tap layout relative to source texel f (x, y), matching FsrEasuF.
_TAPS: tuple[tuple[int, int, str], ...] = (
    (0, -1, "b"),
    (1, -1, "c"),
    (-1, 0, "e"),
    (0, 0, "f"),
    (1, 0, "g"),
    (2, 0, "h"),
    (-1, 1, "i"),
    (0, 1, "j"),
    (1, 1, "k"),
    (2, 1, "l"),
    (0, 2, "n"),
    (1, 2, "o"),
)


def _lanczos2(x: np.ndarray | float) -> np.ndarray:
    """Compatibility helper retained for callers of the former implementation."""
    values = np.asarray(x, dtype=np.float32)
    absolute = np.abs(values)
    result = np.zeros_like(absolute)
    mask = absolute < 2.0
    result[mask] = np.sinc(absolute[mask]) * np.sinc(absolute[mask] / 2.0)
    return result


def _gather(
    img: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    offset_y: int,
    offset_x: int,
) -> np.ndarray:
    """Gather a clamped source tap for an output tile."""
    height, width = img.shape[:2]
    y = np.clip(iy + offset_y, 0, height - 1)
    x = np.clip(ix + offset_x, 0, width - 1)
    return img[y, x]


def _luma(rgb: np.ndarray) -> np.ndarray:
    """FSR's inexpensive luma-times-two approximation."""
    return rgb[..., 1] + np.float32(0.5) * (rgb[..., 0] + rgb[..., 2])


def _accumulate_direction(
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    edge_length: np.ndarray,
    weight: np.ndarray,
    luma_a: np.ndarray,
    luma_b: np.ndarray,
    luma_c: np.ndarray,
    luma_d: np.ndarray,
    luma_e: np.ndarray,
) -> None:
    """Vectorized translation of AMD ``FsrEasuSetF``."""
    dc = luma_d - luma_c
    cb = luma_c - luma_b
    length_x = np.maximum(np.abs(dc), np.abs(cb))
    direction_delta_x = luma_d - luma_b
    direction_x += direction_delta_x * weight
    normalized_x = np.zeros_like(length_x)
    np.divide(
        np.abs(direction_delta_x),
        length_x,
        out=normalized_x,
        where=length_x > 0.0,
    )
    normalized_x = np.clip(normalized_x, 0.0, 1.0)
    edge_length += normalized_x * normalized_x * weight

    ec = luma_e - luma_c
    ca = luma_c - luma_a
    length_y = np.maximum(np.abs(ec), np.abs(ca))
    direction_delta_y = luma_e - luma_a
    direction_y += direction_delta_y * weight
    normalized_y = np.zeros_like(length_y)
    np.divide(
        np.abs(direction_delta_y),
        length_y,
        out=normalized_y,
        where=length_y > 0.0,
    )
    normalized_y = np.clip(normalized_y, 0.0, 1.0)
    edge_length += normalized_y * normalized_y * weight


def _tap_weight(
    offset_x: np.ndarray,
    offset_y: np.ndarray,
    direction_x: np.ndarray,
    direction_y: np.ndarray,
    stretch_x: np.ndarray,
    stretch_y: np.ndarray,
    lobe: np.ndarray,
    clip_distance: np.ndarray,
) -> np.ndarray:
    """Vectorized translation of AMD ``FsrEasuTapF``."""
    rotated_x = offset_x * direction_x + offset_y * direction_y
    rotated_y = offset_x * -direction_y + offset_y * direction_x
    rotated_x *= stretch_x
    rotated_y *= stretch_y
    distance_squared = np.minimum(
        rotated_x * rotated_x + rotated_y * rotated_y,
        clip_distance,
    )
    window = np.float32(0.4) * distance_squared - np.float32(1.0)
    lobe_window = lobe * distance_squared - np.float32(1.0)
    window *= window
    lobe_window *= lobe_window
    window = np.float32(25.0 / 16.0) * window - np.float32(9.0 / 16.0)
    return window * lobe_window


def _easu_grid(
    img: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    fraction_y: np.ndarray,
    fraction_x: np.ndarray,
) -> np.ndarray:
    """Evaluate EASU for an arbitrary broadcast-compatible coordinate grid."""
    tile_shape = np.broadcast_shapes(
        iy.shape, ix.shape, fraction_y.shape, fraction_x.shape
    )
    iy = np.broadcast_to(iy, tile_shape)
    ix = np.broadcast_to(ix, tile_shape)
    fraction_y = np.broadcast_to(fraction_y, tile_shape)
    fraction_x = np.broadcast_to(fraction_x, tile_shape)

    colors: dict[str, np.ndarray] = {}
    lumas: dict[str, np.ndarray] = {}
    for tap_x, tap_y, name in _TAPS:
        color = _gather(img, iy, ix, tap_y, tap_x)
        colors[name] = color
        lumas[name] = _luma(color)

    direction_x = np.zeros(tile_shape, dtype=np.float32)
    direction_y = np.zeros(tile_shape, dtype=np.float32)
    edge_length = np.zeros(tile_shape, dtype=np.float32)
    one = np.float32(1.0)
    _accumulate_direction(
        direction_x,
        direction_y,
        edge_length,
        (one - fraction_x) * (one - fraction_y),
        lumas["b"],
        lumas["e"],
        lumas["f"],
        lumas["g"],
        lumas["j"],
    )
    _accumulate_direction(
        direction_x,
        direction_y,
        edge_length,
        fraction_x * (one - fraction_y),
        lumas["c"],
        lumas["f"],
        lumas["g"],
        lumas["h"],
        lumas["k"],
    )
    _accumulate_direction(
        direction_x,
        direction_y,
        edge_length,
        (one - fraction_x) * fraction_y,
        lumas["f"],
        lumas["i"],
        lumas["j"],
        lumas["k"],
        lumas["n"],
    )
    _accumulate_direction(
        direction_x,
        direction_y,
        edge_length,
        fraction_x * fraction_y,
        lumas["g"],
        lumas["j"],
        lumas["k"],
        lumas["l"],
        lumas["o"],
    )

    direction_squared = direction_x * direction_x + direction_y * direction_y
    flat = direction_squared < _DIRECTION_EPSILON
    inverse_length = np.ones_like(direction_squared)
    np.divide(
        one,
        np.sqrt(direction_squared),
        out=inverse_length,
        where=~flat,
    )
    direction_x = np.where(flat, one, direction_x * inverse_length)
    direction_y = np.where(flat, np.float32(0.0), direction_y * inverse_length)

    # A cancelled/undefined direction cannot support oriented anisotropy.
    # Keeping the accumulated edge length while choosing a fixed x direction
    # creates rare but severe transpose artifacts, so use an isotropic kernel.
    edge_length = np.where(flat, np.float32(0.0), edge_length)
    edge_length = np.square(edge_length * np.float32(0.5))
    max_direction = np.maximum(np.abs(direction_x), np.abs(direction_y))
    stretch = (direction_x * direction_x + direction_y * direction_y) / max_direction
    stretch_x = one + (stretch - one) * edge_length
    stretch_y = one - np.float32(0.5) * edge_length
    lobe = np.float32(0.5) + np.float32(-0.29) * edge_length
    clip_distance = one / lobe

    color_sum = np.zeros((*tile_shape, 3), dtype=np.float32)
    weight_sum = np.zeros(tile_shape, dtype=np.float32)
    for tap_x, tap_y, name in _TAPS:
        offset_x = np.float32(tap_x) - fraction_x
        offset_y = np.float32(tap_y) - fraction_y
        weight = _tap_weight(
            offset_x,
            offset_y,
            direction_x,
            direction_y,
            stretch_x,
            stretch_y,
            lobe,
            clip_distance,
        )
        color_sum += colors[name] * weight[..., None]
        weight_sum += weight

    filtered = np.empty_like(color_sum)
    np.divide(
        color_sum,
        weight_sum[..., None],
        out=filtered,
        where=weight_sum[..., None] != 0.0,
    )
    invalid = weight_sum == 0.0
    if np.any(invalid):
        bilinear = (
            colors["f"] * ((one - fraction_x) * (one - fraction_y))[..., None]
            + colors["g"] * (fraction_x * (one - fraction_y))[..., None]
            + colors["j"] * ((one - fraction_x) * fraction_y)[..., None]
            + colors["k"] * (fraction_x * fraction_y)[..., None]
        )
        filtered[invalid] = bilinear[invalid]

    nearest_min = np.minimum.reduce(
        (colors["f"], colors["g"], colors["j"], colors["k"])
    )
    nearest_max = np.maximum.reduce(
        (colors["f"], colors["g"], colors["j"], colors["k"])
    )
    return np.clip(filtered, nearest_min, nearest_max).astype(np.float32, copy=False)


def _easu_tile(
    img: np.ndarray,
    y_start: int,
    y_stop: int,
    x_start: int,
    x_stop: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Render one tile using global coordinates and clamped source taps."""
    input_height, input_width = img.shape[:2]
    # Compute the affine map in float64 and snap mathematical integers. With
    # float32 alone, a 3x coordinate can land at 0.99999994 on one mirrored
    # side and 1.0 on the other, selecting a different 12-tap neighborhood.
    output_y = np.arange(y_start, y_stop, dtype=np.float64)[:, None]
    output_x = np.arange(x_start, x_stop, dtype=np.float64)[None, :]

    source_y = output_y * (input_height / output_height) + (
        0.5 * input_height / output_height - 0.5
    )
    source_x = output_x * (input_width / output_width) + (
        0.5 * input_width / output_width - 0.5
    )
    source_y = np.where(
        np.abs(source_y - np.rint(source_y)) < 1e-12,
        np.rint(source_y),
        source_y,
    )
    source_x = np.where(
        np.abs(source_x - np.rint(source_x)) < 1e-12,
        np.rint(source_x),
        source_x,
    )
    iy = np.floor(source_y).astype(np.intp)
    ix = np.floor(source_x).astype(np.intp)
    fraction_y = (source_y - iy).astype(np.float32)
    fraction_x = (source_x - ix).astype(np.float32)

    tile_shape = (y_stop - y_start, x_stop - x_start)
    iy = np.broadcast_to(iy, tile_shape)
    ix = np.broadcast_to(ix, tile_shape)
    fraction_y = np.broadcast_to(fraction_y, tile_shape)
    fraction_x = np.broadcast_to(fraction_x, tile_shape)
    result = _easu_grid(img, iy, ix, fraction_y, fraction_x)

    # The reference stencil is anchored to floor(pp), so pp==0 selects a
    # different one-sided tap set than the mathematically identical pp==1 at
    # the preceding texel. Average both limits only on exact integer source
    # coordinates. This removes scale-3 reflection seams without affecting the
    # regular reference path.
    boundary_x = np.flatnonzero(source_x[0] == np.rint(source_x[0]))
    boundary_y = np.flatnonzero(source_y[:, 0] == np.rint(source_y[:, 0]))
    if boundary_x.size:
        alternate_x = _easu_grid(
            img,
            iy[:, boundary_x],
            ix[:, boundary_x] - 1,
            fraction_y[:, boundary_x],
            np.ones_like(fraction_x[:, boundary_x]),
        )
        result[:, boundary_x] = (result[:, boundary_x] + alternate_x) * np.float32(0.5)
    if boundary_y.size:
        alternate_y = _easu_grid(
            img,
            iy[boundary_y] - 1,
            ix[boundary_y],
            np.ones_like(fraction_y[boundary_y]),
            fraction_x[boundary_y],
        )
        if boundary_x.size:
            alternate_xy = _easu_grid(
                img,
                iy[np.ix_(boundary_y, boundary_x)] - 1,
                ix[np.ix_(boundary_y, boundary_x)] - 1,
                np.ones_like(fraction_y[np.ix_(boundary_y, boundary_x)]),
                np.ones_like(fraction_x[np.ix_(boundary_y, boundary_x)]),
            )
            alternate_y[:, boundary_x] = (
                alternate_y[:, boundary_x] + alternate_xy
            ) * np.float32(0.5)
        result[boundary_y] = (result[boundary_y] + alternate_y) * np.float32(0.5)
    return result


def easu(
    img: np.ndarray,
    scale: float,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: CancelCallback | None = None,
    *,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Upscale RGB data with AMD FSR 1.0 EASU.

    ``scale`` must be finite and in [1, 4]. The returned dimensions are the
    input dimensions multiplied by ``scale`` and rounded to the nearest pixel.
    AMD documents good quality through 4x area scaling (2x per dimension);
    larger linear factors are supported here as explicit project extensions.
    """
    source = validate_rgb_image(img)
    scale = float(scale)
    if not np.isfinite(scale) or not 1.0 <= scale <= 4.0:
        raise ValueError("scale must be finite and in [1, 4]")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    output_height = round(source.shape[0] * scale)
    output_width = round(source.shape[1] * scale)
    if output_height < 1 or output_width < 1:
        raise ValueError("scale produces an empty output")

    check_cancelled(cancel_callback)
    if output_height == source.shape[0] and output_width == source.shape[1]:
        result = source.copy()
        if progress_callback is not None:
            progress_callback(1.0)
        return result

    output = np.empty((output_height, output_width, 3), dtype=np.float32)
    vertical_blocks = (output_height + block_size - 1) // block_size
    horizontal_blocks = (output_width + block_size - 1) // block_size
    total_blocks = vertical_blocks * horizontal_blocks
    completed = 0

    for y_start in range(0, output_height, block_size):
        y_stop = min(y_start + block_size, output_height)
        for x_start in range(0, output_width, block_size):
            check_cancelled(cancel_callback)
            x_stop = min(x_start + block_size, output_width)
            output[y_start:y_stop, x_start:x_stop] = _easu_tile(
                source,
                y_start,
                y_stop,
                x_start,
                x_stop,
                output_height,
                output_width,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed / total_blocks)

    return output
