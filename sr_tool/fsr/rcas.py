"""AMD FidelityFX Super Resolution 1.0 RCAS, ported to NumPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sr_tool.fsr.common import CancelCallback, check_cancelled, validate_rgb_image

# AMD reference: GPUOpen-Effects/FidelityFX-FSR, ffx-fsr/ffx_fsr1.h.
DEFAULT_SHARPNESS = 0.2
MAX_SHARPNESS_STOPS = 2.0
_RCAS_LIMIT = np.float32(0.25 - 1.0 / 16.0)
_RCAS_BLOCK = 512


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide without NaNs for the flat black/white 0/0 cases."""
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator != 0.0)
    return result


def _rcas_core(
    center: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    sharpness: float,
    denoise: bool = False,
) -> np.ndarray:
    """Vectorized translation of the reference ``FsrRcasF`` function."""
    ring_min = np.minimum.reduce((up, left, right, down))
    ring_max = np.maximum.reduce((up, left, right, down))

    hit_min = _safe_divide(
        np.minimum(ring_min, center),
        np.float32(4.0) * ring_max,
    )
    hit_max = _safe_divide(
        np.float32(1.0) - np.maximum(ring_max, center),
        np.float32(4.0) * ring_min - np.float32(4.0),
    )
    channel_lobes = np.maximum(-hit_min, hit_max)
    lobe = np.clip(
        np.max(channel_lobes, axis=2),
        -_RCAS_LIMIT,
        np.float32(0.0),
    )
    lobe *= np.float32(2.0 ** -sharpness)

    if denoise:
        luma_up = up[..., 1] + np.float32(0.5) * (up[..., 0] + up[..., 2])
        luma_left = left[..., 1] + np.float32(0.5) * (left[..., 0] + left[..., 2])
        luma_center = center[..., 1] + np.float32(0.5) * (
            center[..., 0] + center[..., 2]
        )
        luma_right = right[..., 1] + np.float32(0.5) * (
            right[..., 0] + right[..., 2]
        )
        luma_down = down[..., 1] + np.float32(0.5) * (down[..., 0] + down[..., 2])
        luma_max = np.maximum.reduce(
            (luma_up, luma_left, luma_center, luma_right, luma_down)
        )
        luma_min = np.minimum.reduce(
            (luma_up, luma_left, luma_center, luma_right, luma_down)
        )
        noise = np.abs(
            np.float32(0.25)
            * (luma_up + luma_left + luma_right + luma_down)
            - luma_center
        )
        noise = np.clip(_safe_divide(noise, luma_max - luma_min), 0.0, 1.0)
        lobe *= np.float32(1.0) - np.float32(0.5) * noise

    neighbor_sum = up + left + right + down
    denominator = np.float32(4.0) * lobe + np.float32(1.0)
    result = (lobe[..., None] * neighbor_sum + center) / denominator[..., None]
    # The reference limiter keeps values in gamut. Clip only float round-off.
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def rcas(
    img: np.ndarray,
    sharpness: float = DEFAULT_SHARPNESS,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: CancelCallback | None = None,
    *,
    denoise: bool = False,
    block_size: int = _RCAS_BLOCK,
) -> np.ndarray:
    """Sharpen an RGB image with AMD FSR 1.0 RCAS.

    ``sharpness`` uses AMD's stop scale: 0 is maximum sharpening and larger
    values halve the effect once per stop. This application exposes [0, 2].
    ``denoise`` enables the optional, more expensive reference path.
    """
    source = validate_rgb_image(img)
    sharpness = float(sharpness)
    if not np.isfinite(sharpness) or not 0.0 <= sharpness <= MAX_SHARPNESS_STOPS:
        raise ValueError("sharpness must be finite and in [0, 2] stops")
    if isinstance(block_size, bool) or not isinstance(block_size, (int, np.integer)):
        raise TypeError("block_size must be an integer")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    check_cancelled(cancel_callback)
    height, width = source.shape[:2]
    output = np.empty_like(source)
    vertical_blocks = (height + block_size - 1) // block_size
    horizontal_blocks = (width + block_size - 1) // block_size
    total_blocks = vertical_blocks * horizontal_blocks
    completed = 0

    for y_start in range(0, height, block_size):
        check_cancelled(cancel_callback)
        y_stop = min(y_start + block_size, height)
        y = np.arange(y_start, y_stop, dtype=np.intp)
        y_up = np.maximum(y - 1, 0)
        y_down = np.minimum(y + 1, height - 1)
        for x_start in range(0, width, block_size):
            check_cancelled(cancel_callback)
            x_stop = min(x_start + block_size, width)
            x = np.arange(x_start, x_stop, dtype=np.intp)
            x_left = np.maximum(x - 1, 0)
            x_right = np.minimum(x + 1, width - 1)
            center = source[y_start:y_stop, x_start:x_stop]
            left = source[y[:, None], x_left[None, :]]
            right = source[y[:, None], x_right[None, :]]
            up = source[y_up[:, None], x[None, :]]
            down = source[y_down[:, None], x[None, :]]
            output[y_start:y_stop, x_start:x_stop] = _rcas_core(
                center,
                left,
                right,
                up,
                down,
                sharpness,
                denoise,
            )
            completed += 1
            if progress_callback is not None:
                progress_callback(completed / total_blocks)

    return output
