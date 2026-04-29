"""RCAS -- Robust Contrast-Adaptive Sharpening (FSR 1.0 pass 2).

Applies adaptive sharpening that respects local contrast:
- Stronger sharpening in high-contrast (textured) areas
- Weaker sharpening in low-contrast (flat) areas to avoid amplifying noise
- Clamps output to local neighborhood range to prevent ringing artifacts.

Processes the image in blocks to keep peak memory low on large outputs.
"""

from typing import Callable, Optional

import numpy as np

# RCAS algorithm constants (from AMD FSR 1.0 specification)
_NEIGHBOR_COUNT = 4.0     # cardinal neighbors: left, right, up, down
_GAIN_DENOM_SCALE = 8.0   # denominator scaling for adaptive gain
_EPSILON = 1e-9           # prevents division by zero in flat regions

DEFAULT_SHARPNESS = 0.25  # AMD-specified default sharpening strength

# Block processing
_RCAS_BLOCK = 512          # output pixels per block side
_RCAS_OVERLAP = 2          # 1px neighbor + 1px safety margin


def _rcas_core(center: np.ndarray,
               left: np.ndarray,
               right: np.ndarray,
               up: np.ndarray,
               down: np.ndarray,
               sharpness: float) -> np.ndarray:
    """Core RCAS computation on pre-extracted neighbor views.

    All inputs are float32 (H, W, 3) in [0, 1].
    """
    n_sum = left + right + up + down
    n_min = np.minimum(np.minimum(left, right), np.minimum(up, down))
    n_max = np.maximum(np.maximum(left, right), np.maximum(up, down))
    n_avg = n_sum / _NEIGHBOR_COUNT

    contrast = n_max - n_min
    deviation = np.abs(_NEIGHBOR_COUNT * center - n_sum)

    gain = 1.0 - deviation / (_GAIN_DENOM_SCALE * contrast + _EPSILON)
    gain = np.clip(gain, 0.0, 1.0)

    sharpened = center + sharpness * gain * (center - n_avg)
    sharpened = np.clip(sharpened, n_min, n_max)
    return np.clip(sharpened, 0.0, 1.0)


def rcas(img: np.ndarray,
         sharpness: float = DEFAULT_SHARPNESS,
         progress_callback: Optional[Callable[[float], None]] = None,
         ) -> np.ndarray:
    """Apply RCAS sharpening to an image.

    Args:
        img: float32 (H, W, 3) array in [0, 1].
        sharpness: sharpening strength, range [0, 1]. Default 0.25 (AMD spec).
        progress_callback: optional callable(fraction: float), 0→1 within RCAS.

    Returns:
        float32 (H, W, 3) array in [0, 1].
    """
    sharpness = float(sharpness)
    if sharpness <= 0.0:
        return img.copy()

    h, w = img.shape[:2]
    _cb = progress_callback

    # For small images, process whole image at once
    if h <= _RCAS_BLOCK and w <= _RCAS_BLOCK:
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
        center = img
        left   = padded[1:h + 1, 0:w, :]
        right  = padded[1:h + 1, 2:w + 2, :]
        up     = padded[0:h, 1:w + 1, :]
        down   = padded[2:h + 2, 1:w + 1, :]
        result = _rcas_core(center, left, right, up, down, sharpness)
        if _cb is not None:
            _cb(1.0)
        return result

    # Block processing for large images
    padded = np.pad(img, ((_RCAS_OVERLAP, _RCAS_OVERLAP),
                           (_RCAS_OVERLAP, _RCAS_OVERLAP),
                           (0, 0)), mode="edge")
    output = np.empty_like(img)

    n_by = (h + _RCAS_BLOCK - 1) // _RCAS_BLOCK
    n_bx = (w + _RCAS_BLOCK - 1) // _RCAS_BLOCK
    total_blocks = n_by * n_bx
    block_idx = 0

    for by in range(0, h, _RCAS_BLOCK):
        bh = min(_RCAS_BLOCK, h - by)
        for bx in range(0, w, _RCAS_BLOCK):
            bw = min(_RCAS_BLOCK, w - bx)

            # Source region in padded image (with overlap for neighbor access)
            py0 = by
            py1 = by + bh + 2 * _RCAS_OVERLAP
            px0 = bx
            px1 = bx + bw + 2 * _RCAS_OVERLAP

            patch = padded[py0:py1, px0:px1, :]
            ph, pw = patch.shape[:2]

            center = patch[_RCAS_OVERLAP:ph - _RCAS_OVERLAP,
                           _RCAS_OVERLAP:pw - _RCAS_OVERLAP, :]
            left   = patch[_RCAS_OVERLAP:ph - _RCAS_OVERLAP,
                           _RCAS_OVERLAP - 1:pw - _RCAS_OVERLAP - 1, :]
            right  = patch[_RCAS_OVERLAP:ph - _RCAS_OVERLAP,
                           _RCAS_OVERLAP + 1:pw - _RCAS_OVERLAP + 1, :]
            up     = patch[_RCAS_OVERLAP - 1:ph - _RCAS_OVERLAP - 1,
                           _RCAS_OVERLAP:pw - _RCAS_OVERLAP, :]
            down   = patch[_RCAS_OVERLAP + 1:ph - _RCAS_OVERLAP + 1,
                           _RCAS_OVERLAP:pw - _RCAS_OVERLAP, :]

            output[by:by + bh, bx:bx + bw, :] = _rcas_core(
                center, left, right, up, down, sharpness)

            block_idx += 1
            if _cb is not None:
                _cb(block_idx / total_blocks)

    return output
