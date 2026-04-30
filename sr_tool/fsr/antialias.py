"""FXAA -- Fast Approximate Anti-Aliasing (post-processing).

A lightweight anti-aliasing filter that smooths jagged edges by detecting
luma discontinuities and blending pixels along edge directions.

Based on FXAA 3.11 by Timothy Lottes (NVIDIA), adapted to NumPy.
Processes images in blocks to keep peak memory bounded.
"""

import numpy as np

# FXAA quality presets
_EDGE_THRESHOLD = 0.0833       # minimum luma contrast to consider an edge
_EDGE_THRESHOLD_MIN = 0.0312   # minimum for dark areas
_SUBPIX_QUALITY = 0.75         # sub-pixel AA strength [0, 1]
_ITERATIONS = 12               # endpoint search iterations

# Block processing
_FXAA_BLOCK = 512              # output pixels per block side
_FXAA_OVERLAP = 16             # 12x search + 1px neighbor + 3px margin


def _rgb2luma(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB float32 array to luma (BT.709)."""
    return (rgb[:, :, 0] * 0.2126 +
            rgb[:, :, 1] * 0.7152 +
            rgb[:, :, 2] * 0.0722)


def _safe_int32(arr: np.ndarray) -> np.ndarray:
    """Safely cast float array to int32, guarding against NaN/Inf."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.int32)


def _sample_luma(luma: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Bilinearly sample luma at sub-pixel positions (y, x)."""
    h, w = luma.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)

    ix = _safe_int32(np.floor(x))
    iy = _safe_int32(np.floor(y))
    fx = x - ix.astype(np.float32)
    fy = y - iy.astype(np.float32)

    ix1 = np.clip(ix + 1, 0, w - 1)
    iy1 = np.clip(iy + 1, 0, h - 1)

    top = luma[iy, ix] * (1 - fx) + luma[iy, ix1] * fx
    bottom = luma[iy1, ix] * (1 - fx) + luma[iy1, ix1] * fx
    return top * (1 - fy) + bottom * fy


# ---------------------------------------------------------------------------
# Core FXAA (per-image or per-block, no chunking)
# ---------------------------------------------------------------------------
def _fxaa_core(img: np.ndarray,
               edge_threshold: float = _EDGE_THRESHOLD,
               edge_threshold_min: float = _EDGE_THRESHOLD_MIN,
               subpix_quality: float = _SUBPIX_QUALITY,
               iterations: int = _ITERATIONS) -> np.ndarray:
    """Apply FXAA to a (H, W, 3) image in [0, 1]."""

    h, w = img.shape[:2]
    luma = _rgb2luma(img)

    lp = np.pad(luma, ((1, 1), (1, 1)), mode="edge")
    luma_n  = lp[0:h,     1:w + 1]
    luma_s  = lp[2:h + 2, 1:w + 1]
    luma_w  = lp[1:h + 1, 0:w]
    luma_e  = lp[1:h + 1, 2:w + 2]
    luma_nw = lp[0:h,     0:w]
    luma_ne = lp[0:h,     2:w + 2]
    luma_sw = lp[2:h + 2, 0:w]
    luma_se = lp[2:h + 2, 2:w + 2]

    luma_min = np.minimum(np.minimum(np.minimum(luma_n, luma_s),
                                     np.minimum(luma_w, luma_e)),
                          np.minimum(np.minimum(luma_nw, luma_ne),
                                     np.minimum(luma_sw, luma_se)))
    luma_max = np.maximum(np.maximum(np.maximum(luma_n, luma_s),
                                     np.maximum(luma_w, luma_e)),
                          np.maximum(np.maximum(luma_nw, luma_ne),
                                     np.maximum(luma_sw, luma_se)))
    contrast = luma_max - luma_min

    threshold = np.maximum(edge_threshold * luma_max, edge_threshold_min)
    edge_mask = contrast > threshold

    horizontal = (np.abs(luma_n + luma_s - 2 * luma) * 2 +
                  np.abs(luma_ne + luma_se - 2 * luma_e) +
                  np.abs(luma_nw + luma_sw - 2 * luma_w))
    vertical = (np.abs(luma_e + luma_w - 2 * luma) * 2 +
                np.abs(luma_ne + luma_nw - 2 * luma_n) +
                np.abs(luma_se + luma_sw - 2 * luma_s))
    is_horizontal = edge_mask & (horizontal >= vertical)

    luma_avg_neighbors = (luma_n + luma_s + luma_w + luma_e) / 4.0
    subpix_amount = np.clip(
        (np.abs(luma_avg_neighbors - luma) / (contrast + 1e-9)) * subpix_quality,
        0.0, 1.0)
    subpix_amount = np.where(contrast > 1e-6, subpix_amount, 0.0)

    # ---- endpoint search (edge pixels only) ----
    yy_full, xx_full = np.mgrid[0:h, 0:w].astype(np.float32)
    dir_x = np.where(is_horizontal, 1.0, 0.0)
    dir_y = np.where(is_horizontal, 0.0, 1.0)

    pos_end_luma = np.zeros_like(luma)
    neg_end_luma = np.zeros_like(luma)
    search_luma_pos = np.where(edge_mask, luma, 0.0)
    search_luma_neg = np.where(edge_mask, luma, 0.0)

    need_pos = edge_mask.copy()
    need_neg = edge_mask.copy()

    for i in range(iterations):
        if not np.any(need_pos) and not np.any(need_neg):
            break
        dist = (i + 1) * 1.0

        if np.any(need_pos):
            sy_pos = np.where(need_pos, yy_full + dir_y * dist, 0.0)
            sx_pos = np.where(need_pos, xx_full + dir_x * dist, 0.0)
            sample_pos = _sample_luma(luma, sy_pos, sx_pos)
            pos_beyond = (sample_pos - luma) * (search_luma_pos - luma) < 0
            found_pos = need_pos & pos_beyond
            pos_end_luma = np.where(found_pos, sample_pos, pos_end_luma)
            need_pos = need_pos & ~found_pos
            search_luma_pos = np.where(need_pos, sample_pos, search_luma_pos)

        if np.any(need_neg):
            sy_neg = np.where(need_neg, yy_full - dir_y * dist, 0.0)
            sx_neg = np.where(need_neg, xx_full - dir_x * dist, 0.0)
            sample_neg = _sample_luma(luma, sy_neg, sx_neg)
            neg_beyond = (sample_neg - luma) * (search_luma_neg - luma) < 0
            found_neg = need_neg & neg_beyond
            neg_end_luma = np.where(found_neg, sample_neg, neg_end_luma)
            need_neg = need_neg & ~found_neg
            search_luma_neg = np.where(need_neg, sample_neg, search_luma_neg)

    pos_end_set = pos_end_luma != 0
    neg_end_set = neg_end_luma != 0

    # ---- blend amount (edge sub-pixel offset) ----
    edge_offset: np.ndarray = np.zeros_like(luma)
    if subpix_quality > 0:
        edge_test = ((luma_e - luma_w) * 0.5 +
                     (luma_se - luma_sw) * 0.25 +
                     (luma_ne - luma_nw) * 0.25)
        edge_offset = np.where(
            np.abs(edge_test) > 1e-6,
            np.clip((luma - luma_w) / (edge_test + 1e-9), -1.0, 1.0) * 0.5,
            edge_offset)

    blend_amount = np.zeros_like(luma)
    blend_amount = np.where(
        edge_mask & pos_end_set & neg_end_set,
        np.clip((pos_end_luma - luma) /
                (pos_end_luma - neg_end_luma + 1e-9) - 0.5 + edge_offset,
                -0.5, 0.5),
        blend_amount)

    blend_sign = np.sign(blend_amount)
    blend_amount = blend_sign * np.maximum(
        np.abs(blend_amount), subpix_amount * subpix_quality)
    blend_amount = np.clip(blend_amount + 0.5, -0.5, 0.5) + 0.5
    blend_amount = np.clip(blend_amount, 0.0, 1.0)

    # ---- apply blending ----
    result = img.copy()
    for ch in range(3):
        ch_luma = img[:, :, ch]
        sample_y = yy_full + np.where(is_horizontal, 0.5, 0.0)
        sample_x = xx_full + np.where(is_horizontal, 0.0, 0.5)
        ch_padded = np.pad(ch_luma, ((1, 1), (1, 1)), mode="edge")
        sample_p = _sample_luma(ch_padded, sample_y + 1, sample_x + 1)
        sample_n = _sample_luma(ch_padded,
                                yy_full - np.where(is_horizontal, 0.5, 0.0) + 1,
                                xx_full - np.where(is_horizontal, 0.0, 0.5) + 1)
        blended = sample_n * blend_amount + sample_p * (1 - blend_amount)
        result[:, :, ch] = np.where(edge_mask, blended, ch_luma)

    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public FXAA with block processing for memory efficiency
# ---------------------------------------------------------------------------
def fxaa(img: np.ndarray,
         edge_threshold: float = _EDGE_THRESHOLD,
         edge_threshold_min: float = _EDGE_THRESHOLD_MIN,
         subpix_quality: float = _SUBPIX_QUALITY,
         iterations: int = _ITERATIONS) -> np.ndarray:
    """Apply FXAA anti-aliasing with block processing.

    Args:
        img: float32 (H, W, 3) in [0, 1].
        edge_threshold: luma contrast threshold for edge detection.
        edge_threshold_min: minimum threshold for very dark areas.
        subpix_quality: sub-pixel AA strength [0, 1]. 0 = off.
        iterations: endpoint search steps.

    Returns:
        float32 (H, W, 3) anti-aliased image in [0, 1].
    """
    h, w = img.shape[:2]

    # Small images: process whole
    if h <= _FXAA_BLOCK and w <= _FXAA_BLOCK:
        return _fxaa_core(img, edge_threshold, edge_threshold_min,
                          subpix_quality, iterations)

    # Large images: block processing with overlap
    OV = _FXAA_OVERLAP
    padded = np.pad(img, ((OV, OV), (OV, OV), (0, 0)), mode="edge")
    output = np.empty_like(img)

    for by in range(0, h, _FXAA_BLOCK):
        bh = min(_FXAA_BLOCK, h - by)
        for bx in range(0, w, _FXAA_BLOCK):
            bw = min(_FXAA_BLOCK, w - bx)

            py0, py1 = by, by + bh + 2 * OV
            px0, px1 = bx, bx + bw + 2 * OV
            patch = padded[py0:py1, px0:px1, :]

            result_patch = _fxaa_core(patch, edge_threshold, edge_threshold_min,
                                      subpix_quality, iterations)
            # Extract interior: discard overlap border
            interior = result_patch[OV:OV + bh, OV:OV + bw, :]
            output[by:by + bh, bx:bx + bw, :] = interior

    return output
