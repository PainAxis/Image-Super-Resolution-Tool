"""EASU -- Edge-Adaptive Spatial Upsampling (super-resolution pass 1).

Upscales an image by the given factor, preserving edges by adapting the
interpolation kernel to the local gradient direction. The kernel stretches
along edges and shrinks across them, reducing blur.

Implementation uses NumPy vectorization with block-level processing.
"""

from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOCK_SIZE = 64          # output pixels per block side
PAD_PX = 2               # border padding for neighborhood access
GRADIENT_EPS = 1e-6      # minimum gradient magnitude before default direction
ASPECT_MIN = 0.25        # minimum kernel anisotropy ratio
ASPECT_MAX = 4.0         # maximum kernel anisotropy ratio
ASPECT_SCALE = 10.0      # gradient→anisotropy scaling factor


# ---------------------------------------------------------------------------
# 12 fixed tap offsets (before rotation by local gradient direction)
# These form 4 concentric rings around the source sample point.
# ---------------------------------------------------------------------------
_TAP_OFFSETS = np.array([
    # Ring 0 (radius ~0.5): 2 taps along edge direction
    [-0.5,  0.0], [0.5, 0.0],
    # Ring 1 (radius ~1.0): 2 along dir, 2 along perp
    [-1.0,  0.0], [1.0, 0.0],
    [ 0.0, -1.0], [0.0, 1.0],
    # Ring 2 (radius ~1.5): 4 taps at diagonal-ish offsets
    [-0.5, -1.5], [-0.5, 1.5],
    [ 0.5, -1.5], [ 0.5, 1.5],
    # Ring 3 (radius ~2.0): 2 taps along edge direction
    [-2.0,  0.0], [2.0, 0.0],
], dtype=np.float32)


def _lanczos2(x):
    """Lanczos2 kernel: sinc(x)*sinc(x/2), support radius 2.

    Handles both scalars and NumPy arrays.
    """
    x = np.abs(x)
    out = np.zeros_like(x)
    mask = x < 2.0
    xm = x[mask]
    out[mask] = np.sinc(xm) * np.sinc(xm / 2.0)
    return out


def easu(img: np.ndarray,
         scale: float,
         progress_callback: Optional[Callable[[float], None]] = None
         ) -> np.ndarray:
    """Apply EASU upscaling.

    Args:
        img: float32 (H, W, 3) array in [0, 1].
        scale: upscale factor (e.g. 2, 3, 4).
        progress_callback: optional callable(fraction: float).

    Returns:
        float32 (H*scale, W*scale, 3) array in [0, 1].
    """
    h_in, w_in = img.shape[:2]
    c = img.shape[2]
    h_out = int(h_in * scale)
    w_out = int(w_in * scale)

    if scale == 1.0:
        return img.copy()

    # Pad input for border handling
    padded = np.pad(img, ((PAD_PX, PAD_PX), (PAD_PX, PAD_PX), (0, 0)), mode="edge")
    hp, wp = padded.shape[:2]

    # Create sliding-window view of all 4x4 patches.
    # padded shape: (H+4, W+4, 3)  →  patches shape: (H+1, W+1, 4, 4, 3)
    from numpy.lib.stride_tricks import sliding_window_view
    patches = sliding_window_view(padded, (4, 4, c)).reshape(
        hp - 3, wp - 3, 4, 4, c)

    output = np.zeros((h_out, w_out, c), dtype=np.float32)

    # Block processing
    n_by = (h_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    n_bx = (w_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = n_by * n_bx
    block_idx = 0

    for by in range(0, h_out, BLOCK_SIZE):
        bh = min(BLOCK_SIZE, h_out - by)
        for bx in range(0, w_out, BLOCK_SIZE):
            bw = min(BLOCK_SIZE, w_out - bx)

            # --- source coordinates for this block (vectorized) ---
            # sy: (bh, 1),  sx: (1, bw)
            sy = ((np.arange(by, by + bh, dtype=np.float32).reshape(-1, 1) + 0.5)
                  * h_in / h_out - 0.5)
            sx = ((np.arange(bx, bx + bw, dtype=np.float32).reshape(1, -1) + 0.5)
                  * w_in / w_out - 0.5)

            iy = np.floor(sy).astype(np.int32)   # (bh, 1)
            ix = np.floor(sx).astype(np.int32)   # (1, bw)
            fy = (sy - iy).astype(np.float32)    # (bh, 1)
            fx = (sx - ix).astype(np.float32)    # (1, bw)

            # Broadcast to (bh, bw)
            iy_b = iy + np.zeros((1, bw), dtype=np.int32)
            ix_b = ix + np.zeros((bh, 1), dtype=np.int32)
            fy_b = fy + np.zeros((1, bw), dtype=np.float32)
            fx_b = fx + np.zeros((bh, 1), dtype=np.float32)

            # Padded coordinates: +2 for the 2px border
            py_b = iy_b + 2
            px_b = ix_b + 2

            # --- extract 4x4 patches for all pixels (vectorized) ---
            # patches shape: (H+1, W+1, 4, 4, 3)
            # For each pixel, grab the 4x4 window at (py_b-1, px_b-1)
            # This gives (bh, bw, 4, 4, 3)
            block_patches = patches[py_b - 1, px_b - 1]

            # --- gradient estimation (vectorized) ---
            # Luma from RGB
            luma = (block_patches[:, :, :, :, 0] * 0.2126 +
                    block_patches[:, :, :, :, 1] * 0.7152 +
                    block_patches[:, :, :, :, 2] * 0.0722)  # (bh, bw, 4, 4)

            # Sobel-like gradient from the inner 2x2 of the 4x4 patch
            # Inner positions: (1,1), (1,2), (2,1), (2,2)
            dx = ((luma[:, :, 1, 2] - luma[:, :, 1, 1]) +
                  (luma[:, :, 2, 2] - luma[:, :, 2, 1]))  # (bh, bw)
            dy = ((luma[:, :, 2, 1] - luma[:, :, 1, 1]) +
                  (luma[:, :, 2, 2] - luma[:, :, 1, 2]))  # (bh, bw)

            grad_len = np.sqrt(dx * dx + dy * dy)  # (bh, bw)

            # Direction vectors (default horizontal for flat regions)
            flat_mask = grad_len < GRADIENT_EPS
            dx_safe = np.where(flat_mask, 1.0, dx)
            dy_safe = np.where(flat_mask, 0.0, dy)
            grad_len_safe = np.where(flat_mask, 1.0, grad_len)

            dir_x = dx_safe / grad_len_safe    # (bh, bw)
            dir_y = dy_safe / grad_len_safe    # (bh, bw)
            perp_x = -dir_y                    # (bh, bw)
            perp_y = dir_x                     # (bh, bw)

            # Aspect ratio: stretch kernel along edges for strong gradients
            aspect = np.clip(grad_len * ASPECT_SCALE, ASPECT_MIN, ASPECT_MAX)

            # Pre-compute flat-index scaffolding (reused across 12 taps + fallback)
            flat_n = bh * bw
            flat_patches = block_patches.reshape(flat_n, 4, 4, c)
            flat_idx = np.arange(flat_n, dtype=np.int32)

            # --- accumulate 12 taps (loop over taps, vectorized inside) ---
            color_acc = np.zeros((bh, bw, c), dtype=np.float32)
            weight_sum = np.zeros((bh, bw), dtype=np.float32)

            for tap_ox, tap_oy in _TAP_OFFSETS:
                # Rotate tap offset by direction and perpendicular
                rx = tap_ox * dir_x + tap_oy * perp_x  # (bh, bw)
                ry = tap_ox * dir_y + tap_oy * perp_y  # (bh, bw)

                # Anisotropic distance
                along = rx * dir_x + ry * dir_y        # (bh, bw)
                across = rx * perp_x + ry * perp_y     # (bh, bw)
                aniso_d = np.sqrt(along * along +
                                  (across * aspect) * (across * aspect))

                # Lanczos2 weight
                w = _lanczos2(aniso_d)  # (bh, bw), 0 for d >= 2

                # Bilinear sample at (px_b + rx, py_b + ry)
                # rx, ry are in source-pixel units
                sample_x = fx_b + rx     # fractional within the 4x4 patch
                sample_y = fy_b + ry     # (bh, bw)

                # Map to 4x4 patch coordinate system:
                #   patch[0]=src[ix-1], patch[1]=src[ix], patch[2]=src[ix+1], patch[3]=src[ix+2]
                #   The integer source pixel ix lives at patch index 1.0.
                #   So: patch_coord = 1.0 + fx_b + tap_offset
                sp_x = sample_x + 1.0    # (bh, bw), now in patch coords
                sp_y = sample_y + 1.0

                # Clamp to patch bounds [0, 3]
                sp_x_clamped = np.clip(sp_x, 0.0, 3.0 - 1e-6)
                sp_y_clamped = np.clip(sp_y, 0.0, 3.0 - 1e-6)

                sx0 = np.floor(sp_x_clamped).astype(np.int32)
                sy0 = np.floor(sp_y_clamped).astype(np.int32)
                sfx = sp_x_clamped - sx0.astype(np.float32)
                sfy = sp_y_clamped - sy0.astype(np.float32)

                sx1 = np.clip(sx0 + 1, 0, 3)
                sy1 = np.clip(sy0 + 1, 0, 3)

                # ---- Bilinear interpolation on the 4x4 patch (1D flat-indexed) ----
                f_sy0 = sy0.ravel()
                f_sx0 = sx0.ravel()
                f_sy1 = sy1.ravel()
                f_sx1 = sx1.ravel()

                fp00 = flat_patches[flat_idx, f_sy0, f_sx0, :]  # (flat_n, 3)
                fp10 = flat_patches[flat_idx, f_sy0, f_sx1, :]
                fp01 = flat_patches[flat_idx, f_sy1, f_sx0, :]
                fp11 = flat_patches[flat_idx, f_sy1, f_sx1, :]

                p00 = fp00.reshape(bh, bw, 3)
                p10 = fp10.reshape(bh, bw, 3)
                p01 = fp01.reshape(bh, bw, 3)
                p11 = fp11.reshape(bh, bw, 3)

                # Blend
                sfx = sfx[:, :, None]  # (bh, bw, 1)
                sfy = sfy[:, :, None]
                top = p00 * (1 - sfx) + p10 * sfx
                bottom = p01 * (1 - sfx) + p11 * sfx
                sample = top * (1 - sfy) + bottom * sfy  # (bh, bw, 3)

                w3 = w[:, :, None]  # (bh, bw, 1)
                color_acc += sample * w3
                weight_sum += w

            # ---- Normalize with fallback ----
            valid = weight_sum > 0.0

            # Fallback bilinear coords: sample at source pixel center
            #   patch index 1.0 = source pixel ix, so coord = 1.0 + fx
            sy_c = fy_b.ravel() + 1.0
            sx_c = fx_b.ravel() + 1.0
            sx_c = np.clip(sx_c, 0.0, 3.0 - 1e-6)
            sy_c = np.clip(sy_c, 0.0, 3.0 - 1e-6)
            fisx = np.floor(sx_c).astype(np.int32)
            fisy = np.floor(sy_c).astype(np.int32)
            fsfx = sx_c - fisx.astype(np.float32)
            fsfy = sy_c - fisy.astype(np.float32)
            fisx1 = np.clip(fisx + 1, 0, 3)
            fisy1 = np.clip(fisy + 1, 0, 3)

            for ch in range(c):
                ch_out = output[by:by + bh, bx:bx + bw, ch]
                ch_acc = color_acc[:, :, ch]
                ch_acc[valid] /= weight_sum[valid]

                # Fallback: 1D-indexed bilinear sample at patch center
                ff00 = flat_patches[flat_idx, fisy, fisx, ch]
                ff10 = flat_patches[flat_idx, fisy, fisx1, ch]
                ff01 = flat_patches[flat_idx, fisy1, fisx, ch]
                ff11 = flat_patches[flat_idx, fisy1, fisx1, ch]

                ftop = ff00 * (1 - fsfx) + ff10 * fsfx
                fbot = ff01 * (1 - fsfx) + ff11 * fsfx
                ch_fallback = (ftop * (1 - fsfy) + fbot * fsfy).reshape(bh, bw)

                ch_out[~valid] = ch_fallback[~valid]
                ch_out[valid] = ch_acc[valid]

            # Report progress
            block_idx += 1
            if progress_callback:
                progress_callback(block_idx / total_blocks)

    return output
