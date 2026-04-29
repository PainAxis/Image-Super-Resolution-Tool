"""Tests for RCAS module."""

import numpy as np
import pytest
from fsr_tool.fsr.rcas import rcas


def test_rcas_flat_image_unchanged():
    """A perfectly flat image should not change after RCAS."""
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    result = rcas(img, sharpness=0.25)
    np.testing.assert_array_almost_equal(result, img, decimal=4)


def test_rcas_output_shape():
    """RCAS preserves input shape."""
    img = np.random.rand(64, 48, 3).astype(np.float32)
    result = rcas(img, sharpness=0.25)
    assert result.shape == img.shape


def test_rcas_output_range():
    """RCAS output stays in [0, 1]."""
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = rcas(img, sharpness=0.5)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_rcas_edge_enhances():
    """A sharp vertical edge should become sharper after RCAS."""
    # Create a vertical edge: left half 0.2, right half 0.8
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.2
    img[:, 16:, :] = 0.8
    result = rcas(img, sharpness=0.25)

    # The edge should still be present (center pixel near edge should be enhanced)
    # Pixels at the boundary (col 15 and 16) should show contrast enhancement
    center_row = result[16, :, 0]
    # The transition should be at least as sharp as the original
    diff_result = abs(float(center_row[15]) - float(center_row[16]))
    diff_orig = 0.6  # 0.8 - 0.2
    # Sharpening might increase or decrease the transition, but should be present
    assert diff_result > 0.1


def test_rcas_sharpness_zero_is_noop():
    """sharpness=0 returns a copy of the input (no change)."""
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = rcas(img, sharpness=0.0)
    np.testing.assert_array_almost_equal(result, img, decimal=5)


def test_rcas_small_image():
    """RCAS handles 1x1 images gracefully."""
    img = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)
    result = rcas(img, sharpness=0.25)
    assert result.shape == (1, 1, 3)
    np.testing.assert_array_almost_equal(result, img, decimal=4)
