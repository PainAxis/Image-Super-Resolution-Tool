"""Tests for FXAA module."""

import numpy as np
import pytest
from sr_tool.fsr.antialias import fxaa


def test_fxaa_shape_preserved():
    img = np.random.rand(32, 32, 3).astype(np.float32)
    result = fxaa(img)
    assert result.shape == img.shape


def test_fxaa_output_range():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = fxaa(img)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-5


def test_fxaa_no_nan():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = fxaa(img)
    assert not np.any(np.isnan(result))


def test_fxaa_flat_image_unchanged():
    img = np.full((16, 16, 3), 0.5, dtype=np.float32)
    result = fxaa(img)
    np.testing.assert_array_almost_equal(result, img, decimal=4)


def test_fxaa_small_image():
    img = np.ones((1, 1, 3), dtype=np.float32) * 0.5
    result = fxaa(img)
    assert result.shape == (1, 1, 3)


def test_fxaa_edge_no_nan():
    """FXAA on an edge image should not produce NaN."""
    img = np.ones((32, 32, 3), dtype=np.float32) * 0.2
    img[:, 16:] = 0.9
    result = fxaa(img)
    assert not np.any(np.isnan(result))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_fxaa_no_warnings_on_flat():
    """FXAA should not raise warnings on flat images."""
    img = np.full((16, 16, 3), 0.3, dtype=np.float32)
    with np.errstate(all="raise"):
        try:
            result = fxaa(img)
            assert result.shape == img.shape
        except FloatingPointError as e:
            pytest.fail(f"FXAA raised floating-point error on flat image: {e}")
