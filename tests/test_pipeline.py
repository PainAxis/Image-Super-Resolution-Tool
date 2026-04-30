"""Integration tests for super-resolution pipeline."""

import numpy as np
import pytest
from sr_tool.fsr.pipeline import process_image


def test_pipeline_output_shape():
    img = np.random.rand(24, 32, 3).astype(np.float32)
    for scale in [2, 3, 4]:
        result = process_image(img, scale)
        assert result.shape == (24 * scale, 32 * scale, 3)


def test_pipeline_output_range():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = process_image(img, 2)
    assert result.min() >= 0.0
    assert result.max() <= 1.0 + 1e-5


def test_pipeline_no_nan():
    img = np.random.rand(16, 16, 3).astype(np.float32)
    result = process_image(img, 2)
    assert not np.any(np.isnan(result))


def test_pipeline_flat_image():
    """Flat image should remain approximately flat after pipeline."""
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    result = process_image(img, 2)
    np.testing.assert_allclose(result, 0.5, atol=0.05)
