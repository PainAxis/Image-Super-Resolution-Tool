"""Tests for EASU module."""

import numpy as np
import pytest
from fsr_tool.fsr.easu import easu, _lanczos2


class TestLanczos2:
    def test_at_zero(self):
        assert _lanczos2(0.0) == 1.0

    def test_at_one(self):
        val = _lanczos2(1.0)
        # sinc(1)*sinc(0.5) = 0 * (sin(pi/2)/(pi/2))
        # sinc(1) = sin(pi)/(pi) = 0
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_out_of_support(self):
        assert _lanczos2(2.0) == 0.0
        assert _lanczos2(3.0) == 0.0
        assert _lanczos2(-2.5) == 0.0

    def test_array_input(self):
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.0], dtype=np.float32)
        result = _lanczos2(x)
        assert result.shape == (5,)
        assert result[0] == 1.0
        assert result[3] == 0.0
        assert result[4] == 0.0


class TestEASU:
    def test_output_shape(self):
        img = np.random.rand(32, 32, 3).astype(np.float32)
        for scale in [1, 2, 3, 4]:
            result = easu(img, scale)
            assert result.shape == (32 * scale, 32 * scale, 3)

    def test_output_range(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        result = easu(img, 2)
        # Lanczos2 kernel has negative lobes; allow tiny undershoot
        assert result.min() >= -0.02
        assert result.max() <= 1.0 + 0.02

    def test_scale_one_is_preserving(self):
        img = np.random.rand(8, 8, 3).astype(np.float32)
        result = easu(img, 1.0)
        # Should return the same shape
        assert result.shape == img.shape

    def test_flat_image_stays_flat(self):
        """A uniformly-colored image should stay uniform after upscaling."""
        for color in [0.0, 0.5, 1.0]:
            img = np.full((8, 8, 3), color, dtype=np.float32)
            result = easu(img, 2)
            # Allow small floating-point deviation
            np.testing.assert_allclose(result, color, atol=0.02)

    def test_no_nan(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        result = easu(img, 2)
        assert not np.any(np.isnan(result))

    def test_small_image(self):
        """EASU should handle very small images."""
        img = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)  # 1x1
        result = easu(img, 2)
        assert result.shape == (2, 2, 3)

    def test_progress_callback(self):
        img = np.random.rand(16, 16, 3).astype(np.float32)
        progress_values = []

        def cb(frac):
            progress_values.append(frac)

        easu(img, 2, progress_callback=cb)
        assert len(progress_values) > 0
        assert progress_values[-1] == pytest.approx(1.0)
