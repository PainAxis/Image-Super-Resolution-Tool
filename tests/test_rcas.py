"""Reference semantics and regression tests for RCAS."""

import numpy as np
import pytest

from sr_tool.fsr.common import ProcessingCancelled
from sr_tool.fsr.rcas import rcas


def test_flat_and_tiny_images_are_preserved() -> None:
    for image in (
        np.full((32, 32, 3), 0.5, dtype=np.float32),
        np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32),
    ):
        np.testing.assert_allclose(rcas(image), image, atol=1e-6)


def test_isolated_highlight_is_not_erased() -> None:
    image = np.zeros((9, 9, 3), dtype=np.float32)
    image[4, 4] = 1.0
    result = rcas(image, sharpness=0.0)
    np.testing.assert_array_equal(result, image)


def test_stop_scale_zero_means_maximum_sharpening() -> None:
    image = np.random.default_rng(5).random((32, 35, 3), dtype=np.float32)
    changes = [
        float(np.mean(np.abs(rcas(image, sharpness=stops) - image)))
        for stops in (0.0, 1.0, 2.0)
    ]
    assert changes[0] > changes[1] > changes[2] > 0.0


def test_output_contract() -> None:
    image = np.random.default_rng(9).random((31, 29, 3), dtype=np.float32)
    result = rcas(image, sharpness=0.5)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_block_size_is_pixel_invariant() -> None:
    image = np.random.default_rng(13).random((35, 41, 3), dtype=np.float32)
    expected = rcas(image, 0.2, block_size=4096)
    actual = rcas(image, 0.2, block_size=7)
    np.testing.assert_array_equal(actual, expected)


def test_optional_denoise_reduces_sharpening_of_noise() -> None:
    image = np.random.default_rng(17).random((40, 40, 3), dtype=np.float32)
    regular = rcas(image, 0.0)
    denoised = rcas(image, 0.0, denoise=True)
    assert np.mean(np.abs(denoised - image)) < np.mean(np.abs(regular - image))


def test_immediate_cancellation_precedes_output_allocation() -> None:
    with pytest.raises(ProcessingCancelled):
        rcas(
            np.zeros((4, 4, 3), dtype=np.float32),
            cancel_callback=lambda: True,
        )


@pytest.mark.parametrize("sharpness", [-0.01, 2.01, float("nan")])
def test_invalid_sharpness_is_rejected(sharpness: float) -> None:
    with pytest.raises(ValueError):
        rcas(np.zeros((2, 2, 3), dtype=np.float32), sharpness)
