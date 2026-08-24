"""Behavioral and partition-invariance tests for FXAA 3.11."""

import numpy as np
import pytest

from sr_tool.fsr.antialias import fxaa
from sr_tool.fsr.common import ProcessingCancelled


def test_flat_and_isolated_pixels_do_not_gain_directional_halos() -> None:
    flat = np.full((16, 16, 3), 0.3, dtype=np.float32)
    np.testing.assert_array_equal(fxaa(flat), flat)
    impulse = np.zeros((9, 9, 3), dtype=np.float32)
    impulse[4, 4] = 1.0
    np.testing.assert_allclose(fxaa(impulse), impulse, atol=1e-6)


def test_diagonal_staircase_is_antialiased() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    for row in range(32):
        image[row, row:] = 1.0
    result = fxaa(image)
    assert np.count_nonzero(np.abs(result - image) > 1e-5) > 0
    assert np.any((result > 0.0) & (result < 1.0))


def test_output_contract() -> None:
    image = np.random.default_rng(29).random((21, 25, 3), dtype=np.float32)
    result = fxaa(image)
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_block_size_is_exactly_invariant() -> None:
    image = np.random.default_rng(37).random((43, 47, 3), dtype=np.float32)
    expected = fxaa(image, block_size=4096)
    actual = fxaa(image, block_size=9)
    np.testing.assert_array_equal(actual, expected)


def test_mirrors_are_equivariant() -> None:
    image = np.random.default_rng(41).random((38, 45, 3), dtype=np.float32)
    expected = fxaa(image)
    for axis in (0, 1):
        transformed = np.flip(fxaa(np.flip(image, axis)), axis)
        np.testing.assert_allclose(transformed, expected, atol=4e-6, rtol=0.0)


def test_progress_is_reported_for_every_block() -> None:
    values: list[float] = []
    image = np.zeros((12, 13, 3), dtype=np.float32)
    fxaa(image, progress_callback=values.append, block_size=5)
    assert len(values) == 9
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_immediate_cancellation_precedes_output_allocation() -> None:
    with pytest.raises(ProcessingCancelled):
        fxaa(
            np.zeros((4, 4, 3), dtype=np.float32),
            cancel_callback=lambda: True,
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("edge_threshold", -0.1), ("edge_threshold_min", 1.1), ("subpix_quality", 2.0)],
)
def test_invalid_quality_knobs_are_rejected(keyword: str, value: float) -> None:
    with pytest.raises(ValueError):
        fxaa(np.zeros((2, 2, 3), dtype=np.float32), **{keyword: value})
