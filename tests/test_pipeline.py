"""End-to-end super-resolution pipeline tests."""

import numpy as np
import pytest

from sr_tool.fsr.common import ProcessingCancelled
from sr_tool.fsr.pipeline import process_image


@pytest.mark.parametrize("scale", [2, 3, 4])
def test_pipeline_output_contract(scale: int) -> None:
    image = np.random.default_rng(scale).random((8, 11, 3), dtype=np.float32)
    result = process_image(image, scale, antialias=True)
    assert result.shape == (8 * scale, 11 * scale, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_pipeline_preserves_flat_color() -> None:
    image = np.full((7, 9, 3), 0.5, dtype=np.float32)
    np.testing.assert_allclose(process_image(image, 3, antialias=True), 0.5, atol=2e-6)


def test_progress_is_monotonic_through_fxaa() -> None:
    values: list[float] = []
    image = np.random.default_rng(43).random((12, 13, 3), dtype=np.float32)
    process_image(image, 2, antialias=True, progress_callback=values.append)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert values == sorted(values)
    assert 0.9 in values
    # FXAA reports its own completion, followed by the pipeline's final fence.
    assert values.count(1.0) == 2


def test_immediate_cancellation() -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ProcessingCancelled):
        process_image(image, 2, cancel_callback=lambda: True)


@pytest.mark.parametrize(
    "bad_image",
    [
        np.zeros((2, 2), dtype=np.float32),
        np.zeros((2, 2, 4), dtype=np.float32),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), np.nan, dtype=np.float32),
        np.full((2, 2, 3), 1.1, dtype=np.float32),
    ],
)
def test_invalid_image_contract_is_rejected(bad_image: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        process_image(bad_image, 2)
