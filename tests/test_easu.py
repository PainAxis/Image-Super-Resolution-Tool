"""Conformance and invariant tests for the EASU port."""

import numpy as np
import pytest

from sr_tool.fsr.common import ProcessingCancelled
from sr_tool.fsr.easu import _lanczos2, easu


@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_output_contract(scale: int) -> None:
    image = np.random.default_rng(7).random((9, 13, 3), dtype=np.float32)
    result = easu(image, scale)
    assert result.shape == (9 * scale, 13 * scale, 3)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


@pytest.mark.parametrize("color", [0.0, 0.25, 0.5, 1.0])
def test_flat_image_is_preserved(color: float) -> None:
    image = np.full((8, 11, 3), color, dtype=np.float32)
    np.testing.assert_allclose(easu(image, 4), color, atol=2e-6)


def test_binary_fuzz_cannot_overshoot() -> None:
    image = np.random.default_rng(11).integers(0, 2, (17, 19, 3)).astype(np.float32)
    result = easu(image, 4)
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_rotations_and_reflections_are_equivariant() -> None:
    image = np.random.default_rng(23).random((17, 21, 3), dtype=np.float32)
    expected = easu(image, 3)
    for turns in (1, 2, 3):
        transformed = np.rot90(easu(np.rot90(image, turns), 3), -turns)
        np.testing.assert_allclose(transformed, expected, atol=3e-6, rtol=0.0)
    for axis in (0, 1):
        transformed = np.flip(easu(np.flip(image, axis), 3), axis)
        np.testing.assert_allclose(transformed, expected, atol=3e-6, rtol=0.0)


def test_block_size_does_not_change_pixels() -> None:
    image = np.random.default_rng(31).random((19, 23, 3), dtype=np.float32)
    whole = easu(image, 3, block_size=4096)
    tiled = easu(image, 3, block_size=11)
    np.testing.assert_allclose(tiled, whole, atol=1e-6, rtol=0.0)


def test_tiny_image_and_scale_one() -> None:
    image = np.array([[[0.3, 0.5, 0.7]]], dtype=np.float32)
    assert easu(image, 2).shape == (2, 2, 3)
    result = easu(image, 1)
    np.testing.assert_array_equal(result, image)
    assert result is not image


def test_progress_is_monotonic_and_finishes() -> None:
    values: list[float] = []
    image = np.zeros((12, 12, 3), dtype=np.float32)
    easu(image, 3, progress_callback=values.append, block_size=7)
    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)


def test_cancellation_is_cooperative() -> None:
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(ProcessingCancelled):
        easu(
            np.zeros((16, 16, 3), dtype=np.float32),
            4,
            cancel_callback=cancelled,
            block_size=8,
        )


@pytest.mark.parametrize("scale", [0, 4.01, float("nan"), float("inf")])
def test_invalid_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError):
        easu(np.zeros((2, 2, 3), dtype=np.float32), scale)


def test_legacy_lanczos_helper_contract() -> None:
    values = _lanczos2(np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(values, [1.0, 0.0, 0.0], atol=1e-6)
