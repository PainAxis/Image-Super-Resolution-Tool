"""Shared validation and cancellation primitives for image filters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ProgressCallback = Callable[[float], None]
CancelCallback = Callable[[], bool]


class ProcessingCancelled(RuntimeError):
    """Raised cooperatively when a caller cancels a processing job."""


def check_cancelled(cancel_callback: CancelCallback | None) -> None:
    """Raise :class:`ProcessingCancelled` when cancellation was requested."""
    if cancel_callback is not None and cancel_callback():
        raise ProcessingCancelled("Image processing was cancelled")


def validate_rgb_image(img: np.ndarray) -> np.ndarray:
    """Validate the public filter contract and return float32 RGB data."""
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a NumPy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must have shape (height, width, 3)")
    if img.shape[0] < 1 or img.shape[1] < 1:
        raise ValueError("img dimensions must be positive")
    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError("img must use a floating-point dtype")
    if not np.isfinite(img).all():
        raise ValueError("img contains NaN or infinity")
    if float(np.min(img)) < 0.0 or float(np.max(img)) > 1.0:
        raise ValueError("img values must be in the closed interval [0, 1]")
    return np.asarray(img, dtype=np.float32)


def validate_unit_interval(name: str, value: float) -> float:
    """Return *value* as float after checking that it lies in [0, 1]."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value
