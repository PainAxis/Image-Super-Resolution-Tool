"""Super-resolution pipeline: AMD EASU, AMD RCAS, then optional FXAA."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sr_tool.fsr.antialias import fxaa
from sr_tool.fsr.common import CancelCallback, check_cancelled, validate_rgb_image
from sr_tool.fsr.easu import easu
from sr_tool.fsr.rcas import DEFAULT_SHARPNESS, rcas
from sr_tool.utils.resources import ensure_pipeline_budget


def _map_progress(
    callback: Callable[[float], None] | None,
    start: float,
    weight: float,
) -> Callable[[float], None] | None:
    if callback is None:
        return None

    def mapped(fraction: float) -> None:
        callback(start + weight * min(max(float(fraction), 0.0), 1.0))

    return mapped


def process_image(
    img: np.ndarray,
    scale: float,
    rcas_sharpness: float = DEFAULT_SHARPNESS,
    antialias: bool = False,
    progress_callback: Callable[[float], None] | None = None,
    cancel_callback: CancelCallback | None = None,
    *,
    rcas_denoise: bool = False,
) -> np.ndarray:
    """Run the complete, cancellable super-resolution pipeline.

    ``rcas_sharpness`` follows AMD's stop scale: 0 is maximum sharpening and
    2 is one quarter of that effect. Progress is monotonic across all enabled
    stages, including FXAA.
    """
    source = validate_rgb_image(img)
    scale = float(scale)
    if not np.isfinite(scale) or not 1.0 <= scale <= 4.0:
        raise ValueError("scale must be finite and in [1, 4]")
    ensure_pipeline_budget(source.shape[0], source.shape[1], scale)
    check_cancelled(cancel_callback)
    if progress_callback is not None:
        progress_callback(0.0)

    if antialias:
        easu_weight, rcas_weight, fxaa_weight = 0.80, 0.10, 0.10
    else:
        easu_weight, rcas_weight, fxaa_weight = 0.88, 0.12, 0.0

    result = easu(
        source,
        scale,
        progress_callback=_map_progress(progress_callback, 0.0, easu_weight),
        cancel_callback=cancel_callback,
    )
    check_cancelled(cancel_callback)
    result = rcas(
        result,
        sharpness=rcas_sharpness,
        progress_callback=_map_progress(progress_callback, easu_weight, rcas_weight),
        cancel_callback=cancel_callback,
        denoise=rcas_denoise,
    )
    check_cancelled(cancel_callback)
    if antialias:
        result = fxaa(
            result,
            progress_callback=_map_progress(
                progress_callback, easu_weight + rcas_weight, fxaa_weight
            ),
            cancel_callback=cancel_callback,
        )
    if progress_callback is not None:
        progress_callback(1.0)
    return result
