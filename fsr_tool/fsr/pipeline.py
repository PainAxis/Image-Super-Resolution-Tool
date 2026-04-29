"""FSR 1.0 processing pipeline: EASU upscale -> RCAS sharpen -> FXAA."""

from typing import Callable, Optional

import numpy as np
from fsr_tool.fsr.easu import easu
from fsr_tool.fsr.rcas import rcas, DEFAULT_SHARPNESS
from fsr_tool.fsr.antialias import fxaa

# Progress weight allocation (must sum to 1.0)
_WEIGHT_EASU = 0.80
_WEIGHT_RCAS = 0.10
_WEIGHT_FXAA = 0.10


def process_image(img: np.ndarray,
                  scale: float,
                  rcas_sharpness: float = DEFAULT_SHARPNESS,
                  antialias: bool = False,
                  progress_callback: Optional[Callable[[float], None]] = None,
                  ) -> np.ndarray:
    """Run the full FSR 1.0 pipeline.

    Args:
        img: float32 (H, W, 3) array in [0, 1].
        scale: upscale factor (e.g. 2, 3, 4).
        rcas_sharpness: RCAS sharpening strength [0, 1].
        antialias: apply FXAA anti-aliasing as post-processing.
        progress_callback: optional callable(fraction: float), 0→1.

    Returns:
        float32 (H*scale, W*scale, 3) array in [0, 1].
    """
    _cb = progress_callback

    # --- EASU (weighted 0 → _WEIGHT_EASU) ---
    easu_cb = None
    if _cb is not None:
        def easu_cb(frac: float) -> None:
            _cb(frac * _WEIGHT_EASU)

    # Reuse single variable so each stage's output replaces the previous:
    # GC can collect the old array as soon as the new one is allocated,
    # avoiding peak memory stacking of all intermediate results.
    result = easu(img, scale, progress_callback=easu_cb)

    # --- RCAS (weighted _WEIGHT_EASU → _WEIGHT_EASU + _WEIGHT_RCAS) ---
    rcas_cb = None
    if _cb is not None:
        def rcas_cb(frac: float) -> None:
            _cb(_WEIGHT_EASU + frac * _WEIGHT_RCAS)

    result = rcas(result, sharpness=rcas_sharpness,
                  progress_callback=rcas_cb)

    # --- FXAA (weighted … → 1.0) ---
    if antialias:
        result = fxaa(result)

    if _cb is not None:
        _cb(1.0)

    return result
