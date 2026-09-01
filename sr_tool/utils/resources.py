"""Deterministic resource estimation and pre-allocation safety checks."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_INPUT_PIXELS = 100_000_000
DEFAULT_MAX_OUTPUT_PIXELS = 150_000_000
DEFAULT_MAX_MEMORY_BYTES = 6 * 1024**3
_BLOCK_WORKING_BYTES = 256 * 1024**2
_DECODE_WORKING_BYTES = 64 * 1024**2
_OPAQUE_DECODE_BYTES_PER_PIXEL = 28
_ALPHA_DECODE_BYTES_PER_PIXEL = 40
_RGB_INPUT_BYTES_PER_PIXEL = 3 * 4
_RGB_OUTPUT_BYTES_PER_PIXEL = 32
# Transparent GUI jobs retain the source document, a premultiplied working
# copy, alpha planes, and both sides of the final unpremultiply/clip operation.
_ALPHA_INPUT_BYTES_PER_PIXEL = 28
_ALPHA_OUTPUT_BYTES_PER_PIXEL = 48


class ResourceLimitError(ValueError):
    """Raised before decoding/allocation would exceed a configured limit."""


@dataclass(frozen=True)
class ResourceEstimate:
    input_pixels: int
    output_pixels: int
    peak_bytes: int

    @property
    def peak_mib(self) -> float:
        return self.peak_bytes / 1024**2


def _environment_limit(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ResourceLimitError(f"{name} must be an integer") from exc
    if value < 1:
        raise ResourceLimitError(f"{name} must be positive")
    return value


def max_input_pixels() -> int:
    return _environment_limit("SR_TOOL_MAX_INPUT_PIXELS", DEFAULT_MAX_INPUT_PIXELS)


def max_output_pixels() -> int:
    return _environment_limit("SR_TOOL_MAX_OUTPUT_PIXELS", DEFAULT_MAX_OUTPUT_PIXELS)


def configured_memory_limit() -> int:
    mebibytes = _environment_limit(
        "SR_TOOL_MAX_MEMORY_MIB", DEFAULT_MAX_MEMORY_BYTES // 1024**2
    )
    return mebibytes * 1024**2


def _read_memory_counter(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="ascii").strip()
        if raw == "max":
            return None
        value = int(raw)
    except (OSError, ValueError):
        return None
    return value if 0 <= value < 1 << 60 else None


def _cgroup_available_bytes(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Return remaining Linux cgroup memory for v2 or legacy v1 layouts."""
    layouts = (
        (root / "memory.max", root / "memory.current"),
        (
            root / "memory" / "memory.limit_in_bytes",
            root / "memory" / "memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in layouts:
        limit = _read_memory_counter(limit_path)
        usage = _read_memory_counter(usage_path)
        if limit is not None and usage is not None:
            return max(0, limit - usage)
    return None


def _sysconf_available_bytes() -> int | None:
    """Return available physical memory on POSIX systems that expose sysconf."""
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        pages = int(sysconf("SC_AVPHYS_PAGES"))
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        return None
    if pages < 0 or page_size < 1:
        return None
    return pages * page_size


def available_memory_bytes() -> int | None:
    """Best-effort available-memory query without an optional dependency."""
    candidates: list[int] = []
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii") as meminfo:
                for line in meminfo:
                    if line.startswith("MemAvailable:"):
                        candidates.append(int(line.split()[1]) * 1024)
                        break
        except (OSError, ValueError, IndexError):
            pass

        cgroup_available = _cgroup_available_bytes()
        if cgroup_available is not None:
            candidates.append(cgroup_available)
    if candidates:
        return min(candidates)

    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            windll = getattr(ctypes, "windll", None)
            if windll is not None and windll.kernel32.GlobalMemoryStatusEx(
                ctypes.byref(status)
            ):
                return int(status.available_physical)
        except (AttributeError, OSError, ValueError):
            pass
    if os.name == "posix":
        return _sysconf_available_bytes()
    return None


def _effective_memory_limit() -> int:
    memory_limit = configured_memory_limit()
    available = available_memory_bytes()
    if available is not None:
        memory_limit = min(memory_limit, int(available * 0.75))
    return memory_limit


def estimate_decode_bytes(height: int, width: int, *, has_alpha: bool) -> int:
    """Estimate peak bytes while decoding and normalizing an input image."""
    pixels = int(height) * int(width)
    bytes_per_pixel = (
        _ALPHA_DECODE_BYTES_PER_PIXEL
        if has_alpha
        else _OPAQUE_DECODE_BYTES_PER_PIXEL
    )
    return pixels * bytes_per_pixel + _DECODE_WORKING_BYTES


def ensure_input_budget(height: int, width: int, *, has_alpha: bool) -> int:
    """Reject an input whose decode/normalization peak exceeds safe memory."""
    ensure_input_size(height, width)
    estimate = estimate_decode_bytes(height, width, has_alpha=has_alpha)
    memory_limit = _effective_memory_limit()
    if estimate > memory_limit:
        required = estimate / 1024**3
        allowed = memory_limit / 1024**3
        raise ResourceLimitError(
            f"Estimated image decode memory is {required:.2f} GiB, above the safe "
            f"budget of {allowed:.2f} GiB. Choose a smaller image or set "
            "SR_TOOL_MAX_MEMORY_MIB intentionally."
        )
    return estimate


def estimate_peak_resources(
    height: int,
    width: int,
    scale: float,
    *,
    has_alpha: bool = False,
) -> ResourceEstimate:
    """Estimate the pipeline peak, including simultaneous stage buffers."""
    input_pixels = int(height) * int(width)
    output_height = round(height * float(scale))
    output_width = round(width * float(scale))
    output_pixels = output_height * output_width
    if has_alpha:
        input_bytes = input_pixels * _ALPHA_INPUT_BYTES_PER_PIXEL
        output_bytes = output_pixels * _ALPHA_OUTPUT_BYTES_PER_PIXEL
    else:
        input_bytes = input_pixels * _RGB_INPUT_BYTES_PER_PIXEL
        output_bytes = output_pixels * _RGB_OUTPUT_BYTES_PER_PIXEL
    # Include vectorized tile temporaries and allocator/headroom. The alpha
    # factors also cover the final resize/unpremultiply stage in the GUI.
    peak_bytes = input_bytes + output_bytes + _BLOCK_WORKING_BYTES
    return ResourceEstimate(input_pixels, output_pixels, peak_bytes)


def ensure_input_size(height: int, width: int) -> None:
    pixels = int(height) * int(width)
    limit = max_input_pixels()
    if height < 1 or width < 1:
        raise ResourceLimitError("Image dimensions must be positive")
    if pixels > limit:
        raise ResourceLimitError(
            f"Input has {pixels:,} pixels; configured limit is {limit:,}. "
            "Set SR_TOOL_MAX_INPUT_PIXELS to an intentional higher value."
        )


def ensure_pipeline_budget(
    height: int,
    width: int,
    scale: float,
    *,
    has_alpha: bool = False,
) -> ResourceEstimate:
    estimate = estimate_peak_resources(height, width, scale, has_alpha=has_alpha)
    output_limit = max_output_pixels()
    if estimate.output_pixels > output_limit:
        raise ResourceLimitError(
            f"Output would have {estimate.output_pixels:,} pixels; configured "
            f"limit is {output_limit:,}. Choose a smaller scale or set "
            "SR_TOOL_MAX_OUTPUT_PIXELS intentionally."
        )

    memory_limit = _effective_memory_limit()
    if estimate.peak_bytes > memory_limit:
        required = estimate.peak_bytes / 1024**3
        allowed = memory_limit / 1024**3
        raise ResourceLimitError(
            f"Estimated peak memory is {required:.2f} GiB, above the safe "
            f"budget of {allowed:.2f} GiB. Choose a smaller image/scale or "
            "set SR_TOOL_MAX_MEMORY_MIB intentionally."
        )
    return estimate
