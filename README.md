# Image Super-Resolution Tool

A desktop CPU image upscaler built from NumPy ports of AMD FidelityFX Super
Resolution 1.0 EASU/RCAS, with optional NVIDIA FXAA 3.11 post-processing.

This is a spatial upscaler, not a learned generative model: it improves edge
reconstruction and contrast but cannot recover real detail that is absent from
the source image.

## Features

- 2×, 3× and 4× EASU upscaling using the official FSR 1.0 12-tap structure.
- RCAS sharpening with AMD's stop scale: `0` is strongest and `2` is one
  quarter of that effect.
- Optional FXAA 3.11 quality preset 12.
- English, Chinese and Russian UI with system-locale detection.
- Truly synchronized comparison: pan and zoom are stored in original-image
  coordinates, so both panes show the same content even at different pixel
  resolutions.
- Cooperative cancellation, stale-result protection, safe window shutdown and
  progress reporting for every processing stage.
- Block-bounded algorithm temporaries plus pre-allocation pixel/memory checks.

The CPU ports retain the upstream kernels and add conservative handling for
mathematically ambiguous directions. This prevents single-direction impulse
halos and removes reflection seams caused by exact integer sample positions.

## Requirements and installation

- Python 3.10 or newer on Windows, macOS or Linux
- Tk (normally bundled on Windows/macOS; on Debian/Ubuntu install `python3-tk`)

For a reproducible runtime:

```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

For an editable install and the `image-super-resolution` command:

```bash
python -m pip install -e .
image-super-resolution
```

## Usage

1. Open a PNG, JPEG, BMP or TIFF image.
2. Select 2×, 3× or 4×.
3. Select the RCAS reduction in stops (`0` = maximum sharpening) and whether
   to apply FXAA.
4. Start processing. Press **Cancel** or Escape to stop at the next tile.
5. Drag or use the mouse wheel in either pane; the other pane follows the same
   logical source region.
6. Save the result. The write is atomic, so an encoder failure does not replace
   an existing destination.

## Image fidelity

The loader applies EXIF orientation, converts valid embedded ICC profiles to
sRGB, and carries safe EXIF/DPI/text metadata to the result. Straight alpha is
resized separately and preserved in PNG/TIFF output.

| Data | Supported behavior |
|---|---|
| 8-bit RGB | PNG, JPEG, BMP and TIFF input/output |
| 8-bit alpha | Preserved with PNG or TIFF output |
| 16-bit grayscale | Loaded without 8-bit truncation; PNG/TIFF output remains 16-bit |
| 16-bit color | Rejected rather than silently reduced to 8-bit |

JPEG and BMP cannot preserve transparency. Choosing either for an image with
alpha produces an explicit error instead of silently discarding it.

## Resource limits

The pipeline must still hold the input and full output arrays; “block
processing” bounds temporary working tiles, not the returned image itself.
Before allocation it estimates the worst stage at roughly 32 bytes per output
pixel plus input and tile headroom. Defaults are:

- 100,000,000 input pixels
- 150,000,000 output pixels
- 6 GiB configured memory, further limited to 75% of currently available RAM

A 3840×2160 image at 4× has 132.7 million output pixels and an estimated peak
of about 4.30 GiB. An 8K image at 4× is rejected by the default output limit.
Intentional overrides are available through `SR_TOOL_MAX_INPUT_PIXELS`,
`SR_TOOL_MAX_OUTPUT_PIXELS`, and `SR_TOOL_MAX_MEMORY_MIB`.

## Development and verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m coverage run -m pytest && python -m coverage report
python -m ruff check .
python -m mypy sr_tool run.py
python -m bandit -q -r sr_tool
python -m pip_audit -r requirements.txt
python -m build
```

CI exercises the Python 3.10 and current 3.14 boundaries on Ubuntu, Windows
and macOS. Tk lifecycle tests run under Xvfb on Ubuntu and natively on the
Windows/macOS hosted runners. Core coverage has an enforced 80% floor; GUI
event glue is exercised separately by the platform smoke tests.

Reproduce a performance measurement instead of relying on a machine-independent
speed claim:

```bash
python -m benchmarks.benchmark_pipeline --width 512 --height 512 --scale 2
python -m benchmarks.benchmark_pipeline --width 512 --height 512 --scale 2 --fxaa
```

The benchmark reports median wall time, output megapixels per second, and the
same conservative memory estimate used by the preflight check.
See [PERFORMANCE.md](PERFORMANCE.md) for the audited-commit comparison and
regression policy.

## Algorithm provenance

- EASU/RCAS: AMD FidelityFX Super Resolution 1.0, 32-bit reference paths.
- FXAA: NVIDIA FXAA 3.11 by Timothy Lottes, quality preset 12.

See [NOTICE](NOTICE) for upstream copyright and license terms. The project
itself is available under the [MIT License](LICENSE).
