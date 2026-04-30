# Image Super-Resolution Tool

A desktop tool for image super-resolution. Currently powered by AMD FidelityFX Super Resolution 1.0.

## Features

- **Super-resolution upscaling**: EASU (Edge-Adaptive Spatial Upsampling) + RCAS
  (Robust Contrast-Adaptive Sharpening)
- **FXAA anti-aliasing**: NVIDIA FXAA 3.11 post-processing
- **Multi-language**: English, Chinese, Russian — auto-detects system locale
- **Side-by-side comparison**: original and result, synchronized zoom and pan
- **Adjustable parameters**: scale factor (2x/3x/4x), sharpness (0–1), AA toggle

## Quick Start

### Requirements
- Python >= 3.10

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

### Usage

1. Click **Open Image** or press Ctrl+O to open an image
2. Choose the upscale factor (2x / 3x / 4x)
3. Adjust sharpness and anti-aliasing as needed
4. Click **Process** to start
5. Compare original and result — drag and scroll to inspect details
6. Click **Save Result** or press Ctrl+S to save

## Project Structure

```
FSR/
├── run.py                     # entry point
├── requirements.txt           # dependencies
├── README.md
├── LICENSE                    # MIT License
├── fsr_tool/
│   ├── fsr/
│   │   ├── easu.py            # EASU upsampling
│   │   ├── rcas.py            # RCAS sharpening
│   │   ├── antialias.py       # FXAA anti-aliasing
│   │   └── pipeline.py        # processing pipeline
│   ├── gui/
│   │   └── app.py             # tkinter GUI
│   ├── locale/
│   │   ├── en.json            # English
│   │   ├── zh.json            # Chinese
│   │   ├── ru.json            # Russian
│   │   └── i18n.py            # i18n module
│   └── utils/
│       └── image_io.py        # image I/O
└── tests/
    ├── test_easu.py
    ├── test_rcas.py
    ├── test_antialias.py
    └── test_pipeline.py
```

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Technical Notes

- **Pure Python + NumPy**: no GPU, no CUDA, no C++ compilation required
- **Block processing**: all algorithms use chunked processing to bound peak memory
- **Thread-safe**: background worker threads + main-thread UI updates

## Developer

[PainAxis](https://github.com/PainAxis)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Based on AMD FidelityFX Super Resolution 1.0.
