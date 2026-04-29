"""Image loading and saving via Pillow, with NumPy conversion."""

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image file and return as float32 (H, W, 3) in [0, 1]."""
    with Image.open(path) as img:
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def save_image(arr: np.ndarray, path: str) -> None:
    """Save a float32 (H, W, 3) array in [0, 1] to an image file."""
    clipped = np.clip(arr, 0.0, 1.0)
    # rint → nearest integer (avoids truncation bias, e.g. 254.9 → 254)
    img = Image.fromarray(np.rint(clipped * 255.0).astype(np.uint8))
    img.save(path)


def array_to_photoimage(arr: np.ndarray) -> object:
    """Convert float32 (H, W, 3) in [0, 1] to a PIL ImageTk.PhotoImage for tkinter display."""
    clipped = np.clip(arr, 0.0, 1.0)
    uint8_arr = np.rint(clipped * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(uint8_arr, mode="RGB")
    # Import here to avoid requiring tkinter at module level
    from PIL import ImageTk
    return ImageTk.PhotoImage(pil_img)
