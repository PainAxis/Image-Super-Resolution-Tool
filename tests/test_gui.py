"""Tk lifecycle smoke tests (run under Xvfb in CI)."""

import time
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import sr_tool.gui.app as app_module
from sr_tool.gui.app import Application


@pytest.fixture
def application() -> tuple[Application, tk.Tk]:
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk display is unavailable; CI runs this test under Xvfb")
    root.withdraw()
    app = Application(root)
    yield app, root
    if not app._closing:
        app._on_close()


def test_open_process_and_save_lifecycle(
    application: tuple[Application, tk.Tk],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, root = application
    source_path = tmp_path / "source.png"
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 192
    Image.fromarray(rgba).save(source_path)
    monkeypatch.setattr(
        app_module.filedialog, "askopenfilename", lambda **_kwargs: str(source_path)
    )
    app.open_image()
    assert app.source_document is not None
    assert app.source_document.alpha is not None
    assert app.process_btn["state"] == tk.NORMAL

    def fake_process(
        image: np.ndarray,
        scale: int,
        **kwargs: Any,
    ) -> np.ndarray:
        kwargs["progress_callback"](0.5)
        assert not kwargs["cancel_callback"]()
        return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

    monkeypatch.setattr(app_module, "sr_process", fake_process)
    app.process_image()
    deadline = time.monotonic() + 3.0

    def poll() -> None:
        if not app.processing or time.monotonic() >= deadline:
            root.quit()
        else:
            root.after(10, poll)

    root.after(10, poll)
    root.mainloop()
    assert not app.processing
    assert app.result_document is not None
    assert app.result_document.rgb.shape == (6, 8, 3)
    assert app.result_document.alpha is not None
    assert app.result_document.alpha.shape == (6, 8)

    output_path = tmp_path / "result.png"
    monkeypatch.setattr(
        app_module.filedialog,
        "asksaveasfilename",
        lambda **_kwargs: str(output_path),
    )
    app.save_result()
    assert output_path.exists()
    assert np.asarray(Image.open(output_path)).shape == (6, 8, 4)


def test_cancel_and_stale_completion_are_safe(
    application: tuple[Application, tk.Tk],
) -> None:
    app, _root = application
    app.processing = True
    app._job_id = 4
    app._cancel_event = app_module.threading.Event()
    app.cancel_processing()
    assert app._cancel_event.is_set()
    stale = np.zeros((2, 2, 3), dtype=np.float32)
    document = app_module.image_io.ImageDocument(
        stale, None, app_module.image_io.ImageMetadata()
    )
    app._on_complete(3, document, 2, 0.1)
    assert app.result_document is None
