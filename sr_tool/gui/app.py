"""Tk application with cancellable processing and synchronized comparison."""

from __future__ import annotations

import math
import threading
import time
import tkinter as tk
import traceback
from functools import partial
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

from sr_tool.fsr.common import ProcessingCancelled
from sr_tool.fsr.pipeline import process_image as sr_process
from sr_tool.gui.view_state import ViewState
from sr_tool.locale.i18n import (
    available_languages,
    current_language,
    on_language_change,
    set_language,
    t,
)
from sr_tool.utils import image_io
from sr_tool.utils.resources import ensure_pipeline_budget

_PROGRESS_INTERVAL_SECONDS = 1.0 / 30.0


class Application:
    """Image Super-Resolution desktop application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(t("app_title"))
        self.root.geometry("1200x700")
        self.root.minsize(800, 500)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.source_document: image_io.ImageDocument | None = None
        self.result_document: image_io.ImageDocument | None = None
        self.source_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.scale_factor = tk.StringVar(value="2")
        self.sharpness = tk.DoubleVar(value=0.2)
        self.antialias = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value=t("status_ready"))
        self.processing = False
        self._closing = False
        self._last_path: str | None = None
        self._job_id = 0
        self._cancel_event: threading.Event | None = None
        self._worker: threading.Thread | None = None

        self._live_widgets: list[tuple[tk.Widget, str, str, dict[str, Any]]] = []
        self._processing_controls: list[tk.Widget] = []
        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()
        on_language_change(self._on_language_changed)

    def _after(self, function: Any, *args: Any) -> None:
        """Schedule a callback only while the Tk interpreter is alive."""
        if self._closing:
            return
        try:
            self.root.after(0, function, *args)
        except (tk.TclError, RuntimeError):
            pass

    def _on_language_changed(self) -> None:
        if self._closing:
            return
        self.root.title(t("app_title"))
        self._rebuild_menu()
        for widget, attribute, key, formatting in self._live_widgets:
            try:
                if widget.winfo_exists():
                    widget[attribute] = t(key, **formatting)
            except tk.TclError:
                pass
        if not self.processing:
            if self.source_image is None:
                self.status_text.set(t("status_ready"))
            else:
                self.status_text.set(
                    t(
                        "status_loaded",
                        name=Path(self._last_path or "?").name,
                        w=self.source_image.shape[1],
                        h=self.source_image.shape[0],
                    )
                )

    # ------------------------------------------------------------------
    # Menu and layout
    # ------------------------------------------------------------------
    def _rebuild_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self._populate_menu(menubar)
        self._update_action_states()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self._populate_menu(menubar)

    def _populate_menu(self, menubar: tk.Menu) -> None:
        self._file_menu = tk.Menu(menubar, tearoff=0)
        self._file_menu.add_command(
            label=t("menu_open"), command=self.open_image, accelerator="Ctrl+O"
        )
        self._file_menu.add_command(
            label=t("menu_save"), command=self.save_result, accelerator="Ctrl+S"
        )
        self._file_menu.add_separator()
        self._file_menu.add_command(label=t("menu_exit"), command=self._on_close)
        menubar.add_cascade(label=t("menu_file"), menu=self._file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label=t("menu_fit_window"), command=self._fit_to_window)
        view_menu.add_command(label=t("menu_zoom_100"), command=self._zoom_100)
        menubar.add_cascade(label=t("menu_view"), menu=view_menu)

        language_menu = tk.Menu(menubar, tearoff=0)
        self.lang_var = tk.StringVar(value=current_language())
        for code, name in available_languages().items():
            language_menu.add_radiobutton(
                label=name,
                value=code,
                variable=self.lang_var,
                command=partial(set_language, code),
            )
        menubar.add_cascade(label=t("menu_language"), menu=language_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label=t("menu_about"), command=self._show_about)
        menubar.add_cascade(label=t("menu_help"), menu=help_menu)

    def _build_layout(self) -> None:
        control_frame = tk.Frame(self.root, padx=8, pady=6)
        control_frame.pack(fill=tk.X)
        first_row = tk.Frame(control_frame)
        first_row.pack(fill=tk.X)

        self.open_btn = tk.Button(
            first_row, text=t("btn_open"), command=self.open_image
        )
        self.open_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.open_btn, "text", "btn_open", {}))
        ttk.Separator(first_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6
        )

        scale_label = tk.Label(first_row, text=t("lbl_scale"))
        scale_label.pack(side=tk.LEFT)
        self._live_widgets.append((scale_label, "text", "lbl_scale", {}))
        self.scale_menu = tk.OptionMenu(first_row, self.scale_factor, "2", "3", "4")
        self.scale_menu.config(width=4)
        self.scale_menu.pack(side=tk.LEFT, padx=2)
        ttk.Separator(first_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6
        )

        self.process_btn = tk.Button(
            first_row,
            text=t("btn_process"),
            command=self.process_image,
            state=tk.DISABLED,
        )
        self.process_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.process_btn, "text", "btn_process", {}))
        self.cancel_btn = tk.Button(
            first_row,
            text=t("btn_cancel"),
            command=self.cancel_processing,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.cancel_btn, "text", "btn_cancel", {}))
        self.save_btn = tk.Button(
            first_row,
            text=t("btn_save"),
            command=self.save_result,
            state=tk.DISABLED,
        )
        self.save_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.save_btn, "text", "btn_save", {}))

        second_row = tk.Frame(control_frame)
        second_row.pack(fill=tk.X, pady=(4, 0))
        sharpness_label = tk.Label(second_row, text=t("lbl_sharpness"))
        sharpness_label.pack(side=tk.LEFT, padx=(0, 2))
        self._live_widgets.append((sharpness_label, "text", "lbl_sharpness", {}))
        self.sharpness_scale = tk.Scale(
            second_row,
            from_=0.0,
            to=2.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.sharpness,
            length=180,
            showvalue=True,
        )
        self.sharpness_scale.pack(side=tk.LEFT, padx=2)
        ttk.Separator(second_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8
        )
        self.antialias_check = tk.Checkbutton(
            second_row, text=t("chk_antialias"), variable=self.antialias
        )
        self.antialias_check.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.antialias_check, "text", "chk_antialias", {}))
        self._processing_controls.extend(
            [self.scale_menu, self.sharpness_scale, self.antialias_check]
        )

        pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=4,
        )
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        left_frame = tk.LabelFrame(pane, text=t("frame_original"), padx=2, pady=2)
        self._live_widgets.append((left_frame, "text", "frame_original", {}))
        self.left_canvas = tk.Canvas(left_frame, bg="#1e1e1e", highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        pane.add(left_frame, minsize=300)
        right_frame = tk.LabelFrame(pane, text=t("frame_result"), padx=2, pady=2)
        self._live_widgets.append((right_frame, "text", "frame_result", {}))
        self.right_canvas = tk.Canvas(right_frame, bg="#1e1e1e", highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        pane.add(right_frame, minsize=300)

        self._canvas_images: list[ImageTk.PhotoImage | None] = [None, None]
        self._pan_start: tuple[int, int] | None = None
        self._interacting = False
        self._redraw_after: str | None = None
        self._view = ViewState()
        self._bind_canvas_events(self.left_canvas, 0)
        self._bind_canvas_events(self.right_canvas, 1)

        status_frame = tk.Frame(self.root, padx=8, pady=4)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_frame, textvariable=self.status_text, anchor=tk.W).pack(
            fill=tk.X, side=tk.LEFT
        )
        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT, padx=4)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _event: self.open_image())
        self.root.bind("<Control-s>", lambda _event: self.save_result())
        self.root.bind("<Control-q>", lambda _event: self._on_close())
        self.root.bind("<Escape>", lambda _event: self.cancel_processing())

    # ------------------------------------------------------------------
    # Synchronized canvas interaction
    # ------------------------------------------------------------------
    def _bind_canvas_events(self, canvas: tk.Canvas, index: int) -> None:
        canvas.bind("<ButtonPress-1>", lambda event: self._on_pan_start(event))
        canvas.bind("<B1-Motion>", lambda event: self._on_pan_drag(event, index))
        canvas.bind("<ButtonRelease-1>", lambda event: self._on_pan_end(event))
        canvas.bind("<MouseWheel>", lambda event: self._on_zoom(event, index))
        canvas.bind("<Button-4>", lambda event: self._on_zoom(event, index))
        canvas.bind("<Button-5>", lambda event: self._on_zoom(event, index))
        canvas.bind("<Configure>", lambda _event: self._redraw_all())

    def _canvas(self, index: int) -> tk.Canvas:
        return self.left_canvas if index == 0 else self.right_canvas

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self._interacting = True

    def _on_pan_drag(self, event: tk.Event, index: int) -> None:
        if self._pan_start is None or self.source_image is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        canvas = self._canvas(index)
        self._view.pan(dx, dy, canvas.winfo_width(), canvas.winfo_height())
        self._pan_start = (event.x, event.y)
        self._redraw_all(use_low_quality=True)

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None
        self._interacting = False
        self._redraw_all(use_low_quality=False)

    def _on_zoom(self, event: tk.Event, index: int) -> str:
        if self.source_image is None:
            return "break"
        event_number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        zoom_in = event_number == 4 or (event_number not in {4, 5} and delta > 0)
        factor = 1.1 if zoom_in else 1.0 / 1.1
        canvas = self._canvas(index)
        self._view.zoom_at(
            factor,
            event.x,
            event.y,
            canvas.winfo_width(),
            canvas.winfo_height(),
        )
        self._interacting = True
        self._redraw_all(use_low_quality=True)
        if self._redraw_after is not None:
            try:
                self.root.after_cancel(self._redraw_after)
            except tk.TclError:
                pass
        self._redraw_after = self.root.after(200, self._redraw_high_quality)
        return "break"

    def _redraw_high_quality(self) -> None:
        self._interacting = False
        self._redraw_after = None
        self._redraw_all(use_low_quality=False)

    def _redraw_all(self, use_low_quality: bool | None = None) -> None:
        self._redraw(0, use_low_quality)
        self._redraw(1, use_low_quality)

    def _redraw(self, index: int, use_low_quality: bool | None = None) -> None:
        if use_low_quality is None:
            use_low_quality = self._interacting
        canvas = self._canvas(index)
        document = self.source_document if index == 0 else self.result_document
        canvas.delete("all")
        if document is None:
            return
        image_arr = document.rgb
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        image_height, image_width = image_arr.shape[:2]
        zoom_x, zoom_y, origin_x, origin_y = self._view.image_geometry(
            canvas_width,
            canvas_height,
            image_width,
            image_height,
        )
        source_x1 = max(0.0, -origin_x / zoom_x)
        source_y1 = max(0.0, -origin_y / zoom_y)
        source_x2 = min(image_width, (canvas_width - origin_x) / zoom_x)
        source_y2 = min(image_height, (canvas_height - origin_y) / zoom_y)
        x1 = max(0, math.floor(source_x1))
        y1 = max(0, math.floor(source_y1))
        x2 = min(image_width, max(x1 + 1, math.ceil(source_x2)))
        y2 = min(image_height, max(y1 + 1, math.ceil(source_y2)))
        if x2 <= x1 or y2 <= y1:
            return

        rgb8 = np.rint(np.clip(image_arr[y1:y2, x1:x2], 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
        if document.alpha is None:
            crop_image = PILImage.fromarray(rgb8)
        else:
            alpha8 = np.rint(
                np.clip(document.alpha[y1:y2, x1:x2], 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            crop_image = PILImage.fromarray(np.dstack((rgb8, alpha8)))
        display_width = max(1, round((x2 - x1) * zoom_x))
        display_height = max(1, round((y2 - y1) * zoom_y))
        resampling = (
            PILImage.Resampling.NEAREST
            if use_low_quality
            else PILImage.Resampling.LANCZOS
        )
        crop_image = crop_image.resize((display_width, display_height), resampling)
        self._canvas_images[index] = ImageTk.PhotoImage(crop_image)
        canvas.create_image(
            origin_x + x1 * zoom_x,
            origin_y + y1 * zoom_y,
            anchor=tk.NW,
            image=self._canvas_images[index],
        )

    def _fit_to_window(self) -> None:
        if self.source_image is None:
            return
        widths = [self.left_canvas.winfo_width(), self.right_canvas.winfo_width()]
        heights = [self.left_canvas.winfo_height(), self.right_canvas.winfo_height()]
        self._view.fit(max(1, min(widths)), max(1, min(heights)))
        self._interacting = False
        self._redraw_all(use_low_quality=False)

    def _zoom_100(self) -> None:
        if self.source_image is None:
            return
        self._view.zoom = 1.0
        self._view.center_x = self._view.source_width / 2.0
        self._view.center_y = self._view.source_height / 2.0
        self._interacting = False
        self._redraw_all(use_low_quality=False)

    # ------------------------------------------------------------------
    # Jobs and actions
    # ------------------------------------------------------------------
    def _update_action_states(self) -> None:
        has_source = self.source_document is not None
        has_result = self.result_document is not None
        idle = not self.processing
        self.open_btn.config(state=tk.NORMAL if idle else tk.DISABLED)
        self.process_btn.config(state=tk.NORMAL if idle and has_source else tk.DISABLED)
        self.save_btn.config(state=tk.NORMAL if idle and has_result else tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL if self.processing else tk.DISABLED)
        for control in self._processing_controls:
            control.configure({"state": tk.NORMAL if idle else tk.DISABLED})
        try:
            self._file_menu.entryconfig(0, state=tk.NORMAL if idle else tk.DISABLED)
            self._file_menu.entryconfig(
                1, state=tk.NORMAL if idle and has_result else tk.DISABLED
            )
        except tk.TclError:
            pass

    def open_image(self) -> None:
        if self.processing:
            return
        path = filedialog.askopenfilename(
            title=t("dialog_open_title"),
            filetypes=[
                (
                    t("filetype_images"),
                    ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"),
                ),
                (t("filetype_all"), "*.*"),
            ],
        )
        if not path:
            return
        try:
            document = image_io.load_image_document(path)
        except (image_io.ImageIOError, ValueError) as exc:
            messagebox.showerror(
                t("dialog_error_title"), t("dialog_load_failed", error=str(exc))
            )
            return

        self.source_document = document
        self.result_document = None
        self.source_image = document.rgb
        self.result_image = None
        self._last_path = path
        self._view.reset(document.rgb.shape[1], document.rgb.shape[0])
        self.status_text.set(
            t(
                "status_loaded",
                name=Path(path).name,
                w=document.rgb.shape[1],
                h=document.rgb.shape[0],
            )
        )
        self._update_action_states()
        self.root.update_idletasks()
        self._fit_to_window()

    def process_image(self) -> None:
        if self.source_document is None or self.processing:
            return
        source_document = self.source_document
        scale = int(self.scale_factor.get())
        sharpness = self.sharpness.get()
        use_antialias = self.antialias.get()
        try:
            ensure_pipeline_budget(
                source_document.rgb.shape[0], source_document.rgb.shape[1], scale
            )
        except ValueError as exc:
            messagebox.showerror(
                t("dialog_error_title"), t("dialog_process_failed", error=str(exc))
            )
            return

        self.processing = True
        self._job_id += 1
        job_id = self._job_id
        cancel_event = threading.Event()
        self._cancel_event = cancel_event
        self.result_document = None
        self.result_image = None
        self.progress.config(mode="determinate", value=0)
        self.status_text.set(t("status_processing", scale=scale))
        self._update_action_states()
        self._redraw(1)
        last_update = [0.0]

        def progress_callback(fraction: float) -> None:
            now = time.monotonic()
            if now - last_update[0] < _PROGRESS_INTERVAL_SECONDS and fraction < 1.0:
                return
            last_update[0] = now
            self._after(self._on_progress, job_id, fraction)

        def run() -> None:
            started = time.monotonic()
            try:
                working_rgb = source_document.rgb
                if source_document.alpha is not None:
                    working_rgb = image_io.premultiply_rgb(
                        working_rgb, source_document.alpha
                    )
                result = sr_process(
                    working_rgb,
                    scale,
                    rcas_sharpness=sharpness,
                    antialias=use_antialias,
                    progress_callback=progress_callback,
                    cancel_callback=cancel_event.is_set,
                )
                if cancel_event.is_set():
                    raise ProcessingCancelled("Image processing was cancelled")
                result_alpha = None
                if source_document.alpha is not None:
                    result_alpha = image_io.resize_alpha(
                        source_document.alpha, result.shape[:2]
                    )
                    if cancel_event.is_set():
                        raise ProcessingCancelled("Image processing was cancelled")
                    result = image_io.unpremultiply_rgb(result, result_alpha)
                    if cancel_event.is_set():
                        raise ProcessingCancelled("Image processing was cancelled")
                result_document = source_document.with_pixels(result, result_alpha)
                self._after(
                    self._on_complete,
                    job_id,
                    result_document,
                    scale,
                    time.monotonic() - started,
                )
            except ProcessingCancelled:
                self._after(self._on_cancelled, job_id)
            except MemoryError:
                traceback.print_exc()
                self._after(self._on_error, job_id, t("error_out_of_memory"))
            except Exception as exc:  # noqa: BLE001 - worker boundary reports all failures
                traceback.print_exc()
                self._after(self._on_error, job_id, str(exc))

        self._worker = threading.Thread(
            target=run, name=f"super-resolution-{job_id}", daemon=True
        )
        self._worker.start()

    def cancel_processing(self) -> None:
        if not self.processing or self._cancel_event is None:
            return
        self._cancel_event.set()
        self.cancel_btn.config(state=tk.DISABLED)
        self.status_text.set(t("status_cancelling"))

    def _on_progress(self, job_id: int, fraction: float) -> None:
        if job_id != self._job_id or not self.processing:
            return
        fraction = min(max(float(fraction), 0.0), 1.0)
        self.progress.config(value=fraction * 100.0)
        self.status_text.set(t("status_progress", pct=int(fraction * 100.0)))

    def _finish_job(self) -> None:
        self.processing = False
        self._cancel_event = None
        self._worker = None
        self._update_action_states()

    def _on_complete(
        self,
        job_id: int,
        document: image_io.ImageDocument,
        scale: int,
        elapsed: float,
    ) -> None:
        if job_id != self._job_id or self._closing:
            return
        self.result_document = document
        self.result_image = document.rgb
        self.progress.config(value=100)
        result_height, result_width = document.rgb.shape[:2]
        self.status_text.set(
            t(
                "status_complete",
                w=result_width,
                h=result_height,
                scale=scale,
                elapsed=elapsed,
            )
        )
        self._finish_job()
        self._redraw_all(use_low_quality=False)

    def _on_cancelled(self, job_id: int) -> None:
        if job_id != self._job_id or self._closing:
            return
        self.progress.config(value=0)
        self.status_text.set(t("status_cancelled"))
        self._finish_job()

    def _on_error(self, job_id: int, message: str) -> None:
        if job_id != self._job_id or self._closing:
            return
        self.progress.config(value=0)
        self.status_text.set(t("status_error"))
        self._finish_job()
        messagebox.showerror(
            t("dialog_error_title"), t("dialog_process_failed", error=message)
        )

    def save_result(self) -> None:
        if self.result_document is None or self.processing:
            return
        path = filedialog.asksaveasfilename(
            title=t("dialog_save_title"),
            defaultextension=".png",
            filetypes=[
                (t("filetype_png"), "*.png"),
                (t("filetype_tiff"), ("*.tif", "*.tiff")),
                (t("filetype_jpeg"), ("*.jpg", "*.jpeg")),
                (t("filetype_bmp"), "*.bmp"),
            ],
        )
        if not path:
            return
        try:
            image_io.save_image_document(self.result_document, path)
            self.status_text.set(t("status_saved", name=Path(path).name))
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                t("dialog_error_title"), t("dialog_save_failed", error=str(exc))
            )

    def _show_about(self) -> None:
        messagebox.showinfo(t("dialog_about_title"), t("dialog_about_text"))

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._redraw_after is not None:
            try:
                self.root.after_cancel(self._redraw_after)
            except tk.TclError:
                pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass
