"""Main application entry point and Application class."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import threading
import time
import numpy as np
from pathlib import Path

from sr_tool.utils import image_io
from sr_tool.fsr.pipeline import process_image as sr_process
from sr_tool.locale.i18n import t, set_language, on_language_change, \
    available_languages, current_language


class Application:
    """Image Super-Resolution application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(t("app_title"))
        self.root.geometry("1200x700")
        self.root.minsize(800, 500)

        # State
        self.source_image = None
        self.result_image = None
        self.scale_factor = tk.IntVar(value=2)
        self.sharpness = tk.DoubleVar(value=0.25)
        self.antialias = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value=t("status_ready"))
        self.processing = False

        # Widgets that need text updates on language change
        self._live_widgets: list[tuple[tk.Widget, str, str]] = []

        self._build_menu()
        self._build_layout()
        self._bind_shortcuts()

        on_language_change(self._on_language_changed)

    def _after(self, func, *args):
        """Schedule *func* on the main thread (convenience wrapper)."""
        self.root.after(0, func, *args)

    def _on_language_changed(self):
        """Refresh all live-widget texts after a language switch."""
        self.root.title(t("app_title"))
        self._rebuild_menu()

        for widget, attr, key, fmt in self._live_widgets:
            try:
                text = t(key, **fmt) if fmt else t(key)
                if widget.winfo_exists():
                    widget[attr] = text
            except tk.TclError:
                pass

        # Refresh status bar if it holds a dynamic key
        if not self.processing:
            if self.source_image is None:
                self.status_text.set(t("status_ready"))
            else:
                self.status_text.set(
                    t("status_loaded",
                      name=Path(self._last_path).name if hasattr(self, "_last_path") else "?",
                      w=self.source_image.shape[1],
                      h=self.source_image.shape[0]))

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _rebuild_menu(self):
        """Destroy and rebuild the menu bar (used after language switch)."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self._populate_menu(menubar)

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        self._populate_menu(menubar)

    def _populate_menu(self, menubar: tk.Menu):
        # --- File ---
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(
            label=t("menu_open"), command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(
            label=t("menu_save"), command=self.save_result, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label=t("menu_exit"), command=self.root.quit)
        menubar.add_cascade(label=t("menu_file"), menu=file_menu)

        # --- View ---
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label=t("menu_fit_window"), command=self._fit_to_window)
        view_menu.add_command(label=t("menu_zoom_100"), command=self._zoom_100)
        menubar.add_cascade(label=t("menu_view"), menu=view_menu)

        # --- Language ---
        lang_menu = tk.Menu(menubar, tearoff=0)
        # Shared variable so radiobuttons are mutually exclusive
        self.lang_var = tk.StringVar(value=current_language())
        for code, name in available_languages().items():
            lang_menu.add_radiobutton(
                label=name, value=code, variable=self.lang_var,
                command=lambda c=code: set_language(c))
        menubar.add_cascade(label=t("menu_language"), menu=lang_menu)

        # --- Help ---
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label=t("menu_about"), command=self._show_about)
        menubar.add_cascade(label=t("menu_help"), menu=help_menu)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        # --- Control bar (two rows) ---
        control_frame = tk.Frame(self.root, padx=8, pady=6)
        control_frame.pack(fill=tk.X)

        # Row 1: main actions
        row1 = tk.Frame(control_frame)
        row1.pack(fill=tk.X)

        open_btn = tk.Button(row1, text=t("btn_open"), command=self.open_image)
        open_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((open_btn, "text", "btn_open", {}))

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        scale_lbl = tk.Label(row1, text=t("lbl_scale"))
        scale_lbl.pack(side=tk.LEFT)
        self._live_widgets.append((scale_lbl, "text", "lbl_scale", {}))

        scale_menu = tk.OptionMenu(row1, self.scale_factor, 2, 3, 4)
        scale_menu.config(width=4)
        scale_menu.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.process_btn = tk.Button(
            row1, text=t("btn_process"), command=self.process_image,
            state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.process_btn, "text", "btn_process", {}))

        self.save_btn = tk.Button(
            row1, text=t("btn_save"), command=self.save_result,
            state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((self.save_btn, "text", "btn_save", {}))

        # Row 2: sharpness + anti-alias
        row2 = tk.Frame(control_frame)
        row2.pack(fill=tk.X, pady=(4, 0))

        sharp_lbl = tk.Label(row2, text=t("lbl_sharpness"))
        sharp_lbl.pack(side=tk.LEFT, padx=(0, 2))
        self._live_widgets.append((sharp_lbl, "text", "lbl_sharpness", {}))

        sharp_scale = tk.Scale(
            row2, from_=0.0, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.sharpness,
            length=140, showvalue=True)
        sharp_scale.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8)

        aa_chk = tk.Checkbutton(
            row2, text=t("chk_antialias"), variable=self.antialias)
        aa_chk.pack(side=tk.LEFT, padx=2)
        self._live_widgets.append((aa_chk, "text", "chk_antialias", {}))

        # Image display area
        pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
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

        self._bind_canvas_events(self.left_canvas)
        self._bind_canvas_events(self.right_canvas)

        # Canvas state
        self._canvas_images = [None, None]
        self._pan_start = [None, None]
        self._offset = [(0, 0), (0, 0)]
        self._zoom = [1.0, 1.0]
        self._interacting = [False, False]    # True during drag/zoom
        self._redraw_after = [None, None]     # scheduled high-quality redraw IDs

        # Status bar
        status_frame = tk.Frame(self.root, padx=8, pady=4)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(status_frame, textvariable=self.status_text, anchor=tk.W).pack(
            fill=tk.X, side=tk.LEFT)
        self.progress = ttk.Progressbar(status_frame, mode="determinate", length=200)
        self.progress.pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------
    def _bind_shortcuts(self):
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_result())
        self.root.bind("<Control-q>", lambda e: self.root.quit())

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------
    def _bind_canvas_events(self, canvas: tk.Canvas):
        idx = 0 if canvas is self.left_canvas else 1
        canvas.bind("<ButtonPress-1>", lambda e: self._on_pan_start(e, idx))
        canvas.bind("<B1-Motion>", lambda e: self._on_pan_drag(e, idx))
        canvas.bind("<ButtonRelease-1>", lambda e: self._on_pan_end(e, idx))
        canvas.bind("<MouseWheel>", lambda e: self._on_zoom(e, idx))
        canvas.bind("<Configure>", lambda e: self._redraw(idx))

    def _on_pan_start(self, event, idx):
        self._pan_start[idx] = (event.x, event.y)
        self._interacting[idx] = True

    def _on_pan_drag(self, event, idx):
        if self._pan_start[idx] is None:
            return
        dx = event.x - self._pan_start[idx][0]
        dy = event.y - self._pan_start[idx][1]
        ox, oy = self._offset[idx]
        self._offset[idx] = (ox + dx, oy + dy)
        self._pan_start[idx] = (event.x, event.y)
        self._redraw(idx)  # fast path while dragging

    def _on_pan_end(self, event, idx):
        self._interacting[idx] = False
        self._pan_start[idx] = None
        self._redraw_high_quality(idx)

    def _on_zoom(self, event, idx):
        old_zoom = self._zoom[idx]
        factor = 1.1 if event.delta > 0 else 0.9
        new_zoom = max(0.05, min(old_zoom * factor, 20.0))
        factor = new_zoom / old_zoom  # actual clamped factor

        self._zoom[idx] = new_zoom

        # Adjust offset so the pixel under the cursor stays in place
        ox, oy = self._offset[idx]
        mx, my = event.x, event.y
        self._offset[idx] = (
            mx - (mx - ox) * factor,
            my - (my - oy) * factor,
        )

        self._interacting[idx] = True
        self._redraw(idx)  # fast path during zoom

        # Schedule high-quality redraw after zoom stops
        if self._redraw_after[idx] is not None:
            self.root.after_cancel(self._redraw_after[idx])
        self._redraw_after[idx] = self.root.after(200, self._redraw_high_quality, idx)

    def _redraw_high_quality(self, idx):
        """Final high-quality redraw after interaction ends."""
        self._interacting[idx] = False
        self._redraw_after[idx] = None
        self._redraw(idx, use_low_quality=False)

    def _redraw(self, idx, use_low_quality=None):
        """Redraw canvas *idx*. If *use_low_quality* is None, auto-detect
        from interaction state (fast NEAREST during drag, LANCZOS otherwise).
        """
        if use_low_quality is None:
            use_low_quality = self._interacting[idx]

        canvas = self.left_canvas if idx == 0 else self.right_canvas
        image_arr = self.source_image if idx == 0 else self.result_image
        canvas.delete("all")

        if image_arr is None:
            return

        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 1 or h <= 1:
            return

        zoom = self._zoom[idx]
        ox, oy = self._offset[idx]

        img_h, img_w = image_arr.shape[:2]
        disp_w = img_w * zoom
        disp_h = img_h * zoom

        if disp_w < w:
            ox = (w - disp_w) / 2  # center when image fits
        else:
            ox = max(w - disp_w, min(0, ox))  # clamp when image overflows

        if disp_h < h:
            oy = (h - disp_h) / 2
        else:
            oy = max(h - disp_h, min(0, oy))

        self._offset[idx] = (ox, oy)

        src_x1 = max(0, -ox / zoom)
        src_y1 = max(0, -oy / zoom)
        src_x2 = min(img_w, (w - ox) / zoom)
        src_y2 = min(img_h, (h - oy) / zoom)

        # Use floor/ceil so extreme zoom never produces a zero-dimension slice.
        # Enforce at least 1 source pixel to keep Pillow happy.
        x1 = max(0, math.floor(src_x1))
        y1 = max(0, math.floor(src_y1))
        x2 = min(img_w, max(x1 + 1, math.ceil(src_x2)))
        y2 = min(img_h, max(y1 + 1, math.ceil(src_y2)))

        if x2 <= x1 or y2 <= y1:
            return

        crop = image_arr[y1:y2, x1:x2, :]
        display_w = int(max(1, (src_x2 - src_x1) * zoom))
        display_h = int(max(1, (src_y2 - src_y1) * zoom))

        from PIL import Image as PILImage
        pil_crop = PILImage.fromarray(np.rint(np.clip(crop, 0, 1) * 255.0).astype(np.uint8))
        if display_w > 0 and display_h > 0:
            filter_mode = PILImage.Resampling.NEAREST if use_low_quality else PILImage.Resampling.LANCZOS
            pil_crop = pil_crop.resize((display_w, display_h), filter_mode)

        from PIL import ImageTk
        self._canvas_images[idx] = ImageTk.PhotoImage(pil_crop)

        canvas.create_image(
            ox + (x1 * zoom),
            oy + (y1 * zoom),
            anchor=tk.NW, image=self._canvas_images[idx])

    # ------------------------------------------------------------------
    # Fit / zoom helpers
    # ------------------------------------------------------------------
    def _fit_to_window(self):
        if self.source_image is not None:
            canvas = self.left_canvas
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            ih, iw = self.source_image.shape[:2]
            fit_zoom = min(cw / iw, ch / ih) * 0.95
            self._zoom = [fit_zoom, fit_zoom]
            self._offset = [(0, 0), (0, 0)]
            self._interacting = [False, False]
            self._redraw(0, use_low_quality=False)
            self._redraw(1, use_low_quality=False)

    def _zoom_100(self):
        self._zoom = [1.0, 1.0]
        self._offset = [(0, 0), (0, 0)]
        self._interacting = [False, False]
        self._redraw(0, use_low_quality=False)
        self._redraw(1, use_low_quality=False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_image(self):
        path = filedialog.askopenfilename(
            title=t("dialog_open_title"),
            filetypes=[
                (t("filetype_images"),
                 ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif")),
                (t("filetype_all"), "*.*"),
            ])
        if not path:
            return

        try:
            self.source_image = image_io.load_image(path)
            self.result_image = None
            self._last_path = path
            self._zoom = [1.0, 1.0]
            self._offset = [(0, 0), (0, 0)]
            self.status_text.set(
                t("status_loaded",
                  name=Path(path).name,
                  w=self.source_image.shape[1],
                  h=self.source_image.shape[0]))
            self.process_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self._fit_to_window()
        except Exception as e:
            messagebox.showerror(t("dialog_error_title"),
                                 t("dialog_load_failed", error=str(e)))

    def process_image(self):
        if self.source_image is None or self.processing:
            return

        scale = self.scale_factor.get()
        sharpness = self.sharpness.get()
        use_aa = self.antialias.get()
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress.config(mode="determinate", value=0)
        self.status_text.set(t("status_processing", scale=scale))

        # Throttle: skip UI updates within MIN_PROGRESS_INTERVAL seconds
        MIN_PROGRESS_INTERVAL = 0.033  # ~30 fps
        last_update = [0.0]  # mutable for closure

        def progress_cb(frac):
            now = time.time()
            if now - last_update[0] < MIN_PROGRESS_INTERVAL and frac < 1.0:
                return
            last_update[0] = now
            self._after(self._on_progress, frac)

        def run():
            start_time = time.time()
            try:
                result = sr_process(
                    self.source_image, scale,
                    rcas_sharpness=sharpness,
                    antialias=use_aa,
                    progress_callback=progress_cb,
                )
                elapsed = time.time() - start_time
                self._after(self._on_complete, result, scale, elapsed)
            except MemoryError:
                import traceback as _tb
                _tb.print_exc()  # log full traceback for OOM debugging
                self._after(self._on_error,
                            "Out of memory. The image is too large for the\n"
                            "available RAM. Try a smaller image or lower\n"
                            "scale factor.",
                            is_fatal=True)
            except Exception as e:
                import traceback as _tb
                _tb.print_exc()
                self._after(self._on_error, f"{e}\n\n{_tb.format_exc()}")

        threading.Thread(target=run, daemon=True).start()

    def _on_progress(self, frac):
        self.progress.config(value=frac * 100)
        self.status_text.set(t("status_progress", pct=int(frac * 100)))

    def _on_complete(self, result, scale, elapsed):
        self.result_image = result
        self._zoom[1] = self._zoom[0]
        # Sync offset so the result view shows the same region the user was
        # inspecting in the original — preserves A/B comparison continuity.
        self._offset[1] = self._offset[0]
        self._redraw(1)
        rh, rw = result.shape[:2]
        self.status_text.set(
            t("status_complete", w=rw, h=rh, scale=scale, elapsed=elapsed))
        self.save_btn.config(state=tk.NORMAL)
        self.progress.config(value=100)
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)

    def _on_error(self, msg, is_fatal=False):
        if is_fatal:
            messagebox.showerror(
                t("dialog_error_title"), msg)
        else:
            messagebox.showerror(
                t("dialog_error_title"),
                t("dialog_process_failed", error=msg))
        self.status_text.set(t("status_error"))
        self.progress.config(value=0)
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)

    def save_result(self):
        if self.result_image is None:
            return
        path = filedialog.asksaveasfilename(
            title=t("dialog_save_title"),
            defaultextension=".png",
            filetypes=[
                (t("filetype_png"), "*.png"),
                (t("filetype_jpeg"), "*.jpg"),
                (t("filetype_bmp"), "*.bmp"),
            ])
        if not path:
            return
        try:
            image_io.save_image(self.result_image, path)
            self.status_text.set(t("status_saved", name=Path(path).name))
        except Exception as e:
            messagebox.showerror(t("dialog_error_title"),
                                 t("dialog_save_failed", error=str(e)))

    def _show_about(self):
        messagebox.showinfo(t("dialog_about_title"), t("dialog_about_text"))
