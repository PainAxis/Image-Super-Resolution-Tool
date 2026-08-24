"""Pure synchronized-view geometry used by both comparison canvases."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ViewState:
    """A view expressed in original-image coordinates.

    One display pixel per original pixel is ``zoom == 1``. Upscaled result
    pixels use a proportionally smaller pixel zoom, so both canvases always
    show the same logical content rectangle.
    """

    source_width: int = 1
    source_height: int = 1
    center_x: float = 0.5
    center_y: float = 0.5
    zoom: float = 1.0

    def reset(self, source_width: int, source_height: int) -> None:
        if source_width < 1 or source_height < 1:
            raise ValueError("source dimensions must be positive")
        self.source_width = source_width
        self.source_height = source_height
        self.center_x = source_width / 2.0
        self.center_y = source_height / 2.0
        self.zoom = 1.0

    def fit(self, canvas_width: int, canvas_height: int, margin: float = 0.95) -> None:
        if canvas_width < 1 or canvas_height < 1:
            return
        self.zoom = max(
            0.01,
            min(
                canvas_width / self.source_width,
                canvas_height / self.source_height,
            )
            * margin,
        )
        self.center_x = self.source_width / 2.0
        self.center_y = self.source_height / 2.0

    def origin(self, canvas_width: int, canvas_height: int) -> tuple[float, float]:
        return (
            canvas_width / 2.0 - self.center_x * self.zoom,
            canvas_height / 2.0 - self.center_y * self.zoom,
        )

    def clamp(self, canvas_width: int, canvas_height: int) -> None:
        displayed_width = self.source_width * self.zoom
        displayed_height = self.source_height * self.zoom
        if displayed_width <= canvas_width:
            self.center_x = self.source_width / 2.0
        else:
            half_visible = canvas_width / (2.0 * self.zoom)
            self.center_x = min(
                self.source_width - half_visible,
                max(half_visible, self.center_x),
            )
        if displayed_height <= canvas_height:
            self.center_y = self.source_height / 2.0
        else:
            half_visible = canvas_height / (2.0 * self.zoom)
            self.center_y = min(
                self.source_height - half_visible,
                max(half_visible, self.center_y),
            )

    def pan(self, dx: float, dy: float, canvas_width: int, canvas_height: int) -> None:
        self.center_x -= dx / self.zoom
        self.center_y -= dy / self.zoom
        self.clamp(canvas_width, canvas_height)

    def zoom_at(
        self,
        factor: float,
        cursor_x: float,
        cursor_y: float,
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        if factor <= 0.0:
            raise ValueError("zoom factor must be positive")
        origin_x, origin_y = self.origin(canvas_width, canvas_height)
        source_x = (cursor_x - origin_x) / self.zoom
        source_y = (cursor_y - origin_y) / self.zoom
        self.zoom = min(40.0, max(0.01, self.zoom * factor))
        self.center_x = source_x - (cursor_x - canvas_width / 2.0) / self.zoom
        self.center_y = source_y - (cursor_y - canvas_height / 2.0) / self.zoom
        self.clamp(canvas_width, canvas_height)

    def image_geometry(
        self,
        canvas_width: int,
        canvas_height: int,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        """Return image-pixel zoom x/y and common display origin x/y."""
        if image_width < 1 or image_height < 1:
            raise ValueError("image dimensions must be positive")
        self.clamp(canvas_width, canvas_height)
        scale_x = image_width / self.source_width
        scale_y = image_height / self.source_height
        origin_x, origin_y = self.origin(canvas_width, canvas_height)
        return self.zoom / scale_x, self.zoom / scale_y, origin_x, origin_y
