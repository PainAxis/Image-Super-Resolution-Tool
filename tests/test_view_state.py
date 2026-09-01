"""Tests for normalized, synchronized comparison navigation."""

import pytest

from sr_tool.gui.view_state import ViewState


def test_original_and_result_have_same_display_extent() -> None:
    view = ViewState()
    view.reset(100, 80)
    view.zoom = 2.0
    source = view.image_geometry(500, 400, 100, 80)
    result = view.image_geometry(500, 400, 400, 320)
    source_zoom_x, source_zoom_y, source_x, source_y = source
    result_zoom_x, result_zoom_y, result_x, result_y = result
    assert source_x == result_x
    assert source_y == result_y
    assert source_zoom_x * 100 == pytest.approx(result_zoom_x * 400)
    assert source_zoom_y * 80 == pytest.approx(result_zoom_y * 320)


def test_different_canvas_sizes_do_not_shift_the_shared_center() -> None:
    view = ViewState()
    view.reset(100, 100)
    view.zoom = 10.0
    view.center_x = 95.0
    view.clamp(400, 100)

    narrow = view.image_geometry(100, 100, 100, 100)
    wide = view.image_geometry(400, 100, 400, 400)

    assert view.center_x == 80.0
    narrow_center = (50.0 - narrow[2]) / narrow[0]
    wide_center = (200.0 - wide[2]) / wide[0] / 4.0
    assert narrow_center == pytest.approx(80.0)
    assert wide_center == pytest.approx(80.0)


def test_zoom_keeps_cursor_on_same_source_point() -> None:
    view = ViewState()
    view.reset(1000, 800)
    view.zoom = 1.0
    canvas = (500, 400)
    cursor = (123, 234)
    origin_before = view.origin(*canvas)
    point_before = (
        (cursor[0] - origin_before[0]) / view.zoom,
        (cursor[1] - origin_before[1]) / view.zoom,
    )
    view.zoom_at(1.7, *cursor, *canvas)
    origin_after = view.origin(*canvas)
    point_after = (
        (cursor[0] - origin_after[0]) / view.zoom,
        (cursor[1] - origin_after[1]) / view.zoom,
    )
    assert point_after == pytest.approx(point_before)


def test_pan_and_clamp_prevent_empty_margins() -> None:
    view = ViewState()
    view.reset(100, 100)
    view.zoom = 10.0
    view.pan(100_000, -100_000, 200, 200)
    assert view.center_x == pytest.approx(10.0)
    assert view.center_y == pytest.approx(90.0)


def test_fit_centers_content() -> None:
    view = ViewState()
    view.reset(400, 200)
    view.fit(200, 200)
    assert view.zoom == pytest.approx(0.475)
    assert view.center_x == 200
    assert view.center_y == 100
