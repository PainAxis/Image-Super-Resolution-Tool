"""Tests for pre-allocation pixel and memory limits."""

from pathlib import Path

import pytest

from sr_tool.utils import resources


def test_estimator_accounts_for_scale_squared() -> None:
    two = resources.estimate_peak_resources(100, 200, 2)
    four = resources.estimate_peak_resources(100, 200, 4)
    assert two.output_pixels == 80_000
    assert four.output_pixels == 320_000
    assert four.peak_bytes > two.peak_bytes


def test_input_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SR_TOOL_MAX_INPUT_PIXELS", "99")
    with pytest.raises(resources.ResourceLimitError, match="100 pixels"):
        resources.ensure_input_size(10, 10)


def test_output_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SR_TOOL_MAX_OUTPUT_PIXELS", "399")
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)
    with pytest.raises(resources.ResourceLimitError, match="400 pixels"):
        resources.ensure_pipeline_budget(10, 10, 2)


def test_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SR_TOOL_MAX_MEMORY_MIB", "1")
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: None)
    with pytest.raises(resources.ResourceLimitError, match="Estimated peak memory"):
        resources.ensure_pipeline_budget(10, 10, 2)


def test_bad_environment_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SR_TOOL_MAX_INPUT_PIXELS", "many")
    with pytest.raises(resources.ResourceLimitError, match="must be an integer"):
        resources.max_input_pixels()


def test_cgroup_v2_remaining_memory_is_detected(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("1000\n", encoding="ascii")
    (tmp_path / "memory.current").write_text("375\n", encoding="ascii")
    assert resources._cgroup_available_bytes(tmp_path) == 625


def test_unlimited_cgroup_counter_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "memory.max").write_text("max\n", encoding="ascii")
    (tmp_path / "memory.current").write_text("375\n", encoding="ascii")
    assert resources._cgroup_available_bytes(tmp_path) is None
