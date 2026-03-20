"""Tests for render_dynamic_hud and compute_stress_colors in skeleton_overlay.py."""

from __future__ import annotations

import numpy as np

from src.visualization.skeleton_overlay import (
    compute_stress_colors,
    render_dynamic_hud,
)


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 50, dtype=np.uint8)


ANGLES_CONFIG: dict = {
    "efficient_color": [60, 220, 60],
    "warning_color": [0, 180, 255],
    "error_color": [20, 20, 240],
    "efficient_range": [150, 180],
    "warning_range": [90, 150],
}

TRIPLETS: dict[str, tuple[int, int, int]] = {
    "left_elbow": (15, 13, 11),
    "right_elbow": (16, 14, 12),
}


class TestComputeStressColors:
    """Tests for compute_stress_colors()."""

    def test_efficient_angle_gets_green(self) -> None:
        colors = compute_stress_colors(
            {"left_elbow": 170.0}, TRIPLETS, ANGLES_CONFIG,
        )
        assert colors[13] == (60, 220, 60)

    def test_warning_angle_gets_amber(self) -> None:
        colors = compute_stress_colors(
            {"left_elbow": 120.0}, TRIPLETS, ANGLES_CONFIG,
        )
        assert colors[13] == (0, 180, 255)

    def test_error_angle_gets_red(self) -> None:
        colors = compute_stress_colors(
            {"left_elbow": 45.0}, TRIPLETS, ANGLES_CONFIG,
        )
        assert colors[13] == (20, 20, 240)

    def test_unknown_joint_ignored(self) -> None:
        colors = compute_stress_colors(
            {"fake_joint": 45.0}, TRIPLETS, ANGLES_CONFIG,
        )
        assert colors == {}

    def test_multiple_joints(self) -> None:
        colors = compute_stress_colors(
            {"left_elbow": 170.0, "right_elbow": 45.0},
            TRIPLETS, ANGLES_CONFIG,
        )
        assert 13 in colors
        assert 14 in colors


class TestRenderDynamicHud:
    """Tests for render_dynamic_hud()."""

    def test_no_events_returns_copy(self) -> None:
        frame = _make_frame()
        result = render_dynamic_hud(frame, [])
        assert result is not frame
        np.testing.assert_array_equal(result, frame)

    def test_with_events_modifies_frame(self) -> None:
        frame = _make_frame()
        events = [{"joint": "left_elbow", "angle": 45.0, "type": "hold"}]
        result = render_dynamic_hud(frame, events)
        assert not np.array_equal(result, frame)

    def test_does_not_mutate_input(self) -> None:
        frame = _make_frame()
        original = frame.copy()
        events = [{"joint": "left_elbow", "angle": 45.0, "type": "hold"}]
        render_dynamic_hud(frame, events)
        np.testing.assert_array_equal(frame, original)

    def test_output_shape_unchanged(self) -> None:
        frame = _make_frame()
        events = [{"joint": "left_elbow", "angle": 45.0, "type": "hold"}]
        result = render_dynamic_hud(frame, events)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_multiple_events(self) -> None:
        frame = _make_frame()
        events = [
            {"joint": "left_elbow", "angle": 45.0, "type": "hold"},
            {"joint": "right_elbow", "angle": 50.0, "type": "hold"},
        ]
        result = render_dynamic_hud(frame, events)
        assert result.shape == frame.shape
