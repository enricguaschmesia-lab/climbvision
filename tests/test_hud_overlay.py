"""Tests for src/visualization/hud_overlay.py."""

from __future__ import annotations

import numpy as np

from src.visualization.hud_overlay import render_hud


HUD_CONFIG: dict = {
    "font_scale": 0.6,
    "background_alpha": 0.5,
    "position": "top_right",
}

ANGLES_CONFIG: dict = {
    "efficient_color": [60, 220, 60],
    "warning_color": [0, 180, 255],
    "error_color": [20, 20, 240],
    "efficient_range": [150, 180],
    "warning_range": [90, 150],
}


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic blank frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_angles() -> dict[str, float]:
    """Create representative joint angles for HUD rendering."""
    return {
        "left_elbow": 145.0,
        "right_elbow": 138.0,
        "left_shoulder": 120.0,
        "right_shoulder": 115.0,
        "left_hip": 160.0,
        "right_hip": 155.0,
        "left_knee": 170.0,
        "right_knee": 165.0,
    }


class TestRenderHud:
    """Tests for render_hud()."""

    def test_render_hud_returns_new_frame(self) -> None:
        frame = _make_frame()
        original = frame.copy()

        result = render_hud(frame, 12, 0.4, _make_angles(), HUD_CONFIG, ANGLES_CONFIG)

        np.testing.assert_array_equal(frame, original)
        assert result is not frame

    def test_render_hud_shape_unchanged(self) -> None:
        frame = _make_frame()
        result = render_hud(frame, 12, 0.4, _make_angles(), HUD_CONFIG, ANGLES_CONFIG)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_render_hud_with_angles(self) -> None:
        frame = _make_frame()
        result = render_hud(frame, 12, 0.4, _make_angles(), HUD_CONFIG, ANGLES_CONFIG)
        assert not np.array_equal(result, frame)

    def test_render_hud_no_angles(self) -> None:
        frame = _make_frame()
        result = render_hud(frame, 12, 0.4, None, HUD_CONFIG, ANGLES_CONFIG)
        assert result.shape == frame.shape

    def test_render_hud_positions(self) -> None:
        frame = _make_frame()

        for position in ["top_right", "top_left", "bottom_right"]:
            config = {**HUD_CONFIG, "position": position}
            result = render_hud(frame, 12, 0.4, _make_angles(), config, ANGLES_CONFIG)
            assert result.shape == frame.shape
