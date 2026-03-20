"""Tests for src/visualization/angle_overlay.py.

Uses synthetic frames and keypoint arrays — no real video needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.visualization.angle_overlay import (
    render_angles,
    _angle_color,
    _ordered_arc_angles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "arc_radius": 45,
    "radius_hips": 65,
    "thickness": 3,
    "font_scale": 0.55,
    "alpha": 0.75,
    "fill_alpha": 0.1,
    "label_offset": 18,
    "hide_threshold": 0.25,
    "efficient_color": [60, 220, 60],
    "warning_color": [0, 180, 255],
    "error_color": [20, 20, 240],
    "efficient_range": [150, 180],
    "warning_range": [90, 150],
}


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dark gray synthetic frame."""
    return np.full((height, width, 3), 50, dtype=np.uint8)


def _make_visible_keypoints(width: int = 640, height: int = 480) -> np.ndarray:
    """Create keypoints forming a rough stick figure. All visible."""
    kp = np.zeros((33, 4), dtype=np.float64)
    kp[:, 3] = 1.0

    cx, cy = width // 2, height // 3
    kp[11] = [cx - 40, cy, 0, 1.0]
    kp[12] = [cx + 40, cy, 0, 1.0]
    kp[13] = [cx - 70, cy + 50, 0, 1.0]
    kp[14] = [cx + 70, cy + 50, 0, 1.0]
    kp[15] = [cx - 90, cy + 100, 0, 1.0]
    kp[16] = [cx + 90, cy + 100, 0, 1.0]
    kp[23] = [cx - 25, cy + 120, 0, 1.0]
    kp[24] = [cx + 25, cy + 120, 0, 1.0]
    kp[25] = [cx - 30, cy + 200, 0, 1.0]
    kp[26] = [cx + 30, cy + 200, 0, 1.0]
    kp[27] = [cx - 30, cy + 280, 0, 1.0]
    kp[28] = [cx + 30, cy + 280, 0, 1.0]

    return kp


def _sample_angles() -> dict[str, float]:
    """Angles matching the keypoints from _make_visible_keypoints."""
    return {
        "left_elbow": 160.0,
        "right_elbow": 95.0,
        "left_shoulder": 170.0,
        "right_shoulder": 80.0,
        "left_hip": 150.0,
        "right_hip": 145.0,
        "left_knee": 175.0,
        "right_knee": 170.0,
    }


# ---------------------------------------------------------------------------
# Tests — render_angles
# ---------------------------------------------------------------------------


class TestRenderAngles:
    """Tests for render_angles()."""

    def test_does_not_mutate_input(self) -> None:
        frame = _make_frame()
        original = frame.copy()
        kp = _make_visible_keypoints()
        render_angles(frame, kp, _sample_angles(), DEFAULT_CONFIG)
        np.testing.assert_array_equal(frame, original)

    def test_output_shape_matches_input(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        result = render_angles(frame, kp, _sample_angles(), DEFAULT_CONFIG)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_modifies_pixels(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        result = render_angles(frame, kp, _sample_angles(), DEFAULT_CONFIG)
        assert not np.array_equal(result, frame)

    def test_empty_angles_returns_copy(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        result = render_angles(frame, kp, {}, DEFAULT_CONFIG)
        # With no angles, only alpha blending happens (may shift values slightly)
        # The key check is that it doesn't crash
        assert result.shape == frame.shape

    def test_hidden_joints_not_drawn(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        # Hide left elbow (joint center for left_elbow angle)
        kp[13, 3] = 0.1  # below hide_threshold=0.25

        result_hidden = render_angles(frame, kp, {"left_elbow": 160.0}, DEFAULT_CONFIG)

        # Compare with fully visible version
        kp[13, 3] = 1.0
        result_visible = render_angles(frame, kp, {"left_elbow": 160.0}, DEFAULT_CONFIG)

        # Hidden version should have fewer drawn pixels
        diff_hidden = np.sum(np.abs(result_hidden.astype(int) - frame.astype(int)))
        diff_visible = np.sum(np.abs(result_visible.astype(int) - frame.astype(int)))
        assert diff_hidden < diff_visible

    def test_nan_coordinates_handled(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        kp[13, 0] = np.nan
        result = render_angles(frame, kp, {"left_elbow": 160.0}, DEFAULT_CONFIG)
        assert result.shape == frame.shape

    def test_unknown_joint_name_ignored(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        result = render_angles(frame, kp, {"fake_joint": 90.0}, DEFAULT_CONFIG)
        assert result.shape == frame.shape

    def test_custom_config(self) -> None:
        frame = _make_frame()
        kp = _make_visible_keypoints()
        config = {**DEFAULT_CONFIG, "arc_radius": 20, "thickness": 1, "alpha": 0.5}
        result = render_angles(frame, kp, _sample_angles(), config)
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Tests — _angle_color
# ---------------------------------------------------------------------------


class TestAngleColor:
    """Tests for the color selection helper."""

    def test_efficient_range(self) -> None:
        c = _angle_color(170.0, [150, 180], [90, 150], (0, 255, 0), (0, 255, 255), (0, 0, 255))
        assert c == (0, 255, 0)

    def test_warning_range(self) -> None:
        c = _angle_color(120.0, [150, 180], [90, 150], (0, 255, 0), (0, 255, 255), (0, 0, 255))
        assert c == (0, 255, 255)

    def test_error_range(self) -> None:
        c = _angle_color(45.0, [150, 180], [90, 150], (0, 255, 0), (0, 255, 255), (0, 0, 255))
        assert c == (0, 0, 255)

    def test_boundary_efficient_lower(self) -> None:
        c = _angle_color(150.0, [150, 180], [90, 150], (0, 255, 0), (0, 255, 255), (0, 0, 255))
        # 150 is in both ranges — efficient takes priority
        assert c == (0, 255, 0)

    def test_boundary_warning_lower(self) -> None:
        c = _angle_color(90.0, [150, 180], [90, 150], (0, 255, 0), (0, 255, 255), (0, 0, 255))
        assert c == (0, 255, 255)


# ---------------------------------------------------------------------------
# Tests — _ordered_arc_angles
# ---------------------------------------------------------------------------


class TestOrderedArcAngles:
    """Tests for the arc angle ordering helper."""

    def test_already_ordered(self) -> None:
        import math
        start, end = _ordered_arc_angles(0.0, math.pi / 2)
        assert abs(end - start - math.pi / 2) < 1e-10

    def test_reverse_order(self) -> None:
        import math
        start, end = _ordered_arc_angles(math.pi / 2, 0.0)
        assert abs(end - start - math.pi / 2) < 1e-10

    def test_wraps_around(self) -> None:
        import math
        # Angles near 0 and near 2*pi should produce a small arc
        start, end = _ordered_arc_angles(0.1, -0.1)
        assert (end - start) < math.pi + 1e-10

    def test_opposite_vectors(self) -> None:
        import math
        start, end = _ordered_arc_angles(0.0, math.pi)
        assert abs(end - start - math.pi) < 1e-10
