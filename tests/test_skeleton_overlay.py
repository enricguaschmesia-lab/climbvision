"""Tests for src/visualization/skeleton_overlay.py.

Uses synthetic frames and keypoint arrays — no real video needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.visualization.skeleton_overlay import (
    JOINT_LANDMARKS,
    SKELETON_CONNECTIONS,
    render_skeleton,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "thickness": 2,
    "joint_radius": 4,
    "low_confidence_alpha": 0.3,
    "color": [0, 255, 128],
}


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dark gray synthetic frame."""
    return np.full((height, width, 3), 50, dtype=np.uint8)


def _make_visible_keypoints(width: int = 640, height: int = 480) -> np.ndarray:
    """Create keypoints forming a rough stick figure in the frame center.

    All landmarks have high visibility (1.0).
    """
    kp = np.zeros((33, 4), dtype=np.float64)
    kp[:, 3] = 1.0  # all visible

    cx, cy = width // 2, height // 3

    # Shoulders
    kp[11] = [cx - 40, cy, 0, 1.0]       # left_shoulder
    kp[12] = [cx + 40, cy, 0, 1.0]       # right_shoulder
    # Elbows
    kp[13] = [cx - 70, cy + 50, 0, 1.0]  # left_elbow
    kp[14] = [cx + 70, cy + 50, 0, 1.0]  # right_elbow
    # Wrists
    kp[15] = [cx - 90, cy + 100, 0, 1.0] # left_wrist
    kp[16] = [cx + 90, cy + 100, 0, 1.0] # right_wrist
    # Hips
    kp[23] = [cx - 25, cy + 120, 0, 1.0] # left_hip
    kp[24] = [cx + 25, cy + 120, 0, 1.0] # right_hip
    # Knees
    kp[25] = [cx - 30, cy + 200, 0, 1.0] # left_knee
    kp[26] = [cx + 30, cy + 200, 0, 1.0] # right_knee
    # Ankles
    kp[27] = [cx - 30, cy + 280, 0, 1.0] # left_ankle
    kp[28] = [cx + 30, cy + 280, 0, 1.0] # right_ankle

    return kp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderSkeleton:
    """Tests for render_skeleton()."""

    def test_does_not_mutate_input(self) -> None:
        """Input frame should not be modified."""
        frame = _make_frame()
        original = frame.copy()
        kp = _make_visible_keypoints()

        render_skeleton(frame, kp, DEFAULT_CONFIG)
        np.testing.assert_array_equal(frame, original)

    def test_output_shape_matches_input(self) -> None:
        """Output frame should have the same shape as input."""
        frame = _make_frame(640, 480)
        kp = _make_visible_keypoints(640, 480)

        result = render_skeleton(frame, kp, DEFAULT_CONFIG)
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype

    def test_modifies_pixels(self) -> None:
        """Skeleton should change some pixels from the blank frame."""
        frame = _make_frame()
        kp = _make_visible_keypoints()

        result = render_skeleton(frame, kp, DEFAULT_CONFIG)
        assert not np.array_equal(result, frame)

    def test_low_confidence_joints_still_render(self) -> None:
        """Low-visibility joints should still render (faded, not missing)."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        # Set left arm joints to low visibility
        kp[13, 3] = 0.2  # left_elbow
        kp[15, 3] = 0.1  # left_wrist

        result = render_skeleton(frame, kp, DEFAULT_CONFIG)
        # Should still differ from blank frame
        assert not np.array_equal(result, frame)

    def test_all_zero_visibility(self) -> None:
        """All zero-visibility joints should still render (very faded)."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        kp[:, 3] = 0.0  # all invisible

        result = render_skeleton(frame, kp, DEFAULT_CONFIG)
        # Should still produce some output (alpha-blended)
        assert result.shape == frame.shape

    def test_custom_config_values(self) -> None:
        """Custom config values should not crash."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        config = {
            "thickness": 4,
            "joint_radius": 8,
            "low_confidence_alpha": 0.5,
            "color": [255, 0, 0],
        }
        result = render_skeleton(frame, kp, config)
        assert result.shape == frame.shape

    def test_region_colors_config(self) -> None:
        """Per-region colors from config should be used."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        config = {
            **DEFAULT_CONFIG,
            "region_colors": {
                "trunk": [255, 255, 255],
                "left_arm": [255, 0, 0],
                "right_arm": [0, 255, 0],
                "left_leg": [0, 0, 255],
                "right_leg": [128, 128, 0],
            },
        }
        result = render_skeleton(frame, kp, config)
        assert not np.array_equal(result, frame)

    def test_nan_coordinates_handled(self) -> None:
        """NaN in keypoint coordinates should not crash."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        kp[11, 0] = np.nan  # corrupt left_shoulder x
        kp[11, 1] = np.nan

        # Should not raise
        result = render_skeleton(frame, kp, DEFAULT_CONFIG)
        assert result.shape == frame.shape

    def test_connections_cover_major_joints(self) -> None:
        """SKELETON_CONNECTIONS should cover all major body segments."""
        all_indices = set()
        for connections in SKELETON_CONNECTIONS.values():
            for a, b in connections:
                all_indices.add(a)
                all_indices.add(b)

        # Should include shoulders, elbows, wrists, hips, knees, ankles
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            assert idx in all_indices

    def test_joint_landmarks_list(self) -> None:
        """JOINT_LANDMARKS should include all major joints."""
        for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            assert idx in JOINT_LANDMARKS


# ---------------------------------------------------------------------------
# Tests — occlusion-aware hiding
# ---------------------------------------------------------------------------


OCCLUSION_CONFIG: dict = {
    **DEFAULT_CONFIG,
    "hide_threshold": 0.25,
    "low_confidence_threshold": 0.5,
}


class TestOcclusionAwareHiding:
    """Tests for occlusion-aware joint hiding (hide_threshold)."""

    def test_hidden_joints_not_drawn(self) -> None:
        """Joints below hide_threshold produce less drawing than visible ones."""
        frame = _make_frame()

        # All visible
        kp_visible = _make_visible_keypoints()
        result_visible = render_skeleton(frame, kp_visible, OCCLUSION_CONFIG)

        # Left arm joints hidden (below 0.25)
        kp_hidden = _make_visible_keypoints()
        kp_hidden[13, 3] = 0.1  # left_elbow
        kp_hidden[15, 3] = 0.1  # left_wrist
        result_hidden = render_skeleton(frame, kp_hidden, OCCLUSION_CONFIG)

        # Hidden version should differ less from blank frame
        diff_visible = np.sum(np.abs(result_visible.astype(int) - frame.astype(int)))
        diff_hidden = np.sum(np.abs(result_hidden.astype(int) - frame.astype(int)))
        assert diff_hidden < diff_visible

    def test_all_hidden_draws_nothing(self) -> None:
        """All joints below hide_threshold should produce minimal change."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        kp[:, 3] = 0.1  # all below hide_threshold

        result = render_skeleton(frame, kp, OCCLUSION_CONFIG)
        # Alpha blending on empty overlays may cause tiny diffs; check near-identity
        max_diff = np.max(np.abs(result.astype(int) - frame.astype(int)))
        assert max_diff <= 1  # rounding only

    def test_hide_threshold_zero_draws_everything(self) -> None:
        """hide_threshold=0.0 should behave like the old code (draw all)."""
        frame = _make_frame()
        kp = _make_visible_keypoints()
        kp[:, 3] = 0.01  # very low but > 0

        config_no_hide = {**DEFAULT_CONFIG, "hide_threshold": 0.0}
        result = render_skeleton(frame, kp, config_no_hide)
        # Should still draw something even with very low visibility
        assert not np.array_equal(result, frame)

    def test_between_hide_and_low_is_faded(self) -> None:
        """Joints between hide_threshold and low_confidence_threshold are faded."""
        frame = _make_frame()

        # Full visibility
        kp_full = _make_visible_keypoints()
        result_full = render_skeleton(frame, kp_full, OCCLUSION_CONFIG)

        # Visibility at 0.35 (between hide=0.25 and low_conf=0.5)
        kp_faded = _make_visible_keypoints()
        kp_faded[:, 3] = 0.35
        result_faded = render_skeleton(frame, kp_faded, OCCLUSION_CONFIG)

        # Both should differ from blank, but faded should differ less
        diff_full = np.sum(np.abs(result_full.astype(int) - frame.astype(int)))
        diff_faded = np.sum(np.abs(result_faded.astype(int) - frame.astype(int)))
        assert diff_faded < diff_full
