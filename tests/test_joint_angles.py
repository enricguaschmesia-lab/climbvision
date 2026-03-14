"""Tests for src/analysis/joint_angles.py.

Uses synthetic keypoint arrays with known geometric configurations
to verify angle computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.joint_angles import JOINT_TRIPLETS, compute_joint_angles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keypoints_with_joint(
    idx_a: int,
    idx_b: int,
    idx_c: int,
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    point_c: tuple[float, float, float],
    visibility: float = 1.0,
) -> np.ndarray:
    """Create a (33, 4) keypoints array with specific points at three indices.

    All other landmarks are set to (0, 0, 0) with the given visibility.
    """
    kp = np.zeros((33, 4), dtype=np.float64)
    kp[:, 3] = visibility  # set all visibilities

    kp[idx_a, :3] = point_a
    kp[idx_b, :3] = point_b
    kp[idx_c, :3] = point_c

    return kp


# ---------------------------------------------------------------------------
# Tests — compute_joint_angles
# ---------------------------------------------------------------------------


class TestComputeJointAngles:
    """Tests for compute_joint_angles()."""

    def test_right_angle_elbow(self) -> None:
        """Left elbow at exactly 90 degrees."""
        # left_elbow triplet: (15=wrist, 13=elbow, 11=shoulder)
        # Place elbow at origin, wrist along +x, shoulder along +y → 90°
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
        )
        angles = compute_joint_angles(kp)
        assert "left_elbow" in angles
        assert angles["left_elbow"] == pytest.approx(90.0, abs=0.1)

    def test_straight_arm_elbow(self) -> None:
        """Left elbow at 180 degrees (straight arm)."""
        # wrist, elbow, shoulder all collinear
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(2, 0, 0),
            point_b=(1, 0, 0),
            point_c=(0, 0, 0),
        )
        angles = compute_joint_angles(kp)
        assert angles["left_elbow"] == pytest.approx(180.0, abs=0.1)

    def test_45_degree_angle(self) -> None:
        """Right elbow at 45 degrees."""
        # right_elbow triplet: (16=wrist, 14=elbow, 12=shoulder)
        kp = _make_keypoints_with_joint(
            idx_a=16, idx_b=14, idx_c=12,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(1, 1, 0),  # 45° from +x axis
        )
        angles = compute_joint_angles(kp)
        assert "right_elbow" in angles
        assert angles["right_elbow"] == pytest.approx(45.0, abs=0.1)

    def test_2d_shoulder_angle(self) -> None:
        """Verify shoulder angle computation in 2D (x, y)."""
        # left_shoulder triplet: (13=elbow, 11=shoulder, 23=hip)
        # Place shoulder at origin, elbow along +x, hip along -y → 90°
        kp = _make_keypoints_with_joint(
            idx_a=13, idx_b=11, idx_c=23,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
        )
        angles = compute_joint_angles(kp)
        assert "left_shoulder" in angles
        assert angles["left_shoulder"] == pytest.approx(90.0, abs=0.1)

    def test_returns_all_eight_joints(self) -> None:
        """With all landmarks visible, should return 8 joint angles."""
        # Create keypoints where each triplet forms a non-degenerate angle
        kp = np.zeros((33, 4), dtype=np.float64)
        kp[:, 3] = 1.0  # all visible

        # Place landmarks so each triplet has distinct, non-collinear points
        positions = {
            11: (0, 0, 0),    # left_shoulder
            12: (2, 0, 0),    # right_shoulder
            13: (-1, 1, 0),   # left_elbow
            14: (3, 1, 0),    # right_elbow
            15: (-2, 2, 0),   # left_wrist
            16: (4, 2, 0),    # right_wrist
            23: (0, -2, 0),   # left_hip
            24: (2, -2, 0),   # right_hip
            25: (0, -4, 0),   # left_knee
            26: (2, -4, 0),   # right_knee
            27: (0, -6, 0),   # left_ankle
            28: (2, -6, 0),   # right_ankle
        }
        for idx, pos in positions.items():
            kp[idx, :3] = pos

        angles = compute_joint_angles(kp)
        assert len(angles) == 8
        for name in JOINT_TRIPLETS:
            assert name in angles
            assert 0.0 <= angles[name] <= 180.0

    def test_low_visibility_excluded(self) -> None:
        """Joints with low-visibility landmarks should be excluded."""
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
            visibility=0.8,
        )
        # Set one landmark below threshold
        kp[15, 3] = 0.1

        angles = compute_joint_angles(kp, min_visibility=0.5)
        assert "left_elbow" not in angles

    def test_zero_visibility_included_by_default(self) -> None:
        """With default min_visibility=0.0, even zero-vis joints are included."""
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
            visibility=0.0,
        )
        angles = compute_joint_angles(kp)
        assert "left_elbow" in angles

    def test_knee_angle(self) -> None:
        """Left knee at 120 degrees."""
        # left_knee triplet: (23=hip, 25=knee, 27=ankle)
        # 120° angle at the knee
        angle_rad = np.radians(120)
        kp = _make_keypoints_with_joint(
            idx_a=23, idx_b=25, idx_c=27,
            point_a=(0, 1, 0),
            point_b=(0, 0, 0),
            point_c=(np.sin(angle_rad), np.cos(angle_rad), 0),
        )
        # The angle at B between BA and BC:
        # BA = (0,1,0), BC = (sin(120), cos(120), 0) = (0.866, -0.5, 0)
        # angle = arccos(dot / product of norms) = arccos(-0.5) = 120°
        angles = compute_joint_angles(kp)
        assert "left_knee" in angles
        assert angles["left_knee"] == pytest.approx(120.0, abs=0.1)

    def test_hip_angle(self) -> None:
        """Right hip at 90 degrees."""
        # right_hip triplet: (12=right_shoulder, 24=right_hip, 26=right_knee)
        kp = _make_keypoints_with_joint(
            idx_a=12, idx_b=24, idx_c=26,
            point_a=(0, 1, 0),
            point_b=(0, 0, 0),
            point_c=(1, 0, 0),
        )
        angles = compute_joint_angles(kp)
        assert "right_hip" in angles
        assert angles["right_hip"] == pytest.approx(90.0, abs=0.1)

    def test_values_are_degrees(self) -> None:
        """Angles should be in degrees, not radians."""
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(1, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
        )
        angles = compute_joint_angles(kp)
        # 90° in radians would be ~1.57 — make sure it's not that
        assert angles["left_elbow"] > 10

    def test_nan_coordinates_excluded(self) -> None:
        """Joints with NaN coordinates should be excluded, not produce NaN."""
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(float("nan"), 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
        )
        angles = compute_joint_angles(kp)
        assert "left_elbow" not in angles

    def test_inf_coordinates_excluded(self) -> None:
        """Joints with inf coordinates should be excluded."""
        kp = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(float("inf"), 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 1, 0),
        )
        angles = compute_joint_angles(kp)
        assert "left_elbow" not in angles

    def test_uses_2d_not_3d(self) -> None:
        """Angles should use only x, y — z should not affect the result.

        Regression: old code used 3D coords, but z is in a different scale
        (normalized depth vs pixel coordinates), producing wrong angles.
        """
        # Same x,y positions but different z values should give same angle
        kp_z0 = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(100, 0, 0),
            point_b=(0, 0, 0),
            point_c=(0, 100, 0),
        )
        kp_z100 = _make_keypoints_with_joint(
            idx_a=15, idx_b=13, idx_c=11,
            point_a=(100, 0, 999),
            point_b=(0, 0, -500),
            point_c=(0, 100, 42),
        )
        angles_z0 = compute_joint_angles(kp_z0)
        angles_z100 = compute_joint_angles(kp_z100)
        assert angles_z0["left_elbow"] == pytest.approx(
            angles_z100["left_elbow"], abs=0.01
        )
