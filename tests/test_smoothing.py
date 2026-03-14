"""Tests for src/utils/smoothing.py.

Uses synthetic keypoint trajectories — no real video or model needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.smoothing import interpolate_gaps, savgol_smooth_keypoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_keypoints(n_landmarks: int = 33, value: float = 1.0) -> np.ndarray:
    """Create a (n_landmarks, 4) keypoint array filled with a constant."""
    kp = np.full((n_landmarks, 4), value, dtype=np.float64)
    return kp


def _make_linear_trajectory(
    n_frames: int = 20,
    n_landmarks: int = 33,
) -> list[np.ndarray]:
    """Create a sequence where each landmark moves linearly over time.

    x = t, y = 2*t, z = 0, confidence = 1.0  for frame t.
    """
    sequence: list[np.ndarray] = []
    for t in range(n_frames):
        kp = np.zeros((n_landmarks, 4), dtype=np.float64)
        kp[:, 0] = t          # x
        kp[:, 1] = 2.0 * t    # y
        kp[:, 2] = 0.0        # z
        kp[:, 3] = 1.0        # confidence
        sequence.append(kp)
    return sequence


def _add_noise(
    sequence: list[np.ndarray],
    std: float = 0.5,
    seed: int = 42,
) -> list[np.ndarray]:
    """Add Gaussian noise to x, y, z of a keypoint sequence."""
    rng = np.random.default_rng(seed)
    noisy = []
    for kp in sequence:
        noisy_kp = kp.copy()
        noisy_kp[:, :3] += rng.normal(0, std, size=(kp.shape[0], 3))
        noisy.append(noisy_kp)
    return noisy


# ---------------------------------------------------------------------------
# Tests — savgol_smooth_keypoints
# ---------------------------------------------------------------------------


class TestSavgolSmooth:
    """Tests for savgol_smooth_keypoints()."""

    def test_empty_sequence(self) -> None:
        """Empty input should return empty output."""
        assert savgol_smooth_keypoints([]) == []

    def test_preserves_none_pattern(self) -> None:
        """None frames in input should remain None in output."""
        seq: list[np.ndarray | None] = [
            _make_keypoints(value=1.0),
            None,
            _make_keypoints(value=3.0),
            _make_keypoints(value=4.0),
            None,
            _make_keypoints(value=6.0),
            _make_keypoints(value=7.0),
            _make_keypoints(value=8.0),
        ]
        result = savgol_smooth_keypoints(seq, window=3, polyorder=1)
        assert result[1] is None
        assert result[4] is None
        # Valid frames should be non-None
        for i in [0, 2, 3, 5, 6, 7]:
            assert result[i] is not None

    def test_output_shape_preserved(self) -> None:
        """Output arrays should have the same shape as input."""
        seq = _make_linear_trajectory(n_frames=15)
        result = savgol_smooth_keypoints(seq, window=5, polyorder=2)
        for kp in result:
            assert kp is not None
            assert kp.shape == (33, 4)

    def test_confidence_not_smoothed(self) -> None:
        """The confidence column (index 3) should remain unchanged."""
        seq = _make_linear_trajectory(n_frames=15)
        # Set varying confidence values
        for i, kp in enumerate(seq):
            kp[:, 3] = 0.5 + 0.03 * i

        result = savgol_smooth_keypoints(seq, window=5, polyorder=2)
        for i in range(len(seq)):
            np.testing.assert_array_almost_equal(
                result[i][:, 3], seq[i][:, 3]
            )

    def test_reduces_noise(self) -> None:
        """Smoothed trajectory should be closer to the true trajectory than noisy input."""
        true_seq = _make_linear_trajectory(n_frames=30)
        noisy_seq = _add_noise(true_seq, std=1.0)
        smoothed_seq = savgol_smooth_keypoints(noisy_seq, window=7, polyorder=2)

        # Compare RMS error for x coordinate, landmark 0
        true_x = np.array([kp[0, 0] for kp in true_seq])
        noisy_x = np.array([kp[0, 0] for kp in noisy_seq])
        smoothed_x = np.array([kp[0, 0] for kp in smoothed_seq])

        noisy_rms = np.sqrt(np.mean((noisy_x - true_x) ** 2))
        smoothed_rms = np.sqrt(np.mean((smoothed_x - true_x) ** 2))

        assert smoothed_rms < noisy_rms

    def test_too_few_frames_returns_as_is(self) -> None:
        """With fewer valid frames than the window, return unmodified."""
        seq: list[np.ndarray | None] = [_make_keypoints(value=1.0), _make_keypoints(value=2.0)]
        result = savgol_smooth_keypoints(seq, window=7, polyorder=2)
        for i in range(len(seq)):
            np.testing.assert_array_equal(result[i], seq[i])

    def test_single_frame(self) -> None:
        """A single frame should pass through unchanged."""
        seq = [_make_keypoints(value=5.0)]
        result = savgol_smooth_keypoints(seq, window=7, polyorder=2)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], seq[0])

    def test_all_none_returns_all_none(self) -> None:
        """All-None input should produce all-None output."""
        seq: list[np.ndarray | None] = [None, None, None]
        result = savgol_smooth_keypoints(seq, window=3, polyorder=1)
        assert all(r is None for r in result)

    def test_does_not_smooth_across_long_gap(self) -> None:
        """Values separated by a long None gap must not be pulled together.

        Regression: old implementation compacted all valid frames into one
        sequence, smoothing across unfilled gaps.
        """
        # Segment A: frames 0-9, x = 0.0 (constant)
        # Gap: frames 10-29 (all None, > max_gap so unfilled)
        # Segment B: frames 30-39, x = 100.0 (constant)
        n_frames = 40
        seq: list[np.ndarray | None] = [None] * n_frames
        for i in range(10):
            kp = np.zeros((33, 4), dtype=np.float64)
            kp[:, 0] = 0.0
            kp[:, 3] = 1.0
            seq[i] = kp
        for i in range(30, 40):
            kp = np.zeros((33, 4), dtype=np.float64)
            kp[:, 0] = 100.0
            kp[:, 3] = 1.0
            seq[i] = kp

        result = savgol_smooth_keypoints(seq, window=7, polyorder=2)

        # Segment A values should stay near 0
        for i in range(10):
            assert result[i] is not None
            assert abs(result[i][0, 0]) < 5.0, f"Frame {i}: x={result[i][0, 0]}"

        # Segment B values should stay near 100
        for i in range(30, 40):
            assert result[i] is not None
            assert abs(result[i][0, 0] - 100.0) < 5.0, f"Frame {i}: x={result[i][0, 0]}"

    def test_smooths_each_segment_independently(self) -> None:
        """Two separate valid segments should each be smoothed independently."""
        # Segment 1: frames 0-4, linear x = t
        # None at frame 5
        # Segment 2: frames 6-10, linear x = 50 + t
        seq: list[np.ndarray | None] = [None] * 11
        for i in range(5):
            kp = np.zeros((33, 4), dtype=np.float64)
            kp[:, 0] = float(i)
            kp[:, 3] = 1.0
            seq[i] = kp
        for i in range(6, 11):
            kp = np.zeros((33, 4), dtype=np.float64)
            kp[:, 0] = 50.0 + float(i)
            kp[:, 3] = 1.0
            seq[i] = kp

        result = savgol_smooth_keypoints(seq, window=3, polyorder=1)

        # Gap should remain None
        assert result[5] is None

        # Segment 1 should stay near original values
        for i in range(5):
            assert result[i] is not None
            assert abs(result[i][0, 0] - float(i)) < 1.0

        # Segment 2 should stay near original values
        for i in range(6, 11):
            assert result[i] is not None
            assert abs(result[i][0, 0] - (50.0 + float(i))) < 1.0


# ---------------------------------------------------------------------------
# Tests — interpolate_gaps
# ---------------------------------------------------------------------------


class TestInterpolateGaps:
    """Tests for interpolate_gaps()."""

    def test_empty_sequence(self) -> None:
        """Empty input should return empty output."""
        assert interpolate_gaps([]) == []

    def test_no_gaps(self) -> None:
        """Sequence with no None frames should be unchanged."""
        seq = _make_linear_trajectory(n_frames=5)
        result = interpolate_gaps(seq, max_gap=5)
        for i in range(5):
            np.testing.assert_array_equal(result[i], seq[i])

    def test_fills_single_frame_gap(self) -> None:
        """A one-frame gap should be interpolated between neighbors."""
        seq = _make_linear_trajectory(n_frames=5)
        seq[2] = None  # gap at frame 2, between frames 1 and 3

        result = interpolate_gaps(seq, max_gap=5)
        assert result[2] is not None

        # Interpolated x should be midpoint of frame 1 and frame 3
        expected_x = (1.0 + 3.0) / 2.0  # frames 1 and 3 have x=1 and x=3
        assert result[2][0, 0] == pytest.approx(expected_x, abs=0.01)

    def test_fills_multi_frame_gap(self) -> None:
        """A multi-frame gap within max_gap should be filled."""
        seq = _make_linear_trajectory(n_frames=10)
        seq[3] = None
        seq[4] = None
        seq[5] = None  # 3-frame gap between frames 2 and 6

        result = interpolate_gaps(seq, max_gap=5)
        for i in [3, 4, 5]:
            assert result[i] is not None

        # Check that x values are linearly interpolated
        # Before: x=2 (frame 2), After: x=6 (frame 6)
        assert result[3][0, 0] == pytest.approx(3.0, abs=0.01)
        assert result[4][0, 0] == pytest.approx(4.0, abs=0.01)
        assert result[5][0, 0] == pytest.approx(5.0, abs=0.01)

    def test_does_not_fill_long_gap(self) -> None:
        """Gaps longer than max_gap should remain None."""
        seq = _make_linear_trajectory(n_frames=10)
        for i in range(2, 8):
            seq[i] = None  # 6-frame gap

        result = interpolate_gaps(seq, max_gap=5)
        for i in range(2, 8):
            assert result[i] is None

    def test_does_not_fill_edge_gaps(self) -> None:
        """Gaps at the start or end (no bounding frame) stay None."""
        seq = _make_linear_trajectory(n_frames=5)
        seq[0] = None  # gap at start
        seq[4] = None  # gap at end

        result = interpolate_gaps(seq, max_gap=5)
        assert result[0] is None
        assert result[4] is None

    def test_interpolated_confidence_is_zero(self) -> None:
        """Confidence should be 0.0 for interpolated frames (marking them synthetic)."""
        seq = _make_linear_trajectory(n_frames=5)
        seq[2] = None

        result = interpolate_gaps(seq, max_gap=5)
        assert result[2] is not None
        assert np.all(result[2][:, 3] == 0.0)

    def test_multiple_separate_gaps(self) -> None:
        """Multiple independent short gaps should all be filled."""
        seq = _make_linear_trajectory(n_frames=10)
        seq[2] = None  # gap 1
        seq[6] = None  # gap 2

        result = interpolate_gaps(seq, max_gap=5)
        assert result[2] is not None
        assert result[6] is not None

    def test_original_frames_untouched(self) -> None:
        """Valid frames should not be modified."""
        seq = _make_linear_trajectory(n_frames=5)
        seq[2] = None
        original_1 = seq[1].copy()
        original_3 = seq[3].copy()

        result = interpolate_gaps(seq, max_gap=5)
        np.testing.assert_array_equal(result[1], original_1)
        np.testing.assert_array_equal(result[3], original_3)
