"""Tests for src/core/pose_estimator.py.

Tests the PoseEstimator interface, landmark names, and context manager.
Uses real MediaPipe inference on a synthetic image — no video file needed.

Tests that require the model file are skipped if it's not available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.core.pose_estimator import MEDIAPIPE_LANDMARK_NAMES

# Skip all model-dependent tests if model is absent
_MODEL_PATH = "data/models/pose_landmarker_heavy.task"
_HAS_MODEL = Path(_MODEL_PATH).exists()

try:
    from src.core.pose_estimator import PoseEstimator
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False

requires_model = pytest.mark.skipif(
    not (_HAS_MODEL and _HAS_MEDIAPIPE),
    reason="Requires mediapipe and pose model file",
)


@pytest.fixture(scope="module")
def estimator():
    """Create a PoseEstimator with default-like config for all tests."""
    if not (_HAS_MODEL and _HAS_MEDIAPIPE):
        pytest.skip("Requires mediapipe and pose model file")
    config = {
        "model": "mediapipe",
        "min_confidence": 0.3,
        "smoothing": "savgol",
        "smoothing_window": 7,
        "smoothing_polyorder": 2,
        "max_interpolation_gap": 5,
    }
    est = PoseEstimator(config)
    yield est
    est.close()


def _make_blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a blank BGR frame (no person visible)."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests — landmark names (no model needed)
# ---------------------------------------------------------------------------


class TestLandmarkNamesStatic:
    """Tests for MEDIAPIPE_LANDMARK_NAMES constant (no model required)."""

    def test_has_33_names(self) -> None:
        assert len(MEDIAPIPE_LANDMARK_NAMES) == 33

    def test_all_strings(self) -> None:
        assert all(isinstance(n, str) for n in MEDIAPIPE_LANDMARK_NAMES)

    def test_contains_key_joints(self) -> None:
        for joint in [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]:
            assert joint in MEDIAPIPE_LANDMARK_NAMES


# ---------------------------------------------------------------------------
# Tests — landmark names (model required)
# ---------------------------------------------------------------------------


@requires_model
class TestLandmarkNames:
    """Tests for get_landmark_names()."""

    def test_returns_33_names(self, estimator) -> None:
        names = estimator.get_landmark_names()
        assert len(names) == 33

    def test_matches_module_constant(self, estimator) -> None:
        assert estimator.get_landmark_names() == MEDIAPIPE_LANDMARK_NAMES

    def test_returns_copy(self, estimator) -> None:
        names1 = estimator.get_landmark_names()
        names2 = estimator.get_landmark_names()
        assert names1 is not names2


# ---------------------------------------------------------------------------
# Tests — estimate_frame
# ---------------------------------------------------------------------------


@requires_model
class TestEstimateFrame:
    """Tests for estimate_frame()."""

    def test_blank_frame_returns_none_or_low_confidence(self, estimator) -> None:
        frame = _make_blank_frame()
        keypoints, confidence = estimator.estimate_frame(frame)
        if keypoints is not None:
            assert confidence < 0.5
        else:
            assert confidence == 0.0

    def test_returns_correct_shape_on_detection(self, estimator) -> None:
        frame = _make_stick_figure_frame()
        keypoints, confidence = estimator.estimate_frame(frame)
        if keypoints is not None:
            assert keypoints.shape == (33, 4)
            assert confidence > 0.0

    def test_confidence_is_float(self, estimator) -> None:
        frame = _make_blank_frame()
        _, confidence = estimator.estimate_frame(frame)
        assert isinstance(confidence, float)

    def test_pixel_coordinates_within_frame(self, estimator) -> None:
        frame = _make_stick_figure_frame(width=640, height=480)
        keypoints, _ = estimator.estimate_frame(frame)
        if keypoints is not None:
            assert np.all(keypoints[:, 0] >= -100)
            assert np.all(keypoints[:, 0] <= 740)
            assert np.all(keypoints[:, 1] >= -100)
            assert np.all(keypoints[:, 1] <= 580)


# ---------------------------------------------------------------------------
# Tests — estimate_video
# ---------------------------------------------------------------------------


@requires_model
class TestEstimateVideo:
    """Tests for estimate_video()."""

    def test_empty_input(self, estimator) -> None:
        keypoints, confidences = estimator.estimate_video([])
        assert keypoints == []
        assert confidences == []

    def test_output_length_matches_input(self, estimator) -> None:
        frames = [_make_blank_frame() for _ in range(3)]
        keypoints, confidences = estimator.estimate_video(frames)
        assert len(keypoints) == 3
        assert len(confidences) == 3

    def test_blank_frames_produce_none(self, estimator) -> None:
        frames = [_make_blank_frame() for _ in range(3)]
        keypoints, confidences = estimator.estimate_video(frames)
        for kp in keypoints:
            assert kp is None
        for conf in confidences:
            assert conf == 0.0

    def test_rejects_zero_fps(self, estimator) -> None:
        frames = [_make_blank_frame()]
        with pytest.raises(ValueError, match="fps must be positive"):
            estimator.estimate_video(frames, fps=0)

    def test_rejects_negative_fps(self, estimator) -> None:
        frames = [_make_blank_frame()]
        with pytest.raises(ValueError, match="fps must be positive"):
            estimator.estimate_video(frames, fps=-1.0)


# ---------------------------------------------------------------------------
# Tests — context manager
# ---------------------------------------------------------------------------


@requires_model
class TestContextManager:
    """Tests for PoseEstimator as a context manager."""

    def test_context_manager_usage(self) -> None:
        config = {
            "model": "mediapipe",
            "min_confidence": 0.5,
            "smoothing": "none",
        }
        with PoseEstimator(config) as est:
            names = est.get_landmark_names()
            assert len(names) == 33


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stick_figure_frame(
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Draw a crude stick figure on a white background."""
    import cv2

    frame = np.full((height, width, 3), 220, dtype=np.uint8)
    cx, cy = width // 2, height // 3

    cv2.circle(frame, (cx, cy), 20, (180, 140, 100), -1)
    cv2.line(frame, (cx, cy + 20), (cx, cy + 120), (180, 140, 100), 8)
    cv2.line(frame, (cx, cy + 40), (cx - 60, cy + 80), (180, 140, 100), 6)
    cv2.line(frame, (cx, cy + 40), (cx + 60, cy + 80), (180, 140, 100), 6)
    cv2.line(frame, (cx, cy + 120), (cx - 40, cy + 200), (180, 140, 100), 6)
    cv2.line(frame, (cx, cy + 120), (cx + 40, cy + 200), (180, 140, 100), 6)

    return frame
