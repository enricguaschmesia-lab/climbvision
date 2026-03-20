"""Tests for src/core/pipeline_runner.py."""

from __future__ import annotations

import numpy as np

import src.core.pipeline_runner as pipeline_runner_module
from src.core.pipeline_runner import (
    compute_adaptive_smoothing_window,
    run_pose_pass,
)


def _make_config() -> dict:
    """Create a minimal pipeline config for tests."""
    return {
        "pose": {
            "smoothing_window": None,
            "smoothing_window_seconds": 0.23,
            "smoothing_polyorder": 2,
            "max_interpolation_gap": 1,
        },
        "pipeline": {
            "chunk_size_frames": 5,
        },
    }


def _make_frame(value: int) -> np.ndarray:
    """Create a synthetic frame with a constant fill value."""
    return np.full((12, 12, 3), value, dtype=np.uint8)


class _FakeEstimator:
    """Fake estimator that records batch sizes and returns deterministic poses."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.detect_call_sizes: list[int] = []
        self.postprocess_call_sizes: list[int] = []
        self._next_marker = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def detect_video_raw(
        self,
        frames: list[np.ndarray],
        fps: float,
    ) -> tuple[list[np.ndarray | None], list[float]]:
        self.detect_call_sizes.append(len(frames))
        raw_keypoints: list[np.ndarray | None] = []
        confidences: list[float] = []

        for _frame in frames:
            keypoints = np.zeros((33, 4), dtype=np.float64)
            keypoints[0, 0] = float(self._next_marker)
            raw_keypoints.append(keypoints)
            confidences.append(0.5 + (self._next_marker / 100.0))
            self._next_marker += 1

        return raw_keypoints, confidences

    def postprocess_keypoints(
        self,
        raw_keypoints: list[np.ndarray | None],
        smoothing_window: int | None = None,
    ) -> list[np.ndarray | None]:
        self.postprocess_call_sizes.append(len(raw_keypoints))
        return [kp.copy() if kp is not None else None for kp in raw_keypoints]


class TestAdaptiveWindow:
    """Tests for adaptive smoothing window resolution."""

    def test_adaptive_window_30fps(self) -> None:
        assert compute_adaptive_smoothing_window(30.0, _make_config()) == 7

    def test_adaptive_window_60fps(self) -> None:
        assert compute_adaptive_smoothing_window(60.0, _make_config()) == 15

    def test_adaptive_window_120fps(self) -> None:
        assert compute_adaptive_smoothing_window(120.0, _make_config()) == 29

    def test_adaptive_window_manual_override(self) -> None:
        config = _make_config()
        config["pose"]["smoothing_window"] = 11
        assert compute_adaptive_smoothing_window(30.0, config) == 11

    def test_adaptive_window_minimum(self) -> None:
        assert compute_adaptive_smoothing_window(1.0, _make_config()) >= 3

    def test_adaptive_window_forces_odd(self) -> None:
        config = _make_config()
        config["pose"]["smoothing_window_seconds"] = 0.2
        assert compute_adaptive_smoothing_window(20.0, config) == 5


class TestRunPosePass:
    """Tests for chunked Pass 1 processing."""

    def test_run_pose_pass_short_video_single_chunk(self, monkeypatch) -> None:
        config = _make_config()
        estimator = _FakeEstimator()

        def fake_frame_iterator(path: str, max_dimension: int | None = None):
            for idx in range(3):
                yield idx, idx / 30.0, _make_frame(idx)

        monkeypatch.setattr(
            pipeline_runner_module,
            "frame_iterator",
            fake_frame_iterator,
        )
        monkeypatch.setattr(
            pipeline_runner_module,
            "compute_joint_angles",
            lambda keypoints: {"marker": float(keypoints[0, 0])},
        )

        result = run_pose_pass(
            video_path="fake.mp4",
            config=config,
            estimator=estimator,
            fps=30.0,
            smoothing_window=5,
            total_frames=3,
            max_dimension=None,
        )

        assert estimator.reset_calls == 1
        assert estimator.detect_call_sizes == [3]
        assert estimator.postprocess_call_sizes == [3]
        assert sorted(result) == [0, 1, 2]
        assert result[0].image is None
        assert result[1].joint_angles == {"marker": 1.0}

    def test_run_pose_pass_chunked_without_duplicate_detection(self, monkeypatch) -> None:
        config = _make_config()
        estimator = _FakeEstimator()

        def fake_frame_iterator(path: str, max_dimension: int | None = None):
            for idx in range(18):
                yield idx, idx / 30.0, _make_frame(idx)

        monkeypatch.setattr(
            pipeline_runner_module,
            "frame_iterator",
            fake_frame_iterator,
        )
        monkeypatch.setattr(
            pipeline_runner_module,
            "compute_joint_angles",
            lambda keypoints: {"marker": float(keypoints[0, 0])},
        )

        result = run_pose_pass(
            video_path="fake.mp4",
            config=config,
            estimator=estimator,
            fps=30.0,
            smoothing_window=5,
            total_frames=18,
            max_dimension=720,
        )

        assert estimator.reset_calls == 1
        assert estimator.detect_call_sizes == [8, 5, 5]
        assert sum(estimator.detect_call_sizes) == 18
        assert len(result) == 18
        assert sorted(result) == list(range(18))
        assert [int(result[idx].keypoints[0, 0]) for idx in range(18)] == list(range(18))
        assert result[17].joint_angles == {"marker": 17.0}

    def test_run_pose_pass_empty_video(self, monkeypatch) -> None:
        config = _make_config()
        estimator = _FakeEstimator()

        def fake_frame_iterator(path: str, max_dimension: int | None = None):
            return iter([])

        monkeypatch.setattr(
            pipeline_runner_module,
            "frame_iterator",
            fake_frame_iterator,
        )

        result = run_pose_pass(
            video_path="fake.mp4",
            config=config,
            estimator=estimator,
            fps=30.0,
            smoothing_window=5,
            total_frames=0,
            max_dimension=None,
        )

        assert result == {}
        assert estimator.detect_call_sizes == []
