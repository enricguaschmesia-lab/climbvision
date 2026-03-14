"""Tests for src/core/video_io.py.

Uses synthetic videos created programmatically with OpenCV — no external
sample files needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.video_io import (
    VideoMetadata,
    VideoWriter,
    _downscale_frame,
    frame_iterator,
    load_config,
    load_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_video(
    path: str | Path,
    num_frames: int = 10,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Create a small synthetic MP4 video for testing.

    Each frame is a solid color that changes across frames so we can verify
    that frame order is preserved.
    """
    path = Path(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        # Generate a frame with a unique color per index
        color = ((i * 25) % 256, (i * 50) % 256, (i * 75) % 256)
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Tests — load_video
# ---------------------------------------------------------------------------


class TestLoadVideo:
    """Tests for load_video()."""

    def test_returns_correct_metadata(self, tmp_path: Path) -> None:
        """Metadata should match the synthetic video's known properties."""
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", num_frames=15, fps=30.0, width=320, height=240
        )
        meta = load_video(video_path)

        assert isinstance(meta, VideoMetadata)
        assert meta.fps == pytest.approx(30.0, abs=1.0)
        assert meta.total_frames == 15
        assert meta.width == 320
        assert meta.height == 240
        assert meta.duration_s == pytest.approx(15 / 30.0, abs=0.1)
        assert meta.path == str(video_path)

    def test_raises_on_missing_file(self) -> None:
        """Should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_video("/nonexistent/path/video.mp4")

    def test_raises_on_invalid_file(self, tmp_path: Path) -> None:
        """Should raise ValueError for a file that isn't a valid video."""
        bad_file = tmp_path / "not_a_video.mp4"
        bad_file.write_text("this is not a video")
        with pytest.raises(ValueError):
            load_video(bad_file)


# ---------------------------------------------------------------------------
# Tests — frame_iterator
# ---------------------------------------------------------------------------


class TestFrameIterator:
    """Tests for frame_iterator()."""

    def test_yields_correct_number_of_frames(self, tmp_path: Path) -> None:
        """Should yield exactly as many frames as the video contains."""
        num_frames = 8
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", num_frames=num_frames
        )
        frames = list(frame_iterator(video_path))
        assert len(frames) == num_frames

    def test_frame_indices_are_sequential(self, tmp_path: Path) -> None:
        """Frame indices should start at 0 and increment by 1."""
        video_path = _create_synthetic_video(tmp_path / "test.mp4", num_frames=5)
        indices = [idx for idx, _, _ in frame_iterator(video_path)]
        assert indices == [0, 1, 2, 3, 4]

    def test_timestamps_increase(self, tmp_path: Path) -> None:
        """Timestamps should be monotonically increasing."""
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", num_frames=5, fps=30.0
        )
        timestamps = [ts for _, ts, _ in frame_iterator(video_path)]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_timestamps_match_fps(self, tmp_path: Path) -> None:
        """Timestamps should equal frame_idx / fps."""
        fps = 25.0
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", num_frames=5, fps=fps
        )
        for idx, ts, _ in frame_iterator(video_path):
            assert ts == pytest.approx(idx / fps, abs=1e-6)

    def test_frames_are_bgr_arrays(self, tmp_path: Path) -> None:
        """Each frame should be an (H, W, 3) uint8 numpy array."""
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", width=320, height=240
        )
        for _, _, frame in frame_iterator(video_path):
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (240, 320, 3)
            assert frame.dtype == np.uint8

    def test_downscale_large_frames(self, tmp_path: Path) -> None:
        """When max_dimension is set, frames should be downscaled."""
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", width=640, height=480
        )
        for _, _, frame in frame_iterator(video_path, max_dimension=320):
            h, w = frame.shape[:2]
            assert max(h, w) <= 320

    def test_no_downscale_when_already_small(self, tmp_path: Path) -> None:
        """Frames already within max_dimension should not be resized."""
        video_path = _create_synthetic_video(
            tmp_path / "test.mp4", width=320, height=240
        )
        for _, _, frame in frame_iterator(video_path, max_dimension=640):
            assert frame.shape == (240, 320, 3)

    def test_raises_on_missing_file(self) -> None:
        """Should raise FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError):
            list(frame_iterator("/nonexistent/path/video.mp4"))


# ---------------------------------------------------------------------------
# Tests — _downscale_frame
# ---------------------------------------------------------------------------


class TestDownscaleFrame:
    """Tests for _downscale_frame()."""

    def test_no_change_when_within_limit(self) -> None:
        """Frame that fits should be returned unchanged."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _downscale_frame(frame, max_dimension=200)
        assert result.shape == (100, 200, 3)

    def test_downscales_width(self) -> None:
        """Wide frame should be scaled down to fit max_dimension."""
        frame = np.zeros((100, 400, 3), dtype=np.uint8)
        result = _downscale_frame(frame, max_dimension=200)
        assert result.shape[1] == 200
        assert result.shape[0] == 50  # maintains aspect ratio

    def test_downscales_height(self) -> None:
        """Tall frame should be scaled down to fit max_dimension."""
        frame = np.zeros((600, 300, 3), dtype=np.uint8)
        result = _downscale_frame(frame, max_dimension=300)
        assert result.shape[0] == 300
        assert result.shape[1] == 150


# ---------------------------------------------------------------------------
# Tests — VideoWriter
# ---------------------------------------------------------------------------


class TestVideoWriter:
    """Tests for the VideoWriter class."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        """Should create a valid video file."""
        out_path = tmp_path / "out.mp4"
        with VideoWriter(str(out_path), fps=30.0, width=320, height=240) as w:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            w.write(frame)
            w.write(frame)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_frame_count_tracked(self, tmp_path: Path) -> None:
        """frame_count should match the number of write() calls."""
        out_path = tmp_path / "out.mp4"
        with VideoWriter(str(out_path), fps=30.0, width=320, height=240) as w:
            for _ in range(5):
                w.write(np.zeros((240, 320, 3), dtype=np.uint8))
            assert w.frame_count == 5

    def test_output_is_readable(self, tmp_path: Path) -> None:
        """Output video should be loadable and contain the correct frame count."""
        out_path = tmp_path / "out.mp4"
        num_frames = 7
        with VideoWriter(str(out_path), fps=30.0, width=320, height=240) as w:
            for _ in range(num_frames):
                w.write(np.zeros((240, 320, 3), dtype=np.uint8))

        meta = load_video(out_path)
        assert meta.total_frames == num_frames

    def test_resizes_mismatched_frames(self, tmp_path: Path) -> None:
        """Frames with wrong dimensions should be resized automatically."""
        out_path = tmp_path / "out.mp4"
        with VideoWriter(str(out_path), fps=30.0, width=320, height=240) as w:
            big_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            w.write(big_frame)  # should not raise
            assert w.frame_count == 1

    def test_context_manager_releases(self, tmp_path: Path) -> None:
        """After exiting the context manager, the file should be finalized."""
        out_path = tmp_path / "out.mp4"
        writer = VideoWriter(str(out_path), fps=30.0, width=320, height=240)
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.__exit__(None, None, None)
        # File should be readable after release
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Tests — load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for load_config()."""

    def test_loads_default_config(self) -> None:
        """Should load the project's default.yaml without error."""
        config = load_config("configs/default.yaml")
        assert isinstance(config, dict)
        assert "pose" in config
        assert "video" in config
        assert "overlay" in config

    def test_raises_on_missing_config(self) -> None:
        """Should raise FileNotFoundError for a non-existent config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
