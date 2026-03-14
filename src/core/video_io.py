"""Video loading, frame extraction, and writing utilities.

Provides functions to load video metadata, iterate over frames, and write
annotated output videos. All codec/fps settings are driven by configs/default.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import yaml


@dataclass
class VideoMetadata:
    """Metadata extracted from a video file.

    Attributes:
        path: Absolute path to the video file.
        fps: Frames per second.
        total_frames: Total number of frames.
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration_s: Video duration in seconds.
    """
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float


def load_video(path: str | Path) -> VideoMetadata:
    """Load video file and return its metadata.

    Args:
        path: Path to the video file (.mp4, .avi, etc.).

    Returns:
        VideoMetadata with fps, frame count, dimensions, and duration.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video file cannot be opened by OpenCV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) in video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps

        return VideoMetadata(
            path=str(path),
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_s=duration_s,
        )
    finally:
        cap.release()


def frame_iterator(
    path: str | Path,
    max_dimension: int | None = None,
) -> Generator[tuple[int, float, np.ndarray], None, None]:
    """Iterate over video frames, yielding (frame_idx, timestamp, BGR image).

    Args:
        path: Path to the video file.
        max_dimension: If set, downscale frames so the largest dimension
            does not exceed this value. Preserves aspect ratio.

    Yields:
        Tuple of (frame_idx, timestamp_seconds, bgr_frame) for each frame.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video file cannot be opened by OpenCV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Invalid FPS ({fps}) in video: {path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        consecutive_failures = 0
        max_consecutive_failures = 10

        while frame_idx < total_frames or total_frames <= 0:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    break  # genuine EOF or unrecoverable
                frame_idx += 1
                continue

            consecutive_failures = 0

            if max_dimension is not None:
                frame = _downscale_frame(frame, max_dimension)

            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame
            frame_idx += 1
    finally:
        cap.release()


def _downscale_frame(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscale a frame so its largest dimension does not exceed max_dimension.

    Args:
        frame: BGR image as numpy array.
        max_dimension: Maximum allowed size for either width or height.

    Returns:
        Downscaled frame if needed, otherwise the original frame unchanged.
    """
    h, w = frame.shape[:2]
    largest = max(h, w)
    if largest <= max_dimension:
        return frame

    scale = max_dimension / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class VideoWriter:
    """Wrapper around cv2.VideoWriter with config-driven codec and fps.

    Usage::

        writer = VideoWriter("output.mp4", fps=30.0, width=1920, height=1080)
        writer.write(frame)
        writer.release()

    Or as a context manager::

        with VideoWriter("output.mp4", fps=30.0, width=1920, height=1080) as w:
            w.write(frame)

    Attributes:
        path: Output file path.
        fps: Output frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        codec: FourCC codec string.
        frame_count: Number of frames written so far.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        codec: str = "mp4v",
    ) -> None:
        """Initialize the video writer.

        Args:
            path: Output video file path.
            fps: Frames per second for the output video.
            width: Frame width in pixels.
            height: Frame height in pixels.
            codec: FourCC codec string. Defaults to "mp4v".
        """
        self.path = str(Path(path))
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.frame_count = 0

        # Ensure parent directory exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            raise ValueError(
                f"Cannot create video writer at {self.path} "
                f"(codec={codec}, fps={fps}, {width}x{height})"
            )

    def write(self, frame: np.ndarray) -> None:
        """Write a single frame to the output video.

        Args:
            frame: BGR image. Will be resized to (width, height) if dimensions
                don't match.
        """
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        self._writer.write(frame)
        self.frame_count += 1

    def release(self) -> None:
        """Finalize and close the video file."""
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def load_config(config_path: str | Path = "configs/default.yaml") -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
