"""Chunked pose-processing helpers for the ClimbVision pipeline.

This module keeps the scope intentionally narrow: adaptive smoothing
window resolution and Pass 1 pose extraction over chunked frame reads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.joint_angles import compute_joint_angles
from src.core.data_structures import FrameData
from src.core.pose_estimator import PoseEstimator
from src.core.video_io import frame_iterator

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback only used when tqdm missing
    tqdm = None


@dataclass
class _BufferedPoseFrame:
    """Intermediate pose data kept in the rolling chunk buffer."""

    frame_idx: int
    timestamp: float
    raw_keypoints: np.ndarray | None
    confidence: float


class _NullProgress:
    """Fallback progress bar used when tqdm is unavailable."""

    def update(self, _: int = 1) -> None:
        """Ignore progress updates."""

    def close(self) -> None:
        """No-op close method for API parity."""


def compute_adaptive_smoothing_window(fps: float, config: dict) -> int:
    """Compute smoothing window from FPS or use manual override.

    Args:
        fps: Video frames per second.
        config: Full application config dictionary containing the ``pose``
            section.

    Returns:
        Resolved odd Savitzky-Golay smoothing window.

    Raises:
        ValueError: If ``fps`` is not positive.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    pose_config = config["pose"]
    manual_override = pose_config.get("smoothing_window")
    if manual_override is not None:
        return int(manual_override)

    requested = int(round(pose_config["smoothing_window_seconds"] * fps))
    min_valid = max(3, int(pose_config["smoothing_polyorder"]) + 1)
    if min_valid % 2 == 0:
        min_valid += 1

    resolved = max(requested, min_valid)
    if resolved % 2 == 0:
        resolved += 1
    return resolved


def run_pose_pass(
    video_path: str,
    config: dict,
    estimator: PoseEstimator,
    fps: float,
    smoothing_window: int,
    total_frames: int,
    max_dimension: int | None,
) -> dict[int, FrameData]:
    """Run Pass 1 chunked pose extraction with overlap-aware smoothing.

    The pipeline reads frames sequentially, performs MediaPipe VIDEO-mode
    inference exactly once per decoded frame, then reuses the raw detections
    inside a rolling overlap buffer for interpolation and temporal smoothing.

    Args:
        video_path: Path to the source video.
        config: Full application config dictionary.
        estimator: Initialized pose estimator instance.
        fps: Video frames per second.
        smoothing_window: Resolved odd smoothing window to use for Pass 1.
        total_frames: OpenCV-reported frame count. Treated as advisory only.
        max_dimension: Optional max frame dimension for decoding.

    Returns:
        Dictionary mapping actual frame indices to ``FrameData`` records.
        Images are not retained during Pass 1, so ``FrameData.image`` is
        always ``None`` in this output.
    """
    pipeline_config = config.get("pipeline", {})
    pose_config = config["pose"]
    chunk_size = int(pipeline_config.get("chunk_size_frames", 300))
    if chunk_size <= 0:
        raise ValueError(
            f"pipeline.chunk_size_frames must be positive, got {chunk_size}"
        )

    overlap = int(pose_config.get("max_interpolation_gap", 0)) + smoothing_window // 2
    decoded_frames: dict[int, FrameData] = {}
    buffer: list[_BufferedPoseFrame] = []
    left_context = 0

    estimator.reset()
    iterator = frame_iterator(video_path, max_dimension=max_dimension)
    progress = _make_progress(total_frames, "Pass 1")

    try:
        reached_eof = _read_new_frames(
            iterator=iterator,
            estimator=estimator,
            fps=fps,
            target_count=chunk_size + overlap,
            buffer=buffer,
            progress=progress,
        )
        if not buffer:
            return decoded_frames

        if reached_eof:
            processed = estimator.postprocess_keypoints(
                [record.raw_keypoints for record in buffer],
                smoothing_window=smoothing_window,
            )
            _commit_frames(buffer, processed, 0, len(buffer), decoded_frames)
            return decoded_frames

        processed = estimator.postprocess_keypoints(
            [record.raw_keypoints for record in buffer],
            smoothing_window=smoothing_window,
        )
        commit_end = min(chunk_size, max(0, len(buffer) - overlap))
        _commit_frames(buffer, processed, 0, commit_end, decoded_frames)

        trim_start = max(0, commit_end - overlap)
        buffer = buffer[trim_start:]
        left_context = commit_end - trim_start

        while buffer:
            reached_eof = _read_new_frames(
                iterator=iterator,
                estimator=estimator,
                fps=fps,
                target_count=chunk_size,
                buffer=buffer,
                progress=progress,
            )

            processed = estimator.postprocess_keypoints(
                [record.raw_keypoints for record in buffer],
                smoothing_window=smoothing_window,
            )

            if reached_eof:
                _commit_frames(
                    buffer,
                    processed,
                    left_context,
                    len(buffer),
                    decoded_frames,
                )
                break

            commit_start = left_context
            commit_end = min(commit_start + chunk_size, max(commit_start, len(buffer) - overlap))
            if commit_end <= commit_start:
                continue

            _commit_frames(
                buffer,
                processed,
                commit_start,
                commit_end,
                decoded_frames,
            )

            trim_start = max(0, commit_end - overlap)
            buffer = buffer[trim_start:]
            left_context = commit_end - trim_start

        return decoded_frames
    finally:
        progress.close()


def _make_progress(total_frames: int, desc: str):
    """Create a tqdm progress bar or a lightweight fallback."""
    if tqdm is None:
        return _NullProgress()
    total = total_frames if total_frames > 0 else None
    return tqdm(total=total, desc=desc, unit="frame")


def _read_new_frames(
    iterator,
    estimator: PoseEstimator,
    fps: float,
    target_count: int,
    buffer: list[_BufferedPoseFrame],
    progress,
) -> bool:
    """Read and infer a batch of new frames, returning whether EOF was hit."""
    frames: list[np.ndarray] = []
    metadata: list[tuple[int, float]] = []

    while len(frames) < target_count:
        try:
            frame_idx, timestamp, frame = next(iterator)
        except StopIteration:
            break
        metadata.append((frame_idx, timestamp))
        frames.append(frame)

    if not frames:
        return True

    try:
        raw_keypoints, confidences = estimator.detect_video_raw(frames, fps)
    except Exception:
        raw_keypoints = [None] * len(frames)
        confidences = [0.0] * len(frames)

    if len(raw_keypoints) != len(frames):
        raw_keypoints = _resize_sequence(raw_keypoints, len(frames), None)
    if len(confidences) != len(frames):
        confidences = _resize_sequence(confidences, len(frames), 0.0)

    for (frame_idx, timestamp), raw_keypoint, confidence in zip(
        metadata,
        raw_keypoints,
        confidences,
    ):
        buffer.append(
            _BufferedPoseFrame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                raw_keypoints=raw_keypoint,
                confidence=float(confidence),
            )
        )

    progress.update(len(frames))
    return len(frames) < target_count


def _resize_sequence(values: list, size: int, fill_value):
    """Pad or trim a sequence to an expected size."""
    resized = list(values[:size])
    while len(resized) < size:
        resized.append(fill_value)
    return resized


def _commit_frames(
    buffer: list[_BufferedPoseFrame],
    processed_keypoints: list[np.ndarray | None],
    start: int,
    end: int,
    output: dict[int, FrameData],
) -> None:
    """Commit a processed buffer slice into the output frame map."""
    for record, keypoints in zip(buffer[start:end], processed_keypoints[start:end]):
        joint_angles = None
        if keypoints is not None:
            try:
                joint_angles = compute_joint_angles(keypoints)
            except Exception:
                joint_angles = None

        output[record.frame_idx] = FrameData(
            frame_idx=record.frame_idx,
            timestamp=record.timestamp,
            image=None,
            keypoints=keypoints,
            keypoint_confidence=record.confidence,
            holds=None,
            route_holds=None,
            joint_angles=joint_angles,
        )
