"""Temporal smoothing and interpolation for keypoint trajectories.

Applies Savitzky-Golay filtering and linear interpolation to reduce
jitter and fill gaps in pose estimation sequences. All parameters
are driven by configs/default.yaml.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_keypoints(
    keypoints_sequence: list[np.ndarray | None],
    window: int = 7,
    polyorder: int = 2,
) -> list[np.ndarray | None]:
    """Apply Savitzky-Golay smoothing to a sequence of keypoint arrays.

    Smooths each joint coordinate (x, y, z) independently over time.
    The confidence column (index 3) is NOT smoothed — it stays raw.
    Frames where keypoints are None remain None in the output.

    Smoothing is applied per contiguous segment of valid (non-None)
    frames. Long gaps between valid segments are respected — values
    are not pulled across gaps.

    Args:
        keypoints_sequence: List of (33, 4) arrays or None per frame.
            Each array has columns [x, y, z, confidence].
        window: Savitzky-Golay window length (must be odd and > polyorder).
        polyorder: Polynomial order for the filter.

    Returns:
        Smoothed keypoint sequence with the same shape and None pattern.
    """
    if not keypoints_sequence:
        return []

    n_frames = len(keypoints_sequence)
    result: list[np.ndarray | None] = [
        kp.copy() if kp is not None else None for kp in keypoints_sequence
    ]

    # Find contiguous segments of valid frames
    segments = _find_contiguous_segments(keypoints_sequence)

    for seg_start, seg_end in segments:
        seg_length = seg_end - seg_start
        if seg_length < max(polyorder + 1, 3):
            continue  # too short to smooth

        # Compute effective window for this segment
        effective_window = _validated_window(window, polyorder, seg_length)
        if effective_window is None:
            continue  # cannot smooth this segment

        # Extract segment keypoints
        seg_keypoints = np.stack(
            [keypoints_sequence[i] for i in range(seg_start, seg_end)]
        )
        n_landmarks = seg_keypoints.shape[1]

        # Smooth x, y, z (columns 0-2) per landmark
        for landmark_idx in range(n_landmarks):
            for coord_idx in range(3):
                signal = seg_keypoints[:, landmark_idx, coord_idx]
                seg_keypoints[:, landmark_idx, coord_idx] = savgol_filter(
                    signal, effective_window, polyorder
                )

        # Write back
        for i, frame_idx in enumerate(range(seg_start, seg_end)):
            result[frame_idx] = seg_keypoints[i].copy()

    return result


def interpolate_gaps(
    keypoints_sequence: list[np.ndarray | None],
    max_gap: int = 5,
) -> list[np.ndarray | None]:
    """Fill short gaps in keypoint sequences using linear interpolation.

    For each run of consecutive None frames that is at most ``max_gap``
    frames long and bounded on both sides by valid keypoints, linearly
    interpolate all landmark coordinates. Confidence for interpolated
    frames is set to 0.0 (marking them as synthetic).

    Gaps longer than ``max_gap`` or at the edges (no bounding valid frame)
    are left as None.

    Args:
        keypoints_sequence: List of (33, 4) arrays or None per frame.
        max_gap: Maximum number of consecutive missing frames to fill.

    Returns:
        Keypoint sequence with short gaps filled by interpolation.
    """
    if not keypoints_sequence:
        return []

    n_frames = len(keypoints_sequence)
    result: list[np.ndarray | None] = list(keypoints_sequence)

    i = 0
    while i < n_frames:
        if result[i] is not None:
            i += 1
            continue

        # Found start of a gap — find its end
        gap_start = i
        while i < n_frames and result[i] is None:
            i += 1
        gap_end = i  # exclusive

        gap_length = gap_end - gap_start

        # Check if gap is fillable
        if gap_length > max_gap:
            continue
        if gap_start == 0 or gap_end >= n_frames:
            continue

        before = result[gap_start - 1]
        after = result[gap_end]
        if before is None or after is None:
            continue

        # Linear interpolation between before and after
        for j in range(gap_start, gap_end):
            alpha = (j - gap_start + 1) / (gap_length + 1)
            interpolated = before * (1 - alpha) + after * alpha
            # Set confidence column to 0.0 to mark as interpolated
            interpolated[:, 3] = 0.0
            result[j] = interpolated

    return result


def _find_contiguous_segments(
    sequence: list[np.ndarray | None],
) -> list[tuple[int, int]]:
    """Find contiguous runs of non-None frames.

    Args:
        sequence: List of arrays or None.

    Returns:
        List of (start, end) tuples where end is exclusive.
        Each segment contains only non-None entries.
    """
    segments: list[tuple[int, int]] = []
    n = len(sequence)
    i = 0
    while i < n:
        if sequence[i] is None:
            i += 1
            continue
        seg_start = i
        while i < n and sequence[i] is not None:
            i += 1
        segments.append((seg_start, i))
    return segments


def _validated_window(
    window: int,
    polyorder: int,
    segment_length: int,
) -> int | None:
    """Compute a valid Savitzky-Golay window for the given segment.

    Args:
        window: Requested window length.
        polyorder: Polynomial order.
        segment_length: Number of frames in the segment.

    Returns:
        A valid odd window length, or None if smoothing is not possible.
    """
    if polyorder < 0:
        return None

    # Window must be <= segment length
    w = min(window, segment_length)

    # Window must be > polyorder
    if w <= polyorder:
        w = polyorder + 1

    # Window must be odd
    if w % 2 == 0:
        w += 1

    # Final check: still fits in segment?
    if w > segment_length:
        return None

    # Sanity: window must be > polyorder (guaranteed above, but defensive)
    if w <= polyorder:
        return None

    return w
