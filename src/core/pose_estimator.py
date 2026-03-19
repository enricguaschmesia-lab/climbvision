"""Pose estimation wrapper for MediaPipe PoseLandmarker (Tasks API).

Provides a unified interface for extracting human pose keypoints from
video frames, with temporal smoothing and gap interpolation. All
parameters are driven by configs/default.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.smoothing import interpolate_gaps, savgol_smooth_keypoints

# Default model path relative to project root
_DEFAULT_MODEL_PATH = "data/models/pose_landmarker_heavy.task"

# MediaPipe Pose landmark names (33 landmarks)
MEDIAPIPE_LANDMARK_NAMES: list[str] = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def _import_mediapipe():
    """Lazily import mediapipe and return the module.

    Raises:
        ImportError: If mediapipe is not installed.
    """
    try:
        import mediapipe as mp
        return mp
    except ImportError as e:
        raise ImportError(
            "mediapipe is required for pose estimation. "
            "Install it with: pip install mediapipe>=0.10.9"
        ) from e


class PoseEstimator:
    """Unified pose estimation interface wrapping MediaPipe PoseLandmarker.

    Extracts 33 landmarks per frame as (x, y, z, visibility) arrays.
    Supports temporal smoothing (Savitzky-Golay) and gap interpolation
    for batch video processing.

    Uses the MediaPipe Tasks API (``PoseLandmarker``) with either IMAGE
    mode (single frame) or VIDEO mode (sequential frames with tracking).

    Attributes:
        config: Pose configuration dictionary from default.yaml.
        min_confidence: Minimum mean visibility to consider a frame valid.
    """

    def __init__(self, config: dict) -> None:
        """Initialize the pose estimator from config.

        Args:
            config: The ``pose`` section of the YAML config. Expected keys:
                - model: ``"mediapipe"`` (only supported model for Phase 1)
                - model_path: Path to the ``.task`` model file. Defaults to
                  ``data/models/pose_landmarker_heavy.task``.
                - min_confidence: float, minimum mean visibility threshold
                - smoothing: ``"savgol"`` | ``"none"``
                - smoothing_window: int, Savitzky-Golay window length
                - smoothing_polyorder: int, polynomial order
                - max_interpolation_gap: int, max consecutive None frames to fill
        """
        self.config = config
        self.min_confidence: float = config.get("min_confidence", 0.5)
        self._smoothing_type: str = config.get("smoothing", "savgol")
        self._smoothing_window: int = config.get("smoothing_window", 7)
        self._smoothing_polyorder: int = config.get("smoothing_polyorder", 2)
        self._max_gap: int = config.get("max_interpolation_gap", 5)
        self._frame_counter: int = 0

        model_name = config.get("model", "mediapipe")
        if model_name != "mediapipe":
            raise ValueError(
                "Only 'mediapipe' pose backend is supported in Phase 1. "
                "YOLOv8 planned for Phase 3."
            )

        self._model_path: str = config.get("model_path", _DEFAULT_MODEL_PATH)
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Pose model not found at {self._model_path}. "
                f"Run: python scripts/download_models.py"
            )

        # Deferred import — only fails here, not at module load
        self._mp = _import_mediapipe()
        _RunningMode = self._mp.tasks.vision.RunningMode

        # IMAGE mode landmarker — for single frame estimation
        self._image_landmarker = self._create_landmarker(_RunningMode.IMAGE)
        self._video_landmarker = self._create_landmarker(_RunningMode.VIDEO)

    def estimate_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Estimate pose keypoints for a single BGR frame.

        Uses IMAGE mode (no temporal tracking). For sequential video
        processing with tracking, use ``estimate_video()`` instead.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            A tuple of:
                - keypoints: (33, 4) array of [x_px, y_px, z, visibility],
                  where x_px and y_px are in pixel coordinates, z is relative
                  depth from MediaPipe, and visibility is 0-1. Returns None
                  if no pose is detected.
                - overall_confidence: Mean visibility across all landmarks.
                  Returns 0.0 if detection failed.
        """
        mp = self._mp
        frame_rgb = frame[:, :, ::-1].copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._image_landmarker.detect(mp_image)
        return self._parse_result(result, frame.shape[:2])

    def estimate_video(
        self,
        frames: list[np.ndarray],
        fps: float = 30.0,
    ) -> tuple[list[np.ndarray | None], list[float]]:
        """Process a sequence of frames with tracking, smoothing, and gap interpolation.

        Uses VIDEO mode for temporal tracking across frames, then applies
        gap interpolation and Savitzky-Golay smoothing. This method keeps
        VIDEO-mode state across calls: timestamps are derived from a
        persistent frame counter, and the landmarker retains its internal
        tracking state until ``reset()`` or ``close()`` is called.

        Args:
            frames: List of BGR images as numpy arrays.
            fps: Frames per second, used to compute timestamps for the
                VIDEO mode landmarker. Must be positive.

        Returns:
            A tuple of:
                - keypoints: List of (33, 4) keypoint arrays (or None for
                  unrecoverable frames), with smoothing and interpolation.
                - confidences: List of per-frame confidence floats (0.0 for
                  frames where detection failed or was interpolated).

        Raises:
            ValueError: If fps is not positive.
        """
        if not frames:
            return [], []

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        mp = self._mp

        # Step 1: Estimate each frame using VIDEO mode (has tracking)
        raw_keypoints: list[np.ndarray | None] = []
        confidences: list[float] = []
        for idx, frame in enumerate(frames):
            frame_rgb = frame[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((self._frame_counter + idx) * (1000.0 / fps))
            result = self._video_landmarker.detect_for_video(mp_image, timestamp_ms)
            kp, conf = self._parse_result(result, frame.shape[:2])
            raw_keypoints.append(kp)
            confidences.append(conf)

        self._frame_counter += len(frames)

        # Step 2: Interpolate short gaps
        interpolated = interpolate_gaps(raw_keypoints, max_gap=self._max_gap)

        # Step 3: Apply temporal smoothing
        if self._smoothing_type == "savgol":
            smoothed = savgol_smooth_keypoints(
                interpolated,
                window=self._smoothing_window,
                polyorder=self._smoothing_polyorder,
            )
        else:
            smoothed = interpolated

        return smoothed, confidences

    def get_landmark_names(self) -> list[str]:
        """Return ordered list of landmark names matching keypoint indices.

        Returns:
            List of 33 landmark name strings corresponding to MediaPipe
            Pose landmark indices.
        """
        return list(MEDIAPIPE_LANDMARK_NAMES)

    def reset(self) -> None:
        """Reset VIDEO-mode tracking state for processing a new video.

        Closes the current VIDEO-mode landmarker and creates a fresh one.
        Resets the internal frame counter to zero so timestamps restart.
        The IMAGE-mode landmarker is unaffected.
        """
        running_mode = self._mp.tasks.vision.RunningMode
        self._video_landmarker.close()
        self._video_landmarker = self._create_landmarker(running_mode.VIDEO)
        self._frame_counter = 0

    def close(self) -> None:
        """Release MediaPipe resources.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._image_landmarker is not None:
            self._image_landmarker.close()
            self._image_landmarker = None
        if self._video_landmarker is not None:
            self._video_landmarker.close()
            self._video_landmarker = None

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _parse_result(
        result,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray | None, float]:
        """Parse a PoseLandmarkerResult into a keypoints array.

        Args:
            result: ``PoseLandmarkerResult`` from MediaPipe.
            frame_shape: ``(height, width)`` of the input frame.

        Returns:
            A tuple of (keypoints array or None, overall confidence).
        """
        if not result.pose_landmarks:
            return None, 0.0

        h, w = frame_shape
        landmarks = result.pose_landmarks[0]  # first (only) detected pose
        keypoints = np.zeros((33, 4), dtype=np.float64)

        for i, lm in enumerate(landmarks):
            keypoints[i, 0] = lm.x * w       # x in pixel coordinates
            keypoints[i, 1] = lm.y * h       # y in pixel coordinates
            keypoints[i, 2] = lm.z           # relative depth
            keypoints[i, 3] = lm.visibility  # 0-1 confidence

        overall_confidence = float(keypoints[:, 3].mean())
        return keypoints, overall_confidence

    def _create_landmarker(self, running_mode: object) -> object:
        """Create a MediaPipe pose landmarker for the given running mode.

        Args:
            running_mode: A ``mediapipe.tasks.vision.RunningMode`` enum value
                (IMAGE or VIDEO).

        Returns:
            A configured ``PoseLandmarker`` instance.
        """
        _BaseOptions = self._mp.tasks.BaseOptions
        _PoseLandmarker = self._mp.tasks.vision.PoseLandmarker
        _PoseLandmarkerOptions = self._mp.tasks.vision.PoseLandmarkerOptions
        return _PoseLandmarker.create_from_options(
            _PoseLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=self._model_path),
                running_mode=running_mode,
                num_poses=1,
                min_pose_detection_confidence=self.min_confidence,
                min_pose_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
        )
