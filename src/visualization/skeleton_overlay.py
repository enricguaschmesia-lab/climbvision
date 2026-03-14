"""Skeleton overlay rendering for climbing video analysis.

Draws pose skeleton lines and joint dots on video frames.
Body regions (arms, legs, trunk) use distinct colors for clarity.
Low-confidence joints are rendered with reduced opacity.

All visual parameters are driven by configs/default.yaml.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# MediaPipe Pose skeleton connections grouped by body region.
# Each connection is a pair of landmark indices.
# ---------------------------------------------------------------------------

SKELETON_CONNECTIONS: dict[str, list[tuple[int, int]]] = {
    "trunk": [
        (11, 12),   # left_shoulder – right_shoulder
        (11, 23),   # left_shoulder – left_hip
        (12, 24),   # right_shoulder – right_hip
        (23, 24),   # left_hip – right_hip
    ],
    "left_arm": [
        (11, 13),   # left_shoulder – left_elbow
        (13, 15),   # left_elbow – left_wrist
    ],
    "right_arm": [
        (12, 14),   # right_shoulder – right_elbow
        (14, 16),   # right_elbow – right_wrist
    ],
    "left_leg": [
        (23, 25),   # left_hip – left_knee
        (25, 27),   # left_knee – left_ankle
    ],
    "right_leg": [
        (24, 26),   # right_hip – right_knee
        (26, 28),   # right_knee – right_ankle
    ],
}

# Landmark indices that we draw dots on (major joints only, skip face)
JOINT_LANDMARKS: list[int] = [
    11, 12,           # shoulders
    13, 14,           # elbows
    15, 16,           # wrists
    23, 24,           # hips
    25, 26,           # knees
    27, 28,           # ankles
]

# Default region colors (BGR) — modern sport-palette.
# These are used if the config does not specify per-region colors.
DEFAULT_REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "trunk":     (200, 180, 60),    # muted cyan-blue
    "left_arm":  (80, 220, 100),    # green
    "right_arm": (80, 220, 100),    # green (same as left for symmetry)
    "left_leg":  (80, 160, 255),    # orange
    "right_leg": (80, 160, 255),    # orange (same as left for symmetry)
}


def render_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    config: dict,
) -> np.ndarray:
    """Draw skeleton lines and joint dots on a video frame.

    Renders body-region-colored limb connections and joint dots. Joints
    with low visibility are rendered with reduced opacity via alpha
    blending on a copy of the frame.

    Args:
        frame: BGR image as numpy array (H, W, 3). Not mutated.
        keypoints: (33, 4) array of [x_px, y_px, z, visibility].
        config: The ``overlay.skeleton`` section of the YAML config.
            Expected keys:
            - thickness: int, line thickness in pixels
            - joint_radius: int, dot radius in pixels
            - low_confidence_alpha: float, opacity for low-vis joints (0-1)
            - low_confidence_threshold: float, visibility below this is
              considered low-confidence (defaults to min_confidence from
              pose config or 0.5)
            - color: list[int], fallback BGR color if no region colors
            - region_colors: optional dict mapping region name to BGR list

    Returns:
        A new frame with the skeleton overlay drawn.
    """
    output = frame.copy()

    thickness = config.get("thickness", 2)
    joint_radius = config.get("joint_radius", 4)
    low_alpha = config.get("low_confidence_alpha", 0.3)
    low_threshold = config.get("low_confidence_threshold", 0.5)
    fallback_color = tuple(config.get("color", [0, 255, 128]))
    region_colors_cfg = config.get("region_colors", {})

    # Build region color map
    region_colors = {}
    for region in SKELETON_CONNECTIONS:
        if region in region_colors_cfg:
            region_colors[region] = tuple(region_colors_cfg[region])
        elif region in DEFAULT_REGION_COLORS:
            region_colors[region] = DEFAULT_REGION_COLORS[region]
        else:
            region_colors[region] = fallback_color

    # Pre-compute which landmarks are low-confidence
    vis = keypoints[:, 3]
    low_mask = vis < low_threshold

    # --- Draw limb connections ---
    # We draw low-confidence connections on a separate overlay for alpha blending
    overlay_low = output.copy()

    for region, connections in SKELETON_CONNECTIONS.items():
        color = region_colors[region]
        for idx_a, idx_b in connections:
            pt_a = _to_point(keypoints[idx_a])
            pt_b = _to_point(keypoints[idx_b])
            if pt_a is None or pt_b is None:
                continue

            either_low = low_mask[idx_a] or low_mask[idx_b]
            if either_low:
                cv2.line(overlay_low, pt_a, pt_b, color, thickness, cv2.LINE_AA)
            else:
                cv2.line(output, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    # Blend low-confidence connections
    cv2.addWeighted(overlay_low, low_alpha, output, 1 - low_alpha, 0, output)

    # --- Draw joint dots ---
    overlay_dots = output.copy()

    for idx in JOINT_LANDMARKS:
        pt = _to_point(keypoints[idx])
        if pt is None:
            continue

        # Joint dot color: white fill with a thin region-based outline
        if low_mask[idx]:
            cv2.circle(overlay_dots, pt, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(output, pt, joint_radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, pt, joint_radius, (80, 80, 80), 1, cv2.LINE_AA)

    # Blend low-confidence dots
    cv2.addWeighted(overlay_dots, low_alpha, output, 1 - low_alpha, 0, output)

    return output


def _to_point(landmark: np.ndarray) -> tuple[int, int] | None:
    """Convert a keypoint row [x, y, z, vis] to an integer (x, y) pixel tuple.

    Args:
        landmark: 1-D array of length 4: [x_px, y_px, z, visibility].

    Returns:
        (x, y) as integers, or None if coordinates are not finite.
    """
    x, y = landmark[0], landmark[1]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (int(round(x)), int(round(y)))
