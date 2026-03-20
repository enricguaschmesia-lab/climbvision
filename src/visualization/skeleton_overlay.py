"""Skeleton overlay rendering for climbing video analysis.

Draws pose skeleton lines and joint dots on video frames.
Body regions (arms, legs, trunk) use distinct colors for clarity.
Low-confidence joints are rendered with reduced opacity, and joints
below the occlusion hide threshold are not drawn at all.

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


def compute_stress_colors(
    joint_angles: dict[str, float],
    joint_triplets: dict[str, tuple[int, int, int]],
    angles_config: dict,
) -> dict[int, tuple[int, int, int]]:
    """Map joint angles to stress-level colors for skeleton rendering.

    Args:
        joint_angles: Joint name to angle in degrees.
        joint_triplets: Joint name to (endpoint_a, center, endpoint_c) indices.
        angles_config: The ``overlay.angles`` config section with range/color keys.

    Returns:
        Dictionary mapping landmark center index to a BGR color tuple.
    """
    efficient_range = angles_config.get("efficient_range", [150, 180])
    warning_range = angles_config.get("warning_range", [90, 150])
    efficient_color = tuple(angles_config.get("efficient_color", [60, 220, 60]))
    warning_color = tuple(angles_config.get("warning_color", [0, 180, 255]))
    error_color = tuple(angles_config.get("error_color", [20, 20, 240]))

    stress_colors: dict[int, tuple[int, int, int]] = {}
    for joint_name, angle in joint_angles.items():
        if joint_name in joint_triplets:
            idx_center = joint_triplets[joint_name][1]
            if angle < warning_range[0]:
                stress_colors[idx_center] = error_color
            elif angle < efficient_range[0]:
                stress_colors[idx_center] = warning_color
            else:
                stress_colors[idx_center] = efficient_color
    return stress_colors


def render_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    config: dict,
    stress_colors: dict[int, tuple[int, int, int]] | None = None,
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
              considered low-confidence (defaults to 0.5)
            - hide_threshold: float, visibility below this means the joint
              is NOT drawn at all (occlusion-aware). Defaults to 0.0
              (draw everything).
            - color: list[int], fallback BGR color if no region colors
            - region_colors: optional dict mapping region name to BGR list
        stress_colors: optional dict mapping landmark index to a BGR color tuple
            for rendering stressed joints.

    Returns:
        A new frame with the skeleton overlay drawn.
    """
    if stress_colors is None:
        stress_colors = {}
        
    output = frame.copy()

    thickness = config.get("thickness", 2)
    joint_radius = config.get("joint_radius", 4)
    low_alpha = config.get("low_confidence_alpha", 0.3)
    low_threshold = config.get("low_confidence_threshold", 0.5)
    hide_threshold = config.get("hide_threshold", 0.0)
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

    # Pre-compute visibility masks
    vis = keypoints[:, 3]
    hidden_mask = vis < hide_threshold     # not drawn at all
    low_mask = vis < low_threshold         # drawn faded (alpha blended)

    # --- Draw limb connections ---
    # We draw low-confidence connections on a separate overlay for alpha blending
    overlay_low = output.copy()

    for region, connections in SKELETON_CONNECTIONS.items():
        color = region_colors[region]
        for idx_a, idx_b in connections:
            # Occlusion-aware: skip if EITHER endpoint is hidden
            if hidden_mask[idx_a] or hidden_mask[idx_b]:
                continue

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
        # Occlusion-aware: skip hidden joints entirely
        if hidden_mask[idx]:
            continue

        pt = _to_point(keypoints[idx])
        if pt is None:
            continue

        # Joint dot color: white fill with a thin region-based outline
        # Or stress color if provided
        dot_color = stress_colors.get(idx, (255, 255, 255))
        
        if low_mask[idx]:
            cv2.circle(overlay_dots, pt, joint_radius, dot_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(output, pt, joint_radius, dot_color, -1, cv2.LINE_AA)
            cv2.circle(output, pt, joint_radius, (80, 80, 80), 1, cv2.LINE_AA)

    # Blend low-confidence dots
    cv2.addWeighted(overlay_dots, low_alpha, output, 1 - low_alpha, 0, output)

    return output


def render_dynamic_hud(
    frame: np.ndarray,
    events: list[dict],
) -> np.ndarray:
    """Draw sleek, non-intrusive 'pills' for significant events.
    
    Args:
        frame: BGR image to draw on.
        events: List of event dictionaries containing 'joint', 'angle', 'type'.
    
    Returns:
        Frame with dynamic HUD elements.
    """
    if not events:
        return frame.copy()
        
    output = frame.copy()
    
    # Starting position for pills (bottom left)
    start_x = 30
    start_y = frame.shape[0] - 50
    pill_spacing = 40
    
    for i, event in enumerate(events):
        joint_name = event["joint"].replace("_", " ").title()
        angle = event["angle"]
        text = f"{joint_name} Lock: {angle:.0f} deg"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        pill_x1 = start_x
        pill_y1 = start_y - i * pill_spacing - text_h - 10
        pill_x2 = start_x + text_w + 20
        pill_y2 = start_y - i * pill_spacing + 10
        
        # Draw pill background (dark gray, rounded via rectangle)
        cv2.rectangle(output, (pill_x1, pill_y1), (pill_x2, pill_y2), (40, 40, 40), -1)
        # Draw border
        cv2.rectangle(output, (pill_x1, pill_y1), (pill_x2, pill_y2), (100, 100, 100), 1)
        
        # Draw text
        text_x = pill_x1 + 10
        text_y = pill_y2 - 10 - baseline // 2
        cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    # Alpha blend the HUD slightly
    alpha = 0.85
    result = frame.copy()
    cv2.addWeighted(output, alpha, result, 1 - alpha, 0, result)
    
    return result


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
