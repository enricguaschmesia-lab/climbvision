"""Joint angle arc overlay for climbing video analysis.

Draws colored arc indicators at key joints to visualize the angle
between two limb segments. Arcs are color-coded by efficiency:
green (efficient), yellow/amber (warning), red (error). Hip arcs
use a larger radius than other joints to avoid overlap with nearby
knee arcs.

All visual parameters are driven by configs/default.yaml.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.analysis.joint_angles import JOINT_TRIPLETS


# Per-joint arc radius overrides.  Hips get a larger radius to avoid
# visual collision with nearby knee arcs.
_RADIUS_OVERRIDES: dict[str, str] = {
    "left_hip": "radius_hips",
    "right_hip": "radius_hips",
}


def render_angles(
    frame: np.ndarray,
    keypoints: np.ndarray,
    angles: dict[str, float],
    config: dict,
) -> np.ndarray:
    """Draw angle arc indicators at key joints on a video frame.

    For each joint in *angles*, draws a colored arc spanning the
    interior angle between the two adjacent limb segments, plus a
    degree label at the arc midpoint.

    Args:
        frame: BGR image as numpy array (H, W, 3). Not mutated.
        keypoints: (33, 4) array of [x_px, y_px, z, visibility].
        angles: Dictionary mapping joint name to angle in degrees,
            as returned by ``compute_joint_angles()``.
        config: The ``overlay.angles`` section of the YAML config.
            Expected keys:
            - arc_radius: int, default arc radius in pixels
            - radius_hips: int, arc radius for hip joints (optional)
            - thickness: int, arc line thickness
            - font_scale: float, degree label font scale
            - alpha: float, arc opacity (0-1)
            - fill_alpha: float, sector fill opacity (0-1)
            - label_offset: int, label distance from arc path
            - efficient_color: list[int], BGR color for efficient range
            - warning_color: list[int], BGR color for warning range
            - error_color: list[int], BGR color for error range
            - efficient_range: list[int], [min_deg, max_deg]
            - warning_range: list[int], [min_deg, max_deg]
            - hide_threshold: float, visibility below which arcs are
              skipped (inherited from skeleton config if absent)

    Returns:
        A new frame with angle arc overlays drawn.
    """
    output = frame.copy()

    arc_radius_default = config.get("arc_radius", 45)
    arc_radius_hips = config.get("radius_hips", 65)
    thickness = config.get("thickness", 3)
    font_scale = config.get("font_scale", 0.55)
    alpha = config.get("alpha", 0.75)
    fill_alpha = config.get("fill_alpha", 0.1)
    label_offset = config.get("label_offset", 18)
    hide_threshold = config.get("hide_threshold", 0.25)

    efficient_color = tuple(config.get("efficient_color", [60, 220, 60]))
    warning_color = tuple(config.get("warning_color", [0, 180, 255]))
    error_color = tuple(config.get("error_color", [20, 20, 240]))

    efficient_range = config.get("efficient_range", [150, 180])
    warning_range = config.get("warning_range", [90, 150])

    # Overlay layer for alpha blending
    overlay_arcs = output.copy()
    overlay_fill = output.copy()

    for joint_name, angle_deg in angles.items():
        triplet = JOINT_TRIPLETS.get(joint_name)
        if triplet is None:
            continue

        idx_a, idx_b, idx_c = triplet

        # Occlusion-aware: skip if joint center visibility is below threshold
        if keypoints[idx_b, 3] < hide_threshold:
            continue

        # Joint center pixel position
        center = keypoints[idx_b, :2]
        pt_a = keypoints[idx_a, :2]
        pt_c = keypoints[idx_c, :2]

        if not (np.all(np.isfinite(center)) and np.all(np.isfinite(pt_a))
                and np.all(np.isfinite(pt_c))):
            continue

        cx, cy = int(round(center[0])), int(round(center[1]))

        # Select radius
        radius_key = _RADIUS_OVERRIDES.get(joint_name)
        if radius_key == "radius_hips":
            radius = arc_radius_hips
        else:
            radius = arc_radius_default

        # Determine color from angle
        color = _angle_color(
            angle_deg, efficient_range, warning_range,
            efficient_color, warning_color, error_color,
        )

        # Compute angles of the two limb vectors relative to +x axis
        vec_a = pt_a - center
        vec_c = pt_c - center

        angle_a_rad = math.atan2(-vec_a[1], vec_a[0])  # negative y for screen coords
        angle_c_rad = math.atan2(-vec_c[1], vec_c[0])

        # Ensure we draw the interior (smaller) arc
        start_rad, end_rad = _ordered_arc_angles(angle_a_rad, angle_c_rad)

        start_deg_cv = -math.degrees(end_rad)    # cv2 uses clockwise angles
        end_deg_cv = -math.degrees(start_rad)

        # Draw filled sector (very light)
        if fill_alpha > 0:
            cv2.ellipse(
                overlay_fill, (cx, cy), (radius, radius),
                0, start_deg_cv, end_deg_cv, color, -1, cv2.LINE_AA,
            )

        # Draw arc outline
        cv2.ellipse(
            overlay_arcs, (cx, cy), (radius, radius),
            0, start_deg_cv, end_deg_cv, color, thickness, cv2.LINE_AA,
        )

        # Draw small tick marks at arc endpoints
        _draw_tick(overlay_arcs, cx, cy, radius, start_rad, color, thickness)
        _draw_tick(overlay_arcs, cx, cy, radius, end_rad, color, thickness)

        # Draw degree label at angular midpoint, offset outward
        mid_rad = (start_rad + end_rad) / 2.0
        label_dist = radius + label_offset
        lx = int(round(cx + label_dist * math.cos(mid_rad)))
        ly = int(round(cy - label_dist * math.sin(mid_rad)))  # screen y is inverted

        label = f"{angle_deg:.0f}"
        _draw_label(overlay_arcs, label, lx, ly, font_scale, color)

    # Blend fill layer
    if fill_alpha > 0:
        cv2.addWeighted(overlay_fill, fill_alpha, output, 1 - fill_alpha, 0, output)

    # Blend arc layer
    cv2.addWeighted(overlay_arcs, alpha, output, 1 - alpha, 0, output)

    return output


def _angle_color(
    angle_deg: float,
    efficient_range: list[int],
    warning_range: list[int],
    efficient_color: tuple[int, ...],
    warning_color: tuple[int, ...],
    error_color: tuple[int, ...],
) -> tuple[int, ...]:
    """Select arc color based on angle value and efficiency ranges.

    Args:
        angle_deg: Joint angle in degrees.
        efficient_range: [min, max] degrees for efficient (green).
        warning_range: [min, max] degrees for warning (yellow).
        efficient_color: BGR tuple for efficient angles.
        warning_color: BGR tuple for warning angles.
        error_color: BGR tuple for error angles.

    Returns:
        BGR color tuple.
    """
    if efficient_range[0] <= angle_deg <= efficient_range[1]:
        return efficient_color
    if warning_range[0] <= angle_deg <= warning_range[1]:
        return warning_color
    return error_color


def _ordered_arc_angles(
    angle_a: float,
    angle_c: float,
) -> tuple[float, float]:
    """Order two radian angles so the arc between them is the smaller one.

    Args:
        angle_a: First vector angle in radians.
        angle_c: Second vector angle in radians.

    Returns:
        (start, end) in radians such that end - start <= pi.
    """
    # Normalize to [0, 2*pi)
    a = angle_a % (2 * math.pi)
    c = angle_c % (2 * math.pi)

    diff = (c - a) % (2 * math.pi)
    if diff <= math.pi:
        return a, a + diff
    else:
        return c, c + (2 * math.pi - diff)


def _draw_tick(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    angle_rad: float,
    color: tuple[int, ...],
    thickness: int,
) -> None:
    """Draw a small perpendicular tick mark at an arc endpoint.

    Args:
        img: Image to draw on (mutated).
        cx: Arc center x.
        cy: Arc center y.
        radius: Arc radius.
        angle_rad: Angle of the endpoint in radians (math convention).
        color: BGR color.
        thickness: Line thickness.
    """
    tick_len = 4
    # Point on the arc
    px = cx + radius * math.cos(angle_rad)
    py = cy - radius * math.sin(angle_rad)
    # Perpendicular direction (tangent to arc)
    tx = -math.sin(angle_rad) * tick_len
    ty = -math.cos(angle_rad) * tick_len
    pt1 = (int(round(px - tx)), int(round(py + ty)))
    pt2 = (int(round(px + tx)), int(round(py - ty)))
    cv2.line(img, pt1, pt2, color, max(1, thickness - 1), cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_scale: float,
    color: tuple[int, ...],
) -> None:
    """Draw a degree label with a dark outline for readability.

    Args:
        img: Image to draw on (mutated).
        text: Label string (e.g. "145").
        x: Label center x position.
        y: Label center y position.
        font_scale: OpenCV font scale.
        color: BGR foreground color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    # Center the text on (x, y)
    tx = x - tw // 2
    ty = y + th // 2
    # Black outline for readability
    cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    # White/color foreground
    cv2.putText(img, text, (tx, ty), font, font_scale, color, 1, cv2.LINE_AA)
