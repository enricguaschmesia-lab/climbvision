"""HUD overlay rendering for annotated climbing videos."""

from __future__ import annotations

import cv2
import numpy as np


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_THICKNESS = 1
_TEXT_COLOR = (240, 240, 240)
_MISSING_COLOR = (180, 180, 180)
_BACKGROUND_COLOR = (18, 18, 18)
_PADDING_X = 14
_PADDING_Y = 14
_LINE_SPACING = 10
_BOX_MARGIN = 18
_JOINT_ROWS: list[tuple[tuple[str, str], tuple[str, str]]] = [
    (("left_elbow", "L-ELB"), ("right_elbow", "R-ELB")),
    (("left_shoulder", "L-SHD"), ("right_shoulder", "R-SHD")),
    (("left_hip", "L-HIP"), ("right_hip", "R-HIP")),
    (("left_knee", "L-KNE"), ("right_knee", "R-KNE")),
]


def render_hud(
    frame: np.ndarray,
    frame_idx: int,
    timestamp: float,
    joint_angles: dict[str, float] | None,
    config: dict,
    angles_config: dict | None = None,
) -> np.ndarray:
    """Render the HUD info box on a video frame.

    Args:
        frame: Input BGR frame. The input is not mutated.
        frame_idx: Frame index to display.
        timestamp: Frame timestamp in seconds.
        joint_angles: Joint angle map, or ``None`` if unavailable.
        config: Either the ``overlay.hud`` config section or the full app
            config dictionary.
        angles_config: Optional ``overlay.angles`` config used for the same
            efficiency color ranges as the angle arcs.

    Returns:
        A new frame with the HUD box rendered.
    """
    output = frame.copy()
    hud_config, resolved_angles_config = _resolve_configs(config, angles_config)

    font_scale = float(hud_config.get("font_scale", 0.6))
    background_alpha = float(hud_config.get("background_alpha", 0.5))
    position = hud_config.get("position", "top_right")

    lines = _build_lines(frame_idx, timestamp, joint_angles, resolved_angles_config)
    line_metrics = [_measure_segments(line, font_scale) for line in lines]

    content_width = max((width for width, _, _ in line_metrics), default=0)
    content_height = sum(height for _, height, _ in line_metrics)
    content_height += max(0, len(line_metrics) - 1) * _LINE_SPACING

    box_width = content_width + (_PADDING_X * 2)
    box_height = content_height + (_PADDING_Y * 2)
    origin_x, origin_y = _resolve_box_origin(
        frame_shape=frame.shape,
        box_width=box_width,
        box_height=box_height,
        position=position,
    )

    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (origin_x, origin_y),
        (origin_x + box_width, origin_y + box_height),
        _BACKGROUND_COLOR,
        -1,
    )
    cv2.addWeighted(
        overlay,
        background_alpha,
        output,
        1.0 - background_alpha,
        0,
        output,
    )

    text_y = origin_y + _PADDING_Y
    for line, (_, line_height, baseline) in zip(lines, line_metrics):
        text_y += line_height
        _draw_segments(output, line, origin_x + _PADDING_X, text_y, font_scale)
        text_y += _LINE_SPACING + baseline

    return output


def _resolve_configs(config: dict, angles_config: dict | None) -> tuple[dict, dict]:
    """Normalize HUD and angle config inputs."""
    if "overlay" in config:
        hud_config = config["overlay"].get("hud", {})
        if angles_config is None:
            angles_config = config["overlay"].get("angles", {})
    else:
        hud_config = config
        if angles_config is None:
            angles_config = {}

    return hud_config, angles_config


def _build_lines(
    frame_idx: int,
    timestamp: float,
    joint_angles: dict[str, float] | None,
    angles_config: dict,
) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """Create color-coded HUD text segments for each line."""
    lines: list[list[tuple[str, tuple[int, int, int]]]] = [
        [(f"Frame: {frame_idx}  Time: {timestamp:.2f}s", _TEXT_COLOR)]
    ]

    for left_joint, right_joint in _JOINT_ROWS:
        line: list[tuple[str, tuple[int, int, int]]] = []
        for joint_name, label in (left_joint, right_joint):
            value_text, value_color = _format_angle_value(
                joint_name=joint_name,
                joint_angles=joint_angles,
                angles_config=angles_config,
            )
            if line:
                line.append(("  ", _TEXT_COLOR))
            line.append((f"{label}: ", _TEXT_COLOR))
            line.append((value_text, value_color))
        lines.append(line)

    return lines


def _format_angle_value(
    joint_name: str,
    joint_angles: dict[str, float] | None,
    angles_config: dict,
) -> tuple[str, tuple[int, int, int]]:
    """Format one joint angle value and its status color."""
    if not joint_angles or joint_name not in joint_angles:
        return "--", _MISSING_COLOR

    angle = float(joint_angles[joint_name])
    return f"{angle:.0f}", _angle_color(angle, angles_config)


def _angle_color(angle: float, angles_config: dict) -> tuple[int, int, int]:
    """Select a HUD value color using the same ranges as angle arcs."""
    efficient_range = angles_config.get("efficient_range", [150, 180])
    warning_range = angles_config.get("warning_range", [90, 150])
    efficient_color = tuple(angles_config.get("efficient_color", [60, 220, 60]))
    warning_color = tuple(angles_config.get("warning_color", [0, 180, 255]))
    error_color = tuple(angles_config.get("error_color", [20, 20, 240]))

    if efficient_range[0] <= angle <= efficient_range[1]:
        return efficient_color
    if warning_range[0] <= angle <= warning_range[1]:
        return warning_color
    return error_color


def _measure_segments(
    segments: list[tuple[str, tuple[int, int, int]]],
    font_scale: float,
) -> tuple[int, int, int]:
    """Measure total width, line height, and max baseline for a segment row."""
    total_width = 0
    max_height = 0
    max_baseline = 0

    for text, _ in segments:
        (width, height), baseline = cv2.getTextSize(
            text,
            _FONT,
            font_scale,
            _FONT_THICKNESS,
        )
        total_width += width
        max_height = max(max_height, height)
        max_baseline = max(max_baseline, baseline)

    return total_width, max_height, max_baseline


def _resolve_box_origin(
    frame_shape: tuple[int, ...],
    box_width: int,
    box_height: int,
    position: str,
) -> tuple[int, int]:
    """Resolve HUD box origin from the configured corner."""
    frame_height, frame_width = frame_shape[:2]

    if position == "top_left":
        return _BOX_MARGIN, _BOX_MARGIN
    if position == "bottom_right":
        return (
            max(_BOX_MARGIN, frame_width - box_width - _BOX_MARGIN),
            max(_BOX_MARGIN, frame_height - box_height - _BOX_MARGIN),
        )

    return (
        max(_BOX_MARGIN, frame_width - box_width - _BOX_MARGIN),
        _BOX_MARGIN,
    )


def _draw_segments(
    frame: np.ndarray,
    segments: list[tuple[str, tuple[int, int, int]]],
    x: int,
    y: int,
    font_scale: float,
) -> None:
    """Draw one HUD text line segment-by-segment."""
    cursor_x = x

    for text, color in segments:
        cv2.putText(
            frame,
            text,
            (cursor_x, y),
            _FONT,
            font_scale,
            (0, 0, 0),
            _FONT_THICKNESS + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (cursor_x, y),
            _FONT,
            font_scale,
            color,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
        (width, _), _ = cv2.getTextSize(text, _FONT, font_scale, _FONT_THICKNESS)
        cursor_x += width
