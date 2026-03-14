"""Joint angle computation from pose keypoints.

Computes angles at key climbing joints (elbows, shoulders, hips, knees)
using 3-point triplets from MediaPipe landmarks. Uses 2-D pixel
coordinates (x, y) only, because MediaPipe's z (relative depth) is in
a different scale and unreliable for posterior climbing views.

For a triplet A-B-C where B is the joint center, the angle is computed as:
    angle = arccos(dot(BA, BC) / (|BA| * |BC|))
"""

from __future__ import annotations

import numpy as np

from src.utils.geometry import angle_between_vectors, vector_from_points


# MediaPipe landmark indices for the joints we care about.
# Each entry: joint_name -> (endpoint_a_idx, joint_center_idx, endpoint_c_idx)
# The angle is measured at the joint center (B) between rays BA and BC.
JOINT_TRIPLETS: dict[str, tuple[int, int, int]] = {
    # Elbow angle: wrist - elbow - shoulder
    "left_elbow":    (15, 13, 11),  # left_wrist, left_elbow, left_shoulder
    "right_elbow":   (16, 14, 12),  # right_wrist, right_elbow, right_shoulder
    # Shoulder angle: elbow - shoulder - hip
    "left_shoulder":  (13, 11, 23),  # left_elbow, left_shoulder, left_hip
    "right_shoulder": (14, 12, 24),  # right_elbow, right_shoulder, right_hip
    # Hip angle: shoulder - hip - knee
    "left_hip":   (11, 23, 25),  # left_shoulder, left_hip, left_knee
    "right_hip":  (12, 24, 26),  # right_shoulder, right_hip, right_knee
    # Knee angle: hip - knee - ankle
    "left_knee":  (23, 25, 27),  # left_hip, left_knee, left_ankle
    "right_knee": (24, 26, 28),  # right_hip, right_knee, right_ankle
}


def compute_joint_angles(
    keypoints: np.ndarray,
    min_visibility: float = 0.0,
) -> dict[str, float]:
    """Compute angles at key climbing joints from pose keypoints.

    For each joint triplet A-B-C, computes the angle at B using 2-D
    pixel coordinates (x, y). The z coordinate is excluded because
    MediaPipe's depth is normalized differently from x/y pixel space,
    making mixed 3-D angles unreliable.

    Args:
        keypoints: (33, 4) array of [x_px, y_px, z, visibility] per landmark.
        min_visibility: Minimum visibility threshold. Joints where any of
            the three landmarks falls below this are excluded from the
            result. Defaults to 0.0 (include all).

    Returns:
        Dictionary mapping joint name (e.g. ``"left_elbow"``) to angle in
        degrees. Joints with insufficient visibility or non-finite
        coordinates are omitted.
    """
    angles: dict[str, float] = {}

    for joint_name, (idx_a, idx_b, idx_c) in JOINT_TRIPLETS.items():
        # Check visibility of all three landmarks
        vis_a = keypoints[idx_a, 3]
        vis_b = keypoints[idx_b, 3]
        vis_c = keypoints[idx_c, 3]

        if min(vis_a, vis_b, vis_c) < min_visibility:
            continue

        # Extract 2-D pixel coordinates only (x, y)
        point_a = keypoints[idx_a, :2]
        point_b = keypoints[idx_b, :2]
        point_c = keypoints[idx_c, :2]

        # Guard against non-finite coordinates (NaN, inf)
        if not (np.all(np.isfinite(point_a))
                and np.all(np.isfinite(point_b))
                and np.all(np.isfinite(point_c))):
            continue

        # Vectors from joint center B to endpoints A and C
        ba = vector_from_points(point_b, point_a)
        bc = vector_from_points(point_b, point_c)

        angle = angle_between_vectors(ba, bc)
        angles[joint_name] = angle

    return angles
