"""Vector math and geometric utilities for biomechanical analysis.

Provides basic vector operations used by joint angle computation,
center of mass calculation, and balance analysis.
"""

from __future__ import annotations

import numpy as np


def vector_from_points(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute the vector from point p1 to point p2.

    Args:
        p1: Origin point as a 1-D numpy array (2-D or 3-D).
        p2: Target point as a 1-D numpy array (same dimensionality as p1).

    Returns:
        The vector p2 - p1 as a numpy array.
    """
    return np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle between two vectors in degrees.

    Uses the dot-product formula:
        angle = arccos(dot(v1, v2) / (|v1| * |v2|))

    Args:
        v1: First vector as a 1-D numpy array.
        v2: Second vector as a 1-D numpy array.

    Returns:
        Angle in degrees in the range [0, 180].
        Returns 0.0 if either vector has zero length.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0

    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    # Clamp to [-1, 1] to handle floating-point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
