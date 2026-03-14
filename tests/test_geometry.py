"""Tests for src/utils/geometry.py.

Uses known geometric configurations to verify vector math.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.geometry import angle_between_vectors, vector_from_points


# ---------------------------------------------------------------------------
# Tests — vector_from_points
# ---------------------------------------------------------------------------


class TestVectorFromPoints:
    """Tests for vector_from_points()."""

    def test_simple_2d(self) -> None:
        """Basic 2-D subtraction."""
        v = vector_from_points(np.array([1, 2]), np.array([4, 6]))
        np.testing.assert_array_equal(v, [3, 4])

    def test_simple_3d(self) -> None:
        """Basic 3-D subtraction."""
        v = vector_from_points(np.array([0, 0, 0]), np.array([1, 2, 3]))
        np.testing.assert_array_equal(v, [1, 2, 3])

    def test_same_point_gives_zero(self) -> None:
        """Same point should produce a zero vector."""
        v = vector_from_points(np.array([5, 5, 5]), np.array([5, 5, 5]))
        np.testing.assert_array_equal(v, [0, 0, 0])

    def test_negative_direction(self) -> None:
        """Reversing the points should negate the vector."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 5.0, 6.0])
        v_fwd = vector_from_points(p1, p2)
        v_rev = vector_from_points(p2, p1)
        np.testing.assert_array_almost_equal(v_fwd, -v_rev)

    def test_returns_float64(self) -> None:
        """Output should be float64 regardless of input type."""
        v = vector_from_points(np.array([1, 2]), np.array([3, 4]))
        assert v.dtype == np.float64


# ---------------------------------------------------------------------------
# Tests — angle_between_vectors
# ---------------------------------------------------------------------------


class TestAngleBetweenVectors:
    """Tests for angle_between_vectors()."""

    def test_right_angle_2d(self) -> None:
        """Perpendicular 2-D vectors should give 90 degrees."""
        angle = angle_between_vectors(np.array([1, 0]), np.array([0, 1]))
        assert angle == pytest.approx(90.0, abs=0.01)

    def test_right_angle_3d(self) -> None:
        """Perpendicular 3-D vectors should give 90 degrees."""
        angle = angle_between_vectors(np.array([1, 0, 0]), np.array([0, 1, 0]))
        assert angle == pytest.approx(90.0, abs=0.01)

    def test_parallel_vectors(self) -> None:
        """Same direction should give 0 degrees."""
        angle = angle_between_vectors(np.array([1, 0, 0]), np.array([2, 0, 0]))
        assert angle == pytest.approx(0.0, abs=0.01)

    def test_antiparallel_vectors(self) -> None:
        """Opposite direction should give 180 degrees."""
        angle = angle_between_vectors(np.array([1, 0, 0]), np.array([-1, 0, 0]))
        assert angle == pytest.approx(180.0, abs=0.01)

    def test_45_degrees(self) -> None:
        """Known 45-degree pair."""
        angle = angle_between_vectors(np.array([1, 0]), np.array([1, 1]))
        assert angle == pytest.approx(45.0, abs=0.01)

    def test_60_degrees(self) -> None:
        """Known 60-degree pair."""
        angle = angle_between_vectors(
            np.array([1, 0, 0]),
            np.array([0.5, np.sqrt(3) / 2, 0]),
        )
        assert angle == pytest.approx(60.0, abs=0.01)

    def test_zero_vector_returns_zero(self) -> None:
        """A zero-length vector should return 0.0 (degenerate case)."""
        angle = angle_between_vectors(np.array([0, 0, 0]), np.array([1, 0, 0]))
        assert angle == 0.0

    def test_both_zero_vectors(self) -> None:
        """Two zero vectors should return 0.0."""
        angle = angle_between_vectors(np.array([0, 0]), np.array([0, 0]))
        assert angle == 0.0

    def test_order_independent(self) -> None:
        """angle(v1, v2) should equal angle(v2, v1)."""
        v1 = np.array([3, 4, 1])
        v2 = np.array([1, -2, 5])
        assert angle_between_vectors(v1, v2) == pytest.approx(
            angle_between_vectors(v2, v1), abs=0.001
        )

    def test_magnitude_independent(self) -> None:
        """Scaling a vector should not change the angle."""
        v1 = np.array([1.0, 1.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        a1 = angle_between_vectors(v1, v2)
        a2 = angle_between_vectors(v1 * 100, v2 * 0.01)
        assert a1 == pytest.approx(a2, abs=0.001)
