"""Tests for AngleTracker in src/analysis/joint_angles.py."""

from __future__ import annotations

from src.analysis.joint_angles import AngleTracker


class TestAngleTracker:
    """Tests for AngleTracker.update()."""

    def test_no_events_before_history_filled(self) -> None:
        """Should not emit events until history_length frames are tracked."""
        tracker = AngleTracker(history_length=5, stress_threshold=90.0)
        for _ in range(4):
            events = tracker.update({"left_elbow": 45.0})
        assert events == []

    def test_hold_event_emitted(self) -> None:
        """Sustained low angle should emit a hold event."""
        tracker = AngleTracker(
            history_length=5, range_threshold=10.0, stress_threshold=90.0,
        )
        events = []
        for _ in range(5):
            events = tracker.update({"left_elbow": 45.0})
        assert len(events) == 1
        assert events[0]["joint"] == "left_elbow"
        assert events[0]["type"] == "hold"
        assert 40 < events[0]["angle"] < 50

    def test_no_event_for_efficient_angle(self) -> None:
        """Sustained high angle (efficient) should NOT emit events."""
        tracker = AngleTracker(
            history_length=5, range_threshold=10.0, stress_threshold=90.0,
        )
        events = []
        for _ in range(10):
            events = tracker.update({"left_elbow": 170.0})
        assert events == []

    def test_no_event_when_angle_varies(self) -> None:
        """Large variation in angle should prevent hold detection."""
        tracker = AngleTracker(
            history_length=5, range_threshold=5.0, stress_threshold=90.0,
        )
        angles = [40.0, 60.0, 30.0, 55.0, 45.0]
        events = []
        for a in angles:
            events = tracker.update({"left_elbow": a})
        assert events == []

    def test_missing_joint_clears_history(self) -> None:
        """Joint disappearing then reappearing should reset history."""
        tracker = AngleTracker(
            history_length=3, range_threshold=10.0, stress_threshold=90.0,
        )
        for _ in range(2):
            tracker.update({"left_elbow": 45.0})
        # Joint disappears for one frame
        tracker.update({})
        # Reappears — should need full history_length again
        events = tracker.update({"left_elbow": 45.0})
        assert events == []

    def test_multiple_joints_tracked_independently(self) -> None:
        """Each joint should be tracked independently."""
        tracker = AngleTracker(
            history_length=3, range_threshold=10.0, stress_threshold=90.0,
        )
        for _ in range(3):
            events = tracker.update({
                "left_elbow": 45.0,
                "right_elbow": 170.0,
            })
        # Only left_elbow should trigger (below stress threshold)
        joint_names = [e["joint"] for e in events]
        assert "left_elbow" in joint_names
        assert "right_elbow" not in joint_names
