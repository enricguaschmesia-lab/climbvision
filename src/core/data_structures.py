"""ClimbVision shared data structures.

All analysis modules communicate through these dataclasses.
No module should define its own ad-hoc data formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ClimbingPhase(Enum):
    """Climbing movement phases (Beltrán et al. 2023)."""
    UNKNOWN = "unknown"
    REST = "rest"
    PREPARATION = "preparation"
    REACHING = "reaching"
    STABILIZATION = "stabilization"


class ErrorSeverity(Enum):
    """Severity levels for detected technique errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class HoldDetection:
    """A single detected climbing hold."""
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    center: tuple[int, int]             # Center pixel coordinates
    color_label: str                    # Classified color name
    confidence: float                   # Detection confidence 0-1
    hold_id: int = -1                   # Assigned ID in route ordering


@dataclass
class MoveEvent:
    """A single hand or foot placement move."""
    limb: str                           # "left_hand", "right_hand", "left_foot", "right_foot"
    from_hold: HoldDetection | None     # Starting hold (None if starting move)
    to_hold: HoldDetection | None       # Target hold (None if falling)
    start_frame: int
    end_frame: int
    start_time: float                   # seconds
    end_time: float                     # seconds
    peak_velocity: float                # px/frame
    distance_px: float                  # pixels traveled


@dataclass
class ErrorEvent:
    """A detected technique error."""
    error_type: str                     # Key from error_rules.yaml
    description: str                    # Human-readable description
    severity: ErrorSeverity
    frame_idx: int
    timestamp: float
    feedback: str                       # Coaching tip
    involved_joints: list[str] = field(default_factory=list)


@dataclass
class FrameData:
    """Per-frame data assembled by the pipeline."""
    frame_idx: int
    timestamp: float                    # seconds from start
    image: np.ndarray | None            # Raw BGR frame (H, W, 3); None during Pass 1 storage
    keypoints: np.ndarray | None        # (33, 4) — x, y, z, confidence; None if detection failed
    keypoint_confidence: float          # Mean visibility across all landmarks
    holds: list[HoldDetection] | None   # All detected holds (from reference frame)
    route_holds: list[HoldDetection] | None  # Filtered to target route
    # Phase 1 convenience — will migrate to PoseAnalysis in Phase 3+
    joint_angles: dict[str, float] | None = None


@dataclass
class PoseAnalysis:
    """Biomechanical analysis results for a single frame."""
    frame_idx: int
    timestamp: float
    joint_angles: dict[str, float]      # Joint name → angle in degrees
    com_position: tuple[float, float] | None  # Center of mass (x, y) in pixels
    balance_polygon: np.ndarray | None  # Convex hull vertices of contact points
    com_in_polygon: bool                # Is CoM projection inside support base?
    contact_points: dict[str, HoldDetection | None]  # Limb → hold it's on (None if not on a hold)
    phase: ClimbingPhase
    errors: list[ErrorEvent] = field(default_factory=list)
    force_estimates: dict[str, float] | None = None  # Limb → estimated force magnitude


@dataclass
class ClimbMetrics:
    """Aggregate metrics for the entire climb."""
    total_time_s: float
    num_moves: int
    avg_move_duration_s: float
    longest_pause_s: float
    percent_time_moving: float
    percent_time_resting: float
    com_path_length_px: float
    com_path_length_m: float | None     # None if not calibrated
    path_efficiency: float              # 0-1, ideal/actual path ratio
    geometric_entropy: float | None     # GIE (Cordier 1993)
    jerk_coefficient: float | None      # Normalized jerk (Seifert 2014)
    avg_elbow_angle_at_rest: float | None
    hand_force_ratio: float | None      # Avg hand force / body weight
    route_completion_pct: float         # 0-100
    error_counts: dict[str, int] = field(default_factory=dict)
    phase_durations: dict[str, float] = field(default_factory=dict)


@dataclass
class ClimbAnalysis:
    """Complete analysis of a single climbing attempt."""
    video_path: str
    fps: float
    total_frames: int
    frame_width: int
    frame_height: int
    frames: list[FrameData] = field(default_factory=list)
    pose_analyses: list[PoseAnalysis] = field(default_factory=list)
    moves: list[MoveEvent] = field(default_factory=list)
    all_errors: list[ErrorEvent] = field(default_factory=list)
    metrics: ClimbMetrics | None = None
