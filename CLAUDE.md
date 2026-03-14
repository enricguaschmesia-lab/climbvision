# ClimbVision — Claude Code Project Instructions

## Project Overview

ClimbVision is a computer-vision pipeline that analyzes bouldering videos and produces:
1. An **annotated video** with skeleton overlay, joint angles, center-of-mass trajectory, balance polygon, force vectors, and error highlights.
2. A **post-climb evaluation report** (HTML) with metrics, graphs, and technique feedback.

**Input:** Static-camera RGB video, single climber, indoor bouldering wall, route defined by same-color holds.

**Tech stack:** Python 3.11+, OpenCV, MediaPipe (pose), Ultralytics YOLOv8 (holds), NumPy, SciPy, Matplotlib/Plotly, Jinja2.

---

## Repository Structure

```
climbvision/
├── CLAUDE.md                  ← You are here. Orchestrator instructions.
├── agents/                    ← Subagent instruction files
│   ├── ORCHESTRATOR.md        ← Main orchestration logic (Claude Code reads this)
│   ├── pose-engineer.md       ← Pose estimation & skeleton pipeline
│   ├── hold-detective.md      ← Hold detection & route mapping
│   ├── biomechanics-engine.md ← CoM, angles, forces, balance
│   ├── movement-analyst.md    ← Phase segmentation, move decomposition, errors
│   ├── viz-renderer.md        ← Video overlay & report generation (Gemini-assisted)
│   └── codex-researcher.md    ← Deep technical tasks (Codex-assisted)
├── docs/
│   ├── literature_review.md
│   ├── feature_brainstorm.md
│   ├── phase_plan.md          ← Detailed phase-by-phase dev plan
│   └── architecture.md        ← System architecture diagram
├── src/
│   ├── core/                  ← Foundation modules
│   │   ├── __init__.py
│   │   ├── video_io.py        ← Video loading, frame extraction, writing
│   │   ├── pose_estimator.py  ← MediaPipe / YOLO pose wrapper
│   │   ├── hold_detector.py   ← YOLO hold detection + color classification
│   │   └── calibration.py     ← Pixel-to-metric conversion, wall plane
│   ├── analysis/              ← Biomechanics & technique modules
│   │   ├── __init__.py
│   │   ├── joint_angles.py    ← Angle computation from keypoints
│   │   ├── center_of_mass.py  ← CoM from segment weights
│   │   ├── balance.py         ← Support polygon, stability analysis
│   │   ├── forces.py          ← Static equilibrium force estimation
│   │   ├── phases.py          ← Climbing phase state machine
│   │   ├── moves.py           ← Move-by-move decomposition
│   │   ├── errors.py          ← Technique error detection rules
│   │   └── metrics.py         ← GIE, jerk, path efficiency, summaries
│   ├── visualization/         ← Rendering & reporting
│   │   ├── __init__.py
│   │   ├── skeleton_overlay.py
│   │   ├── angle_overlay.py
│   │   ├── com_overlay.py
│   │   ├── force_overlay.py
│   │   ├── phase_bar.py
│   │   ├── report_generator.py
│   │   └── templates/         ← Jinja2 HTML report templates
│   └── utils/
│       ├── __init__.py
│       ├── smoothing.py       ← Kalman, Savitzky-Golay filters
│       ├── geometry.py        ← Vector math, projections, convex hull
│       └── anthropometry.py   ← De Leva body segment parameters
├── tests/
│   ├── test_joint_angles.py
│   ├── test_center_of_mass.py
│   └── test_phases.py
├── configs/
│   ├── default.yaml           ← Thresholds, model paths, display settings
│   └── error_rules.yaml       ← Error detection thresholds per phase
├── data/
│   ├── models/                ← Downloaded model weights (gitignored)
│   └── samples/               ← Sample videos for testing
├── scripts/
│   ├── run_analysis.py        ← Main entry point
│   └── download_models.py     ← Fetch YOLO + MediaPipe weights
├── requirements.txt
└── pyproject.toml
```

---

## Multi-Agent Workflow

### Agent Roles

This project uses a **hub-and-spoke** agent architecture. Claude Code is the orchestrator (hub). It delegates to specialized subagents (spokes) for focused implementation tasks.

| Agent | File | Role | Delegation Target |
|-------|------|------|-------------------|
| **Orchestrator** | `agents/ORCHESTRATOR.md` | Coordinates all work, manages phase transitions, reviews integration | Claude Code (main) |
| **Pose Engineer** | `agents/pose-engineer.md` | Pose estimation pipeline, skeleton extraction, temporal smoothing | Claude Code subagent |
| **Hold Detective** | `agents/hold-detective.md` | YOLO hold detection, HSV color classification, route isolation | Claude Code subagent |
| **Biomechanics Engine** | `agents/biomechanics-engine.md` | Joint angles, CoM, forces, balance polygon computation | Claude Code subagent |
| **Movement Analyst** | `agents/movement-analyst.md` | Phase segmentation, move decomposition, error detection rules | Claude Code subagent |
| **Viz Renderer** | `agents/viz-renderer.md` | All overlay rendering, HTML report generation, UI decisions | **Gemini subagent** for design/layout decisions |
| **Codex Researcher** | `agents/codex-researcher.md` | Complex algorithm design, mathematical derivations, optimization | **Codex subagent** for deep technical computation |

### When to Use Each External Agent

**Gemini (via `agents/viz-renderer.md`):**
- ALL visualization decisions: color palettes, overlay layout, graph styling
- HTML report template design and CSS
- UI/UX decisions for the annotated video (what to show, where, how big)
- Review of visual output quality (screenshot → Gemini for critique)
- Any task involving graphical design, typography, or visual hierarchy

**Codex (via `agents/codex-researcher.md`):**
- Complex mathematical derivations (static equilibrium solver, GIE computation)
- Algorithm optimization (vectorized NumPy operations for frame-by-frame processing)
- Deep debugging of numerical issues (pose jitter, force estimation instability)
- Architecture decisions requiring long reasoning chains
- Literature-informed implementation details (translating paper equations to code)

### Agent Spawning Pattern

When implementing a feature, the orchestrator follows this pattern:

```
1. Read the relevant agent .md file for context
2. Define the task scope (which files to create/modify)
3. Spawn the subagent with:
   - Task description referencing the agent .md
   - Input: current codebase state, relevant docs
   - Output: specific files to create or modify
   - Constraints: must pass existing tests, follow project conventions
4. Review the subagent's output
5. Run tests
6. Integrate with the rest of the pipeline
```

For Gemini/Codex tasks:
```
1. Prepare a focused prompt from the agent .md instructions
2. Include ONLY the context needed (not the full codebase)
3. Ask for a specific deliverable (code snippet, design spec, algorithm)
4. Integrate the response into the project via Claude Code
```

**Before spawning an external subagent, read its agent file for the exact CLI invocation:**
- **Codex:** Read `agents/codex-researcher.md` → "CLI Invocation" section for `codex exec` usage
- **Gemini:** Read `agents/viz-renderer.md` → "CLI Invocation" section for `gemini --prompt` usage

---

## Development Phases

Each phase has a **definition of done** and a **handoff checklist**.

### Phase 1: Skeleton + Basic Overlay (Current)
→ See `docs/phase_plan.md` for full details and the starter prompt.

### Phase 2: Hold Detection + Route Mapping
### Phase 3: Core Biomechanics
### Phase 4: Technique Analysis
### Phase 5: Comparison & Polish

---

## Code Conventions

- **Python 3.11+**, type hints on all public functions
- **Docstrings**: Google style on every module, class, and public function
- **Testing**: pytest, one test file per source module, test edge cases
- **Naming**: `snake_case` for files/functions, `PascalCase` for classes
- **Config**: No magic numbers in code — all thresholds go in `configs/default.yaml`
- **Data flow**: Every analysis module takes a `FrameData` dataclass as input and returns a result dataclass. No module reads video frames directly.
- **Error handling**: Graceful degradation — if pose estimation fails on a frame, skip it and interpolate. Never crash the pipeline on a single bad frame.

## Key Data Structures

```python
@dataclass
class FrameData:
    """Per-frame data passed through the pipeline."""
    frame_idx: int
    timestamp: float
    image: np.ndarray                          # Raw BGR frame
    keypoints: np.ndarray | None               # (33, 4) — x, y, z, confidence
    holds: list[HoldDetection] | None          # Detected holds with colors
    route_holds: list[HoldDetection] | None    # Filtered to target route

@dataclass
class HoldDetection:
    """A single detected climbing hold."""
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    center: tuple[int, int]
    color_label: str
    confidence: float

@dataclass  
class PoseAnalysis:
    """Biomechanical analysis results for a single frame."""
    joint_angles: dict[str, float]      # Joint name → angle in degrees
    com_position: tuple[float, float]   # Center of mass (x, y) in pixels
    balance_polygon: np.ndarray | None  # Convex hull of contact points
    com_in_polygon: bool                # Is CoM inside support base?
    contact_points: list[str]           # Which limbs are on holds
    phase: str                          # Current climbing phase

@dataclass
class ClimbAnalysis:
    """Complete analysis of a climbing attempt."""
    frames: list[FrameData]
    pose_analyses: list[PoseAnalysis]
    moves: list[MoveEvent]
    errors: list[ErrorEvent]
    metrics: ClimbMetrics
```

---

## Running the Pipeline

```bash
# Setup
pip install -r requirements.txt
python scripts/download_models.py

# Analyze a video
python scripts/run_analysis.py --video path/to/climb.mp4 --output output/

# Run tests
pytest tests/ -v
```
