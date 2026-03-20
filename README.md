# ClimbVision

**Computer vision pipeline for bouldering video analysis.**

Upload a climbing video. Get back an annotated video with biomechanical overlays and a detailed technique evaluation report.

## What It Does (Planned)

- **Skeleton tracking** -- 33-point body pose estimation with temporal smoothing
- **Joint angle analysis** -- Real-time angle computation at elbows, shoulders, hips, knees
- **Center of mass** -- Trajectory tracking with geometric entropy (efficiency metric)
- **Balance analysis** -- Support polygon and stability assessment
- **Hold detection** -- Automatic route recognition by hold color
- **Technique errors** -- Rule-based detection of common beginner mistakes
- **Force estimation** -- Static equilibrium model for limb loading
- **Multi-attempt comparison** -- DTW-aligned beta comparison across attempts

## Phase 1: Skeleton + Basic Overlay (Complete)

Phase 1 delivers an end-to-end pipeline that takes a climbing video as input and produces an annotated video with:

- **Pose estimation** via MediaPipe PoseLandmarker (heavy model, VIDEO mode with temporal tracking)
- **Temporal smoothing** using Savitzky-Golay filtering with adaptive window (scales with FPS)
- **Gap interpolation** filling short detection gaps via linear interpolation
- **Chunked processing** for memory-bounded analysis of long videos (rolling buffer with overlap)
- **Skeleton overlay** with region-colored limbs (arms, legs, trunk) and occlusion-aware hiding
- **Stress-colored joints** showing angle status: green (efficient), amber (warning), red (high stress)
- **Dynamic HUD** with event-driven pill notifications for sustained stressful joint positions
- **Joint angle computation** for 8 key climbing joints (elbows, shoulders, hips, knees) using 2D pixel coordinates

### Quick Start

```bash
pip install -r requirements.txt
python scripts/download_models.py
python scripts/run_analysis.py --video path/to/climb.mp4 --output output/
```

### Testing & Development

```bash
pytest tests/ -v                                    # Run all tests (126 tests)
python scripts/test_pose.py video.mp4 --save out.mp4  # Dev testing with overlay
python scripts/test_pose.py video.mp4 --raw           # Per-frame mode (no smoothing)
```

### Two-Pass Pipeline Architecture

1. **Pass 1 (Pose Extraction):** Reads video in memory-bounded chunks, runs MediaPipe VIDEO-mode inference once per frame, applies interpolation and Savitzky-Golay smoothing with overlap context. Outputs `dict[int, FrameData]` keyed by frame index.

2. **Pass 2 (Rendering):** Re-reads the video frame by frame, looks up pose data from Pass 1, computes stress colors from joint angles, renders skeleton overlay and dynamic HUD, writes annotated frames to the output video.

## Architecture

See `docs/architecture.md` for the system diagram and data flow.

## Status

- [x] Phase 1: Skeleton + Basic Overlay
- [ ] Phase 2: Hold Detection + Route Mapping
- [ ] Phase 3: Core Biomechanics
- [ ] Phase 4: Technique Analysis
- [ ] Phase 5: Comparison & Polish
