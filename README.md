# ClimbVision

**Computer vision pipeline for bouldering video analysis.**

Upload a climbing video. Get back an annotated video with biomechanical overlays and a detailed technique evaluation report.

## What It Does

- **Skeleton tracking** — 33-point body pose estimation with temporal smoothing
- **Joint angle analysis** — Real-time angle computation at elbows, shoulders, hips, knees
- **Center of mass** — Trajectory tracking with geometric entropy (efficiency metric)
- **Balance analysis** — Support polygon and stability assessment
- **Hold detection** — Automatic route recognition by hold color
- **Technique errors** — Rule-based detection of 6 common beginner mistakes
- **Force estimation** — Static equilibrium model for limb loading
- **Multi-attempt comparison** — DTW-aligned beta comparison across attempts

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_models.py
python scripts/run_analysis.py --video path/to/climb.mp4 --output output/
```

## Development

This project uses a multi-agent architecture with Claude Code as the orchestrator. See `CLAUDE.md` for the full development guide, and `docs/phase_plan.md` for the implementation roadmap.

```bash
pytest tests/ -v          # Run tests
```

## Architecture

See `docs/architecture.md` for the system diagram and data flow.

## Status

- [ ] Phase 1: Skeleton + Basic Overlay
- [ ] Phase 2: Hold Detection + Route Mapping
- [ ] Phase 3: Core Biomechanics
- [ ] Phase 4: Technique Analysis
- [ ] Phase 5: Comparison & Polish
