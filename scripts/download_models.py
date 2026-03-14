#!/usr/bin/env python3
"""Download required model weights for ClimbVision.

Usage:
    python scripts/download_models.py

Downloads:
    - MediaPipe Pose model (auto-downloaded on first use)
    - YOLOv8 pose model (from Ultralytics)
    - YOLOv8 hold detection model (from HuggingFace/Roboflow, if available)
"""

import os
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


def download_yolo_pose():
    """Download YOLOv8 pose estimation model."""
    model_path = MODELS_DIR / "yolov8x-pose.pt"
    if model_path.exists():
        print(f"✓ YOLOv8 pose model already exists at {model_path}")
        return

    print("Downloading YOLOv8x-pose model...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x-pose.pt")
        # Ultralytics auto-downloads to the working directory
        downloaded = Path("yolov8x-pose.pt")
        if downloaded.exists():
            downloaded.rename(model_path)
            print(f"✓ Saved to {model_path}")
    except ImportError:
        print("✗ ultralytics not installed. Run: pip install ultralytics")
    except Exception as e:
        print(f"✗ Download failed: {e}")


def check_mediapipe():
    """Verify MediaPipe is installed (models download on first use)."""
    try:
        import mediapipe  # noqa: F401
        print("✓ MediaPipe installed (pose model downloads on first use)")
    except ImportError:
        print("✗ MediaPipe not installed. Run: pip install mediapipe")


def main():
    """Download all required models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== ClimbVision Model Setup ===\n")
    check_mediapipe()
    download_yolo_pose()
    print("\n=== Setup complete ===")
    print(f"Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
