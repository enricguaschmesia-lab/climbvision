"""Quick test: run pose estimation on a video and visualize results.

Usage:
    python scripts/test_pose.py path/to/video.mp4
    python scripts/test_pose.py path/to/video.mp4 --show       # display with skeleton overlay
    python scripts/test_pose.py path/to/video.mp4 --save out.mp4  # save annotated video
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.core.video_io import load_video, frame_iterator, load_config, VideoWriter
from src.core.pose_estimator import PoseEstimator
from src.analysis.joint_angles import compute_joint_angles
from src.visualization.skeleton_overlay import render_skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pose estimation on a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--show", action="store_true", help="Display frames with skeleton overlay")
    parser.add_argument("--save", type=str, default=None, help="Save annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    # Load video metadata
    meta = load_video(args.video)
    print(f"Video: {meta.path}")
    print(f"  {meta.width}x{meta.height} @ {meta.fps:.1f} fps, "
          f"{meta.total_frames} frames ({meta.duration_s:.1f}s)")

    # Load config and init pose estimator
    config = load_config("configs/default.yaml")
    estimator = PoseEstimator(config["pose"])
    skeleton_cfg = config["overlay"]["skeleton"]

    # Optional video writer
    writer = None
    if args.save:
        max_dim = config["video"]["max_dimension"]
        # Determine output dimensions (match what frame_iterator will produce)
        w, h = meta.width, meta.height
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
        writer = VideoWriter(
            args.save,
            fps=meta.fps,
            width=w,
            height=h,
            codec=config["video"]["output_codec"],
        )

    detected = 0
    total = 0

    for idx, ts, frame in frame_iterator(
        args.video, max_dimension=config["video"]["max_dimension"]
    ):
        if args.max_frames and idx >= args.max_frames:
            break

        kp, conf = estimator.estimate_frame(frame)
        total += 1
        display_frame = frame

        if kp is not None:
            detected += 1

            # Draw skeleton overlay
            display_frame = render_skeleton(frame, kp, skeleton_cfg)

            # Compute and display joint angles as text
            angles = compute_joint_angles(kp)
            angle_strs = [f"{k}: {v:.0f}" for k, v in sorted(angles.items())]
            for i, txt in enumerate(angle_strs):
                cv2.putText(
                    display_frame, txt, (10, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                )

            if idx % 30 == 0:
                angle_summary = ", ".join(f"{k}={v:.0f}" for k, v in angles.items()
                                          if "elbow" in k or "knee" in k)
                print(f"  Frame {idx:4d} | conf={conf:.2f} | {angle_summary}")
        else:
            if idx % 30 == 0:
                print(f"  Frame {idx:4d} | NO POSE DETECTED")

        if writer:
            writer.write(display_frame)

        if args.show:
            cv2.imshow("ClimbVision - Pose Test", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"\nSaved annotated video to: {args.save} ({writer.frame_count} frames)")

    estimator.close()

    pct = 100 * detected / total if total > 0 else 0
    print(f"\nDetected pose in {detected}/{total} frames ({pct:.0f}%)")


if __name__ == "__main__":
    main()
