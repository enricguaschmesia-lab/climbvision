"""Quick test: run pose estimation on a video and visualize results.

Usage:
    python scripts/test_pose.py path/to/video.mp4
    python scripts/test_pose.py path/to/video.mp4 --show       # display with skeleton overlay
    python scripts/test_pose.py path/to/video.mp4 --save out.mp4  # save annotated video
    python scripts/test_pose.py path/to/video.mp4 --raw        # per-frame mode (no smoothing)
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.core.video_io import load_video, frame_iterator, load_config, VideoWriter
from src.core.pose_estimator import PoseEstimator
from src.analysis.joint_angles import compute_joint_angles
from src.visualization.skeleton_overlay import render_skeleton


def _render_frame(
    frame: np.ndarray,
    kp: np.ndarray | None,
    conf: float,
    idx: int,
    skeleton_cfg: dict,
) -> tuple[np.ndarray, bool]:
    """Render skeleton overlay and angle text on a single frame.

    Args:
        frame: BGR image.
        kp: Keypoints array (33, 4) or None if no pose detected.
        conf: Overall confidence float.
        idx: Frame index for progress logging.
        skeleton_cfg: Skeleton overlay config dict.

    Returns:
        Tuple of (annotated frame, whether pose was detected).
    """
    if kp is None:
        if idx % 30 == 0:
            print(f"  Frame {idx:4d} | NO POSE DETECTED")
        return frame, False

    display_frame = render_skeleton(frame, kp, skeleton_cfg)

    angles = compute_joint_angles(kp)
    angle_strs = [f"{k}: {v:.0f}" for k, v in sorted(angles.items())]
    for i, txt in enumerate(angle_strs):
        cv2.putText(
            display_frame, txt, (10, 20 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    if idx % 30 == 0:
        angle_summary = ", ".join(
            f"{k}={v:.0f}" for k, v in angles.items()
            if "elbow" in k or "knee" in k
        )
        print(f"  Frame {idx:4d} | conf={conf:.2f} | {angle_summary}")

    return display_frame, True


def main() -> None:
    """Run pose estimation on a video and visualize or save the results."""
    parser = argparse.ArgumentParser(description="Test pose estimation on a video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--show", action="store_true", help="Display frames with skeleton overlay")
    parser.add_argument("--save", type=str, default=None, help="Save annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Disable temporal smoothing and process frames one by one",
    )
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
    quit_early = False

    if args.raw:
        print("Processing in raw (per-frame) mode...")
        for idx, _, frame in frame_iterator(
            args.video, max_dimension=config["video"]["max_dimension"]
        ):
            if args.max_frames and idx >= args.max_frames:
                break

            kp, conf = estimator.estimate_frame(frame)
            total += 1
            display_frame, found = _render_frame(frame, kp, conf, idx, skeleton_cfg)
            if found:
                detected += 1

            if writer:
                writer.write(display_frame)
            if args.show:
                cv2.imshow("ClimbVision - Pose Test", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    else:
        print("Processing with temporal smoothing...")
        print("  Pass 1/2: Reading frames...")
        frames: list[np.ndarray] = []
        frame_indices: list[int] = []

        for idx, _, frame in frame_iterator(
            args.video, max_dimension=config["video"]["max_dimension"]
        ):
            if args.max_frames and idx >= args.max_frames:
                break
            frame_indices.append(idx)
            frames.append(frame)
            if idx % 100 == 0:
                print(f"    Read {idx} frames...", end="\r")

        print(f"    Read {len(frames)} frames. Running pose estimation + smoothing...")
        keypoints_list, confidences = estimator.estimate_video(frames, fps=meta.fps)

        print("  Pass 2/2: Rendering overlays...")
        for idx, frame, kp, conf in zip(frame_indices, frames, keypoints_list, confidences):
            total += 1
            display_frame, found = _render_frame(frame, kp, conf, idx, skeleton_cfg)
            if found:
                detected += 1

            if writer:
                writer.write(display_frame)
            if args.show:
                cv2.imshow("ClimbVision - Pose Test", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_early = True
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
