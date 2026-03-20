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
from src.core.pipeline_runner import (
    compute_adaptive_smoothing_window,
    run_pose_pass,
)
from src.core.pose_estimator import PoseEstimator
from src.analysis.joint_angles import compute_joint_angles
from src.visualization.skeleton_overlay import render_skeleton
from src.visualization.angle_overlay import render_angles
from src.visualization.hud_overlay import render_hud


def _render_frame(
    frame: np.ndarray,
    kp: np.ndarray | None,
    conf: float,
    idx: int,
    timestamp: float,
    skeleton_cfg: dict,
    angles_cfg: dict,
    hud_cfg: dict,
    joint_angles: dict[str, float] | None = None,
) -> tuple[np.ndarray, bool]:
    """Render skeleton + angle arc overlays on a single frame.

    Args:
        frame: BGR image.
        kp: Keypoints array (33, 4) or None if no pose detected.
        conf: Overall confidence float.
        idx: Frame index for progress logging.
        timestamp: Frame timestamp in seconds.
        skeleton_cfg: Skeleton overlay config dict.
        angles_cfg: Angle overlay config dict.
        hud_cfg: HUD overlay config dict.
        joint_angles: Precomputed joint angles when available.

    Returns:
        Tuple of (annotated frame, whether pose was detected).
    """
    display_frame = frame.copy()

    if kp is None:
        display_frame = render_hud(display_frame, idx, timestamp, None, hud_cfg, angles_cfg)
        if idx % 30 == 0:
            print(f"  Frame {idx:4d} | NO POSE DETECTED")
        return display_frame, False

    if joint_angles is None:
        joint_angles = compute_joint_angles(kp)

    display_frame = render_skeleton(display_frame, kp, skeleton_cfg)
    display_frame = render_angles(display_frame, kp, joint_angles, angles_cfg)
    display_frame = render_hud(display_frame, idx, timestamp, joint_angles, hud_cfg, angles_cfg)

    if idx % 30 == 0:
        angle_summary = ", ".join(
            f"{k}={v:.0f}" for k, v in joint_angles.items()
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
    angles_cfg = config["overlay"]["angles"]
    hud_cfg = config["overlay"]["hud"]
    smoothing_window = compute_adaptive_smoothing_window(meta.fps, config)

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
            display_frame, found = _render_frame(
                frame,
                kp,
                conf,
                idx,
                idx / meta.fps,
                skeleton_cfg,
                angles_cfg,
                hud_cfg,
            )
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
        effective_total_frames = args.max_frames if args.max_frames else meta.total_frames
        use_chunked = effective_total_frames >= 2000 and args.max_frames == 0

        if use_chunked:
            print(f"  Using chunked Pass 1 ({smoothing_window=})...")
            pose_frames = run_pose_pass(
                video_path=args.video,
                config=config,
                estimator=estimator,
                fps=meta.fps,
                smoothing_window=smoothing_window,
                total_frames=meta.total_frames,
                max_dimension=config["video"]["max_dimension"],
            )

            print("  Pass 2/2: Rendering overlays...")
            for idx, timestamp, frame in frame_iterator(
                args.video, max_dimension=config["video"]["max_dimension"]
            ):
                if args.max_frames and idx >= args.max_frames:
                    break

                frame_data = pose_frames.get(idx)
                kp = frame_data.keypoints if frame_data is not None else None
                conf = frame_data.keypoint_confidence if frame_data is not None else 0.0
                joint_angles = frame_data.joint_angles if frame_data is not None else None

                total += 1
                display_frame, found = _render_frame(
                    frame,
                    kp,
                    conf,
                    idx,
                    timestamp,
                    skeleton_cfg,
                    angles_cfg,
                    hud_cfg,
                    joint_angles=joint_angles,
                )
                if found:
                    detected += 1

                if writer:
                    writer.write(display_frame)
                if args.show:
                    cv2.imshow("ClimbVision - Pose Test", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        quit_early = True
                        break
        else:
            print(f"  Using in-memory smoothing path ({smoothing_window=})...")
            print("  Pass 1/2: Reading frames...")
            frames: list[np.ndarray] = []
            frame_indices: list[int] = []
            timestamps: list[float] = []

            for idx, timestamp, frame in frame_iterator(
                args.video, max_dimension=config["video"]["max_dimension"]
            ):
                if args.max_frames and idx >= args.max_frames:
                    break
                frame_indices.append(idx)
                timestamps.append(timestamp)
                frames.append(frame)
                if idx % 100 == 0:
                    print(f"    Read {idx} frames...", end="\r")

            print(f"    Read {len(frames)} frames. Running pose estimation + smoothing...")
            raw_keypoints, confidences = estimator.detect_video_raw(frames, fps=meta.fps)
            keypoints_list = estimator.postprocess_keypoints(
                raw_keypoints, smoothing_window=smoothing_window,
            )

            print("  Pass 2/2: Rendering overlays...")
            for idx, timestamp, frame, kp, conf in zip(
                frame_indices,
                timestamps,
                frames,
                keypoints_list,
                confidences,
            ):
                total += 1
                display_frame, found = _render_frame(
                    frame,
                    kp,
                    conf,
                    idx,
                    timestamp,
                    skeleton_cfg,
                    angles_cfg,
                    hud_cfg,
                )
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
