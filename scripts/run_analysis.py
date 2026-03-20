"""Run the ClimbVision two-pass video analysis pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from src.analysis.joint_angles import compute_joint_angles, AngleTracker, JOINT_TRIPLETS
from src.core.pipeline_runner import (
    compute_adaptive_smoothing_window,
    run_pose_pass,
    make_progress,
)
from src.core.pose_estimator import PoseEstimator
from src.core.video_io import VideoWriter, frame_iterator, load_config, load_video
from src.visualization.skeleton_overlay import (
    compute_stress_colors,
    render_skeleton,
    render_dynamic_hud,
)


def main() -> None:
    """Run Pass 1 pose extraction and Pass 2 video rendering."""
    args = _parse_args()
    config = load_config(args.config)
    metadata = load_video(args.video)

    output_path = _resolve_output_path(args.video, args.output)
    smoothing_window = compute_adaptive_smoothing_window(metadata.fps, config)
    smoothing_source = (
        "manual override"
        if config["pose"].get("smoothing_window") is not None
        else "adaptive"
    )

    output_width, output_height = _scaled_dimensions(
        metadata.width,
        metadata.height,
        config["video"].get("max_dimension"),
    )
    output_fps = config["video"].get("output_fps") or metadata.fps

    print(f"Video: {metadata.path}")
    print(
        f"  Dimensions: {metadata.width}x{metadata.height} "
        f"(output {output_width}x{output_height})"
    )
    print(f"  FPS: {metadata.fps:.2f} | Reported frames: {metadata.total_frames}")
    print(f"  Smoothing window: {smoothing_window} ({smoothing_source})")
    print(f"  Output: {output_path}")

    start_time = time.perf_counter()
    writer: VideoWriter | None = None
    estimator: PoseEstimator | None = None

    # Initialize stateful tracker for dynamic HUD events
    angles_cfg = config.get("overlay", {}).get("angles", {})
    angle_tracker = AngleTracker(
        history_length=angles_cfg.get("tracker_history_length", 15),
        range_threshold=angles_cfg.get("tracker_range_threshold", 10.0),
        stress_threshold=angles_cfg.get("tracker_stress_threshold", 90.0),
    )

    try:
        estimator = PoseEstimator(config["pose"])
        pose_frames = run_pose_pass(
            video_path=args.video,
            config=config,
            estimator=estimator,
            fps=metadata.fps,
            smoothing_window=smoothing_window,
            total_frames=metadata.total_frames,
            max_dimension=config["video"].get("max_dimension"),
        )

        writer = VideoWriter(
            output_path,
            fps=output_fps,
            width=output_width,
            height=output_height,
            codec=config["video"]["output_codec"],
        )

        skeleton_config = config.get("overlay", {}).get("skeleton", {})
        angles_config = config.get("overlay", {}).get("angles", {})

        rendered_frames = 0
        detected_frames = 0
        total_confidence = 0.0
        progress = make_progress(metadata.total_frames, "Pass 2")

        try:
            for frame_idx, timestamp, frame in frame_iterator(
                args.video,
                max_dimension=config["video"].get("max_dimension"),
            ):
                frame_data = pose_frames.get(frame_idx)
                keypoints = frame_data.keypoints if frame_data is not None else None
                confidence = (
                    frame_data.keypoint_confidence if frame_data is not None else 0.0
                )
                joint_angles = (
                    frame_data.joint_angles if frame_data is not None else None
                )

                annotated = frame.copy()
                if keypoints is not None:
                    if joint_angles is None:
                        try:
                            joint_angles = compute_joint_angles(
                                keypoints, min_visibility=0.3,
                            )
                        except Exception:
                            joint_angles = None

                    if joint_angles:
                        events = angle_tracker.update(joint_angles)
                        stress_colors = compute_stress_colors(
                            joint_angles, JOINT_TRIPLETS, angles_config,
                        )
                        annotated = render_skeleton(annotated, keypoints, skeleton_config, stress_colors=stress_colors)
                        annotated = render_dynamic_hud(annotated, events)
                    else:
                        annotated = render_skeleton(annotated, keypoints, skeleton_config)
                
                writer.write(annotated)

                rendered_frames += 1
                total_confidence += confidence
                if keypoints is not None:
                    detected_frames += 1
                progress.update(1)
        finally:
            progress.close()

        elapsed = time.perf_counter() - start_time
        detection_rate = (detected_frames / rendered_frames) if rendered_frames else 0.0
        average_confidence = (
            total_confidence / rendered_frames if rendered_frames else 0.0
        )

        print("Summary:")
        print(f"  Total frames processed: {rendered_frames}")
        print(f"  Detection rate: {detection_rate:.1%}")
        print(f"  Average confidence: {average_confidence:.3f}")
        print(f"  Processing time: {elapsed:.2f}s")
    finally:
        if writer is not None:
            writer.release()
        if estimator is not None:
            estimator.close()


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ClimbVision climbing video analysis pipeline."
    )
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output video path or directory for the annotated result.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the pipeline config file.",
    )
    return parser.parse_args()


def _resolve_output_path(video_path: str, output_arg: str) -> str:
    """Resolve the final annotated video output path."""
    output_path = Path(output_arg)
    if output_path.suffix.lower() in {".mp4", ".avi", ".mov"}:
        return str(output_path)

    video_stem = Path(video_path).stem
    return str(output_path / f"{video_stem}_annotated.mp4")


def _scaled_dimensions(
    width: int,
    height: int,
    max_dimension: int | None,
) -> tuple[int, int]:
    """Compute output frame dimensions after optional downscaling."""
    if max_dimension is None or max(width, height) <= max_dimension:
        return width, height

    scale = max_dimension / max(width, height)
    return int(width * scale), int(height * scale)


if __name__ == "__main__":
    main()
