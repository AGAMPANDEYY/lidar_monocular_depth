#!/usr/bin/env python3
import os
import subprocess
import argparse

def time_to_frame(minutes, fps):
    """Convert time in minutes to frame number"""
    return int(minutes * 60 * fps)

def main():
    parser = argparse.ArgumentParser(description="Process specific time range of video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--start_min", type=float, default=10, help="Start time in minutes")
    parser.add_argument("--end_min", type=float, default=13, help="End time in minutes")
    parser.add_argument("--camera_fps", type=int, default=25, help="Camera FPS")
    parser.add_argument("--lidar_fps", type=int, default=10, help="LiDAR FPS")
    parser.add_argument("--cy_offset", type=str, default="80", help="Manual cy offset for LiDAR projection")
    parser.add_argument("--cx_offset", type=str, default="30", help="Manual cx offset for LiDAR projection")
    args = parser.parse_args()

    # Calculate frame ranges
    cam_start = time_to_frame(args.start_min, args.camera_fps)
    cam_end = time_to_frame(args.end_min, args.camera_fps)
    lidar_start = time_to_frame(args.start_min, args.lidar_fps)
    lidar_end = time_to_frame(args.end_min, args.lidar_fps)

    print(f"\nProcessing time range: {args.start_min}-{args.end_min} minutes")
    print(f"Camera frames: {cam_start}-{cam_end} ({cam_end-cam_start} frames)")
    print(f"LiDAR frames: {lidar_start}-{lidar_end} ({lidar_end-lidar_start} frames)")

    # 1. Extract video frames
    print("\n1. Extracting video frames...")
    cmd = [
        "python", "scripts/extract_frames.py",
        "--video", args.video,
        "--output_dir", "data/frames",
        "--start_frame", str(cam_start),
        "--end_frame", str(cam_end)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2. Process LiDAR data
    print("\n2. Processing LiDAR data...")
    cmd = [
        "python", "scripts/project_lidars.py",
        "--lidar_dir", "data/lidar",
        "--image_dir", "data/frames/front",
        "--output_dir", "data/processed_lidar",
        "--start_frame", str(lidar_start),
        "--end_frame", str(lidar_end),
        "--cam_yaml", "calibration/camera.yaml",
        "--ext_yaml", "calibration/extrinsics_lidar_to_cam.yaml",
        "--cy_offset", args.cy_offset,
        "--cx_offset", args.cx_offset
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 3. Run main processing pipeline
    print("\n3. Running main processing pipeline...")
    cmd = ["python", "main.py"]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print("\nProcessing complete!")
    print("- Check data/fused_output/output_visualization.mp4 for the final video")
    print("- Check data/fused_output/debug/ for individual frame visualizations")

if __name__ == "__main__":
    main()