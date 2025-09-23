#!/usr/bin/env python3
import os
import subprocess
import argparse
import cv2

def get_video_duration(video_path):
    """Get video duration in minutes"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate duration in minutes
    duration_minutes = (total_frames / fps) / 60
    
    cap.release()
    return duration_minutes

def time_to_frame(minutes, fps):
    """Convert time in minutes to frame number"""
    return int(minutes * 60 * fps)

def main():
    parser = argparse.ArgumentParser(description="Process specific time range of video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--start_min", type=float, default=10, help="Start time in minutes")
    parser.add_argument("--end_min", type=float, help="End time in minutes (optional, defaults to video end)")
    parser.add_argument("--camera_fps", type=int, default=25, help="Camera FPS")
    parser.add_argument("--lidar_fps", type=int, default=10, help="LiDAR FPS")
    parser.add_argument("--cy_offset", type=str, default="80", help="Manual cy offset for LiDAR projection")
    parser.add_argument("--cx_offset", type=str, default="30", help="Manual cx offset for LiDAR projection")
    parser.add_argument("--rois_yaml", default="scripts/config/default_rois.yaml",
                      help="Path to ROIs YAML file (defaults to scripts/config/default_rois.yaml)")
    args = parser.parse_args()
    
    # If end_min not provided, use video duration
    if args.end_min is None:
        args.end_min = get_video_duration(args.video)
        print(f"No end time provided, using video duration: {args.end_min:.2f} minutes")
    
    # Generate default ROIs if they don't exist
    if not os.path.exists(args.rois_yaml):
        print(f"\nGenerating default ROIs configuration...")
        cmd = ["python", "scripts/generate_rois_yaml.py",
               "--output_yaml", args.rois_yaml]
        subprocess.run(cmd, check=True)

    # Calculate frame ranges based on video duration
    if args.end_min is None:
        video_duration = get_video_duration(args.video)
        print(f"Video duration detected: {video_duration:.2f} minutes")
        args.end_min = video_duration
    
    # Validate start time is within video duration
    if args.start_min >= args.end_min:
        print(f"Error: Start time ({args.start_min} min) must be less than end time ({args.end_min} min)")
        return 1
    
    # Calculate frame ranges
    cam_start = time_to_frame(args.start_min, args.camera_fps)
    cam_end = time_to_frame(args.end_min, args.camera_fps)
    lidar_start = time_to_frame(args.start_min, args.lidar_fps)
    lidar_end = time_to_frame(args.end_min, args.lidar_fps)

    print(f"\nProcessing time range: {args.start_min}-{args.end_min} minutes")
    print(f"Camera frames: {cam_start}-{cam_end} ({cam_end-cam_start} frames)")
    print(f"LiDAR frames: {lidar_start}-{lidar_end} ({lidar_end-lidar_start} frames)")

    # Ensure output directories exist
    os.makedirs("data/frames", exist_ok=True)
    
    # 1. Extract video frames
    print("\n1. Extracting video frames...")
    cmd = [
        "python3", "scripts/extract_frames.py",
        "--video", args.video,
        "--output_dir", "data/frames",
        "--start_frame", str(cam_start),
        "--end_frame", str(cam_end),
        "--camera_yaml", "calibration/camera.yaml",  # Always use camera calibration
        "--rois_yaml", "scripts/config/default_rois.yaml"  # Always use default ROIs
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 2. Process LiDAR data
    print("\n2. Processing LiDAR data...")
    cmd = [
        "python3", "scripts/project_lidars.py",
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

    print("\nProcessing complete!")
    print(f"- Camera frames extracted to: data/frames/")
    print(f"- LiDAR data processed to: data/processed_lidar/")
    print(f"- Frame range: {cam_start}-{cam_end}")
    print(f"- LiDAR range: {lidar_start}-{lidar_end}")

if __name__ == "__main__":
    main()