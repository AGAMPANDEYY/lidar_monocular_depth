import os
import numpy as np
import pandas as pd
import csv
from datetime import datetime

def get_frame_timestamp(frame_number, fps):
    """Convert frame number to timestamp in seconds"""
    return frame_number / fps

def find_closest_frames(lidar_frames, camera_frames, lidar_fps=10, camera_fps=25):
    """
    Find the closest matching camera frame for each LiDAR frame
    
    Args:
        lidar_frames: List of LiDAR frame numbers
        camera_frames: List of camera frame numbers
        lidar_fps: LiDAR frame rate (default: 10 FPS)
        camera_fps: Camera frame rate (default: 25 FPS)
        
    Returns:
        dict: Mapping of LiDAR frame numbers to closest camera frame numbers
        dict: Mapping of LiDAR frame numbers to time differences (in seconds)
    """
    mapping = {}
    time_diffs = {}
    
    # Convert frame numbers to timestamps
    lidar_times = {frame: get_frame_timestamp(frame, lidar_fps) for frame in lidar_frames}
    camera_times = {frame: get_frame_timestamp(frame, camera_fps) for frame in camera_frames}
    
    # Find closest camera frame for each LiDAR frame
    for lidar_frame in lidar_frames:
        lidar_time = lidar_times[lidar_frame]
        
        # Calculate time differences with all camera frames
        time_differences = {cam_frame: abs(cam_time - lidar_time) 
                          for cam_frame, cam_time in camera_times.items()}
        
        # Find camera frame with minimum time difference
        closest_camera_frame = min(time_differences.keys(), 
                                 key=lambda x: time_differences[x])
        
        mapping[lidar_frame] = closest_camera_frame
        time_diffs[lidar_frame] = time_differences[closest_camera_frame]
    
    return mapping, time_diffs

def save_frame_matches(mapping, time_diffs, output_dir="."):
    """Save frame matching information to CSV and print detailed results"""
    # Prepare data for DataFrame
    data = []
    for lidar_frame, camera_frame in sorted(mapping.items()):
        time_diff_ms = time_diffs[lidar_frame] * 1000
        lidar_time = get_frame_timestamp(lidar_frame, 10)  # 10 FPS for LiDAR
        camera_time = get_frame_timestamp(camera_frame, 25)  # 25 FPS for camera
        
        data.append({
            'lidar_frame': lidar_frame,
            'camera_frame': camera_frame,
            'lidar_timestamp': f"{lidar_time:.3f}",
            'camera_timestamp': f"{camera_time:.3f}",
            'time_diff_ms': f"{time_diff_ms:.2f}",
            'frame_diff': camera_frame - lidar_frame
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"frame_matching_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Print detailed results
    print("\nDetailed Frame Matching Results:")
    print("-" * 100)
    print(f"{'LiDAR Frame':>12} | {'Camera Frame':>12} | {'Time Diff (ms)':>14} | {'LiDAR Time':>12} | {'Camera Time':>12} | {'Frame Diff':>10}")
    print("-" * 100)
    
    for row in data:
        print(f"{row['lidar_frame']:>12} | {row['camera_frame']:>12} | {row['time_diff_ms']:>11} ms | "
              f"{row['lidar_timestamp']:>12} | {row['camera_timestamp']:>12} | {row['frame_diff']:>10}")
    
    print(f"\nCSV file saved to: {csv_path}")
    return df

def main():
    """Example usage"""
    # Create output directory for CSV files
    output_dir = "data/frame_matching"
    os.makedirs(output_dir, exist_ok=True)
    
    # Full dataset
    # LiDAR: 601 frames at 10 FPS
    # Camera: 1500 frames at 25 FPS
    lidar_frames = list(range(6000, 6600))  # 600 LiDAR frames
    camera_frames = list(range(15000, 16500))  # 1500 camera frames
    
    # Get frame matches
    mapping, time_diffs = find_closest_frames(lidar_frames, camera_frames)
    
    # Save and print detailed matches
    df = save_frame_matches(mapping, time_diffs, output_dir)
    
    # Print statistics
    time_diffs_ms = np.array(list(time_diffs.values())) * 1000
    print(f"\nMatching Statistics:")
    print(f"Average time difference: {np.mean(time_diffs_ms):.2f} ms")
    print(f"Maximum time difference: {np.max(time_diffs_ms):.2f} ms")
    print(f"Minimum time difference: {np.min(time_diffs_ms):.2f} ms")
    
    # Additional statistics
    print(f"\nFrame Range Statistics:")
    print(f"LiDAR frames: {min(lidar_frames)} to {max(lidar_frames)} ({len(lidar_frames)} frames)")
    print(f"Camera frames: {min(camera_frames)} to {max(camera_frames)} ({len(camera_frames)} frames)")
    print(f"Average frame difference: {df['frame_diff'].mean():.2f} frames")

if __name__ == "__main__":
    main()