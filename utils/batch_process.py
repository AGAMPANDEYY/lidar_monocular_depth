import os
import glob
import re
import frame_sync
from frame_sync import find_closest_frames
import subprocess

def extract_frame_number(filename, pattern):
    """Extract frame number from filename using regex pattern"""
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def process_frames(lidar_dir, camera_dir, output_dir, 
                  cam_yaml, ext_yaml, lidar_fps=10, camera_fps=25):
    """
    Process multiple frames with proper LiDAR-camera synchronization
    
    Args:
        lidar_dir: Directory containing LiDAR PCD files
        camera_dir: Directory containing camera image files
        output_dir: Directory for output files
        cam_yaml: Path to camera calibration YAML
        ext_yaml: Path to extrinsics YAML
    """
    print(f"Looking for files in:")
    print(f"LiDAR directory: {lidar_dir}")
    print(f"Camera directory: {camera_dir}")
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all LiDAR and camera files
    lidar_files = glob.glob(os.path.join(lidar_dir, "*.pcd"))
    camera_files = glob.glob(os.path.join(camera_dir, "*.png"))
    
    # Extract frame numbers
    lidar_frames = [extract_frame_number(f, r"Frame (\d+)\.pcd") for f in lidar_files]
    camera_frames = [extract_frame_number(f, r"(\d+)\.png") for f in camera_files]
    
    # Remove None values
    lidar_frames = [f for f in lidar_frames if f is not None]
    camera_frames = [f for f in camera_frames if f is not None]
    
    # Find matching frames
    mapping, time_diffs = find_closest_frames(lidar_frames, camera_frames, 
                                            lidar_fps=lidar_fps, camera_fps=camera_fps)
    
    # Save and print matches
    matches_dir = os.path.join(output_dir, "frame_matches")
    os.makedirs(matches_dir, exist_ok=True)
    df = frame_sync.save_frame_matches(mapping, time_diffs, matches_dir)
    
    # Process each matched pair
    for lidar_frame, camera_frame in mapping.items():
        time_diff_ms = time_diffs[lidar_frame] * 1000
        print(f"{lidar_frame:>12} | {camera_frame:>12} | {time_diff_ms:>11.2f} ms")
        
        # Construct file paths
        lidar_file = os.path.join(lidar_dir, f"input (Frame {lidar_frame}).pcd")
        camera_file = os.path.join(camera_dir, f"{camera_frame:05d}.png")
        
        # Skip if files don't exist
        if not os.path.exists(lidar_file) or not os.path.exists(camera_file):
            print(f"Warning: Missing files for LiDAR frame {lidar_frame} or camera frame {camera_frame}")
            continue
        
        # Construct output paths
        out_npz = os.path.join(output_dir, f"{lidar_frame:05d}.npz")
        out_overlay = os.path.join(output_dir, f"{lidar_frame:05d}_overlay.png")
        
        # Construct command
        cmd = [
            "python", "lidar_projection/project_lidar.py",
            "--pcd", lidar_file,
            "--cam_yaml", cam_yaml,
            "--ext_yaml", ext_yaml,
            "--image", camera_file,
            "--out_npz", out_npz,
            "--debug_overlay", out_overlay,
            "--manual_cx_offset", "30",
            "--splat_radius", "0"
        ]
        
        # Run projection
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed frame pair: LiDAR {lidar_frame} - Camera {camera_frame}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing frame pair: LiDAR {lidar_frame} - Camera {camera_frame}")
            print(f"Error: {str(e)}")

def main():
    # Example usage
    lidar_dir = "data/lidar"
    camera_dir = "data/frames/front"
    output_dir = "data/processed_lidar"
    cam_yaml = "calibration/camera.yaml"
    ext_yaml = "calibration/extrinsics_lidar_to_cam.yaml"
    
    process_frames(lidar_dir, camera_dir, output_dir, cam_yaml, ext_yaml)

if __name__ == "__main__":
    main()