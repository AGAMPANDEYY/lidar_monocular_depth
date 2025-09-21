import os
import shutil

# List of camera images for calibration
camera_files = [
    "07453.png", "04004.png", "04001.png", "03996.png", "03939.png", "03890.png", "03878.png", "03739.png", "03679.png", "03588.png", "03515.png", "03338.png", "03307.png", "03283.png", "03214.png", "03178.png", "03137.png", "03087.png", "03054.png"
]

CAMERA_SRC = "data/frames/front"
LIDAR_SRC = "data/lidar/"
CAMERA_DST = "matlab_camera"
LIDAR_DST = "matlab_lidar"

# FPS
CAMERA_FPS = 25
LIDAR_FPS = 10

os.makedirs(CAMERA_DST, exist_ok=True)
os.makedirs(LIDAR_DST, exist_ok=True)

def frame_num(filename):
    return int(filename.split('.')[0])

copied = 0
for cam_file in camera_files:
    cam_path = os.path.join(CAMERA_SRC, cam_file)
    cam_idx = frame_num(cam_file)
    # Find closest lidar frame index
    lidar_idx = int(round(cam_idx * (LIDAR_FPS / CAMERA_FPS)))
    lidar_file = f"input (Frame {lidar_idx}).pcd"
    lidar_path = os.path.join(LIDAR_SRC, lidar_file)
    missing = []
    if not os.path.exists(cam_path):
        missing.append(f"Camera image missing: {cam_file}")
    if not os.path.exists(lidar_path):
        missing.append(f"LiDAR file missing: {lidar_file} (for camera {cam_file})")
    if missing:
        print("[DEBUG]", " | ".join(missing))
    else:
        shutil.copy(cam_path, os.path.join(CAMERA_DST, cam_file))
        shutil.copy(lidar_path, os.path.join(LIDAR_DST, lidar_file))
        copied += 1

print(f"Copied {copied} calibration image-lidar pairs to MATLAB folders.")
