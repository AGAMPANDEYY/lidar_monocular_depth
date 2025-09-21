# LiDAR-Monocular Depth Fusion Project

This project fuses LiDAR point clouds with monocular depth estimation for robust depth perception.

## Setup Instructions

1. **Create and Activate Virtual Environment**
```bash
# Create virtual environment
python -m venv .venv

# Activate on macOS/Linux
source .venv/bin/activate
```

2. **Install Dependencies**
```bash
pip install numpy opencv-python pillow pandas open3d ultralytics torch torchvision
```

## Pipeline Steps

### 1. Extract Frames
```bash
python scripts/extract_frames.py --video data/input.avi --max_frames 1000
```
- Frames will be saved in `data/frames/front/`
- For first 3 minutes at 30fps, use `--max_frames 5400`

### 2. Calibration
> **Note**: You only need to perform calibration ONCE if the camera and LiDAR setup remains fixed.

The current calibration files are:
- Camera intrinsics: `calibration/camera.yaml`
- LiDAR-to-Camera extrinsics: `calibration/extrinsics_lidar_to_cam.yaml`

If you need to recalibrate:
1. Collect corresponding 3D-2D point pairs
2. Run calibration:
```bash
python calibration/calibrate_extrinsics.py \
  --cam_yaml calibration/camera.yaml \
  --img data/frames/front/07543.png \
  --pts3d_csv calibration/lidar_points3d.csv \
  --pts2d_csv calibration/image_points2d.csv \
  --out_yaml calibration/extrinsics_lidar_to_cam.yaml
```

### 3. Project LiDAR Points
Process each LiDAR PCD file corresponding to your frames:
```bash
python lidar_projection/project_lidar.py \
  --pcd "data/lidar/input (Frame XXXX).pcd" \
  --cam_yaml calibration/camera.yaml \
  --ext_yaml calibration/extrinsics_lidar_to_cam.yaml \
  --image data/frames/front/XXXX.png \
  --out_npz data/processed_lidar/XXXX.npz \
  --debug_overlay data/processed_lidar/XXXX_overlay.png \
  --manual_cy_offset 80 \
  --manual_cx_offset 30
```

For batch processing multiple frames, create a script that:
1. Lists all PCD files in `data/lidar/`
2. Matches frame numbers between PCDs and images
3. Runs projection for each pair

### 4. Run Main Processing Pipeline
```bash
python main.py
```

The main pipeline will:
- Load frame images from `data/frames/test/`
- Load corresponding processed LiDAR data from `data/processed_lidar/`
- Perform object detection (YOLOv8)
- Generate monocular depth (MiDaS)
- Fuse LiDAR and monocular depth
- Compute TTC (Time-to-Collision) and ECW (Emergency Collision Warning)
- Save visualizations in `data/fused_output/`

## Project Structure
```
lidar_monocular_depth/
├── calibration/              # Calibration scripts and data
├── data/
│   ├── frames/              # Extracted video frames
│   ├── lidar/               # Raw LiDAR PCD files
│   ├── processed_lidar/     # Projected LiDAR depth maps
│   └── fused_output/        # Final visualization outputs
├── modules/                  # Core processing modules
├── scripts/                  # Utility scripts
└── main.py                  # Main processing pipeline
```

## Important Notes

1. **Calibration Reuse**:
   - If your camera-LiDAR setup is fixed, you DON'T need to recalibrate
   - The current extrinsics in `calibration/extrinsics_lidar_to_cam.yaml` are valid for all frames
   - Only recalibrate if you change the physical setup or notice significant misalignment

2. **LiDAR Projection**:
   - Current offset values (`cy_offset=80`, `cx_offset=30`) work for the current setup
   - You might need to adjust these if you recalibrate or change the setup

3. **Processing Multiple Frames**:
   - Ensure frame numbers in filenames match between LiDAR PCDs and images
   - The main.py script will automatically match corresponding files