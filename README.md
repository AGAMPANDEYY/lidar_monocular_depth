# LiDAR-Monocular Depth Fusion

This project fuses LiDAR point clouds with monocular depth estimation for robust obstacle detection and depth measurement. It combines the sparse but accurate measurements from LiDAR with dense but relative depth from neural networks.

## Features

- Calibrated fusion of LiDAR and neural monocular depth (MiDaS)
- YOLOv8 object detection with per-object depth metrics
- Temporal accumulation of sparse LiDAR data
- RANSAC-based ground plane removal
- Time-to-collision (TTC) estimation
- Hazard detection using LiDAR point clusters
- Efficient camera-LiDAR frame synchronization
- Rich visualizations and debug outputs

## Requirements

- Python 3.8+
- PyTorch (for MiDaS and YOLO)
- OpenCV
- NumPy
- Pandas
- PIL

## Project Structure

```
lidar_monocular_depth/
├── data/
│   ├── frames/              # Camera frames
│   ├── processed_lidar/     # Projected LiDAR data
│   └── fused_output/        # Visualization outputs
├── modules/
│   ├── detection.py         # YOLOv8 object detection
│   ├── depth.py            # MiDaS depth estimation
│   ├── geo.py              # Geometric operations
│   ├── hazard.py           # Hazard detection
│   ├── metrics.py          # TTC and depth stats
│   ├── sync.py            # Frame synchronization
│   └── visualization.py    # Result visualization
├── detection/
│   └── best-e150 (1).pt   # YOLO weights
├── calibration/
│   └── camera.yaml        # Camera parameters
└── main.py                # Pipeline orchestration
```

## Configuration

Key parameters in `main.py`:

```python
# Testing configuration
MAX_TEST_FRAMES = 15     # Maximum frames to process
CAMERA_FPS = 25.0       # Camera frame rate
LIDAR_FPS = 10.0        # LiDAR frame rate
CAMERA_START = 15000    # First camera frame
LIDAR_START = 6000      # First LiDAR frame

# Research configuration
ACCUM_WINDOW = 2        # LiDAR accumulation window
GROUND_REMOVE = True    # RANSAC ground removal
GROUND_HEIGHT = 0.10    # Min height above ground (m)
ECW_Z_THRESH = 8.0      # Bubble depth threshold (m)
```

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision opencv-python pillow pandas numpy ultralytics
   ```

## Usage

1. Organize data:
   ```
   data/
   ├── frames/
   │   └── front/          # Front-view camera frames
   │       ├── 15000.png
   │       ├── 15001.png
   │       └── ...
   └── processed_lidar/    # Projected LiDAR frames
       ├── 6000.npz
       ├── 6001.npz
       └── ...
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. Check outputs in `data/fused_output/`:
   - `*_overlay.png`: Visualizations with detection boxes
   - `debug/`: Individual depth maps and masks
   - `object_depth_metrics.csv`: Per-object measurements
   - `output_visualization.mp4`: Final video output

## Frame Synchronization

The pipeline handles different frame rates:
- Camera: 25 FPS (frames 15000-16499)
- LiDAR: 10 FPS (frames 6000-6599)

Frames are matched using temporal alignment, finding the closest LiDAR frame for each camera frame while tracking timing statistics.

## Output Format

The `object_depth_metrics.csv` contains per-object measurements:
- Frame number
- Object class and confidence
- Bounding box coordinates
- LiDAR and monocular depth estimates
- Time-to-collision (TTC)
- ECW bubble status
- Frame synchronization timing

## Important Notes

1. **Calibration**:
   - Camera calibration parameters must be in `calibration/camera.yaml`
   - Current parameters work for the test sequences

2. **Testing Mode**:
   - Set `MAX_TEST_FRAMES = 15` for quick testing
   - Set to 0 or a large number for full sequence processing

3. **Ground Removal**:
   - RANSAC-based ground plane removal can be disabled with `GROUND_REMOVE = False`
   - Adjust `GROUND_HEIGHT` for different thresholds above detected plane

4. **LiDAR Accumulation**:
   - Set `ACCUM_WINDOW = 0` to disable temporal accumulation
   - Larger windows help with sparse LiDAR data but may introduce lag

## License

TBD (update with chosen license)