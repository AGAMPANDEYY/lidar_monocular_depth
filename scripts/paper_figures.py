#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
import subprocess
from scipy.interpolate import griddata

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from modules.detection import load_yolo_model, run_obstacle_detection, CLASSES
from modules.depth import load_depth_backend
from modules.visualization import overlay_results
from modules.metrics import compute_ecw_bubble, TTCTracker, robust_box_depth
from main import (make_ecw_mask, paint_depth_background,  # Reuse ECW utilities
                 T_WARN_VRU, T_WARN_VEHICLE, T_WARN_DEFAULT, T_HYSTERESIS,  # TTC thresholds
                 MIN_VALID_PIXELS, MIN_PERSISTENCE_FRAMES,  # Warning parameters
                 MIN_SIZE_M, MAX_SIZE_M)  # Object size thresholds

def create_visualization_figures(frame_number):
    """Create visualization figures for the given frame number."""
    # Create output directory
    figures_dir = os.path.join(project_root, "data", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\nCreating figures in: {figures_dir}")
    
    # Load RGB image
    frame_path = os.path.join(figures_dir, f"frame_{frame_number}.png")
    if os.path.exists(frame_path):
        frame_rgb = np.array(Image.open(frame_path))
    else:
        raise FileNotFoundError(f"Frame not found: {frame_path}")
    
    # Get frame dimensions
    H, W = frame_rgb.shape[:2]
    
    # Load models
    print("\nLoading models...")
    yolo_model = load_yolo_model(os.path.join(project_root, "detection/best-e150 (1).pt"))
    run_depth, device, depth_name = load_depth_backend('midas')
    print(f"Loaded YOLO and {depth_name} on {device}")
    
    # Run object detection
    print("\nRunning object detection...")
    annotations_path = os.path.join(project_root, 'data', 'manual_annotations', f'frame_{frame_number}.json')
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        pred_bboxes = data['boxes']
        print("Using manual annotations")
    else:
        pred_bboxes = run_obstacle_detection(frame_rgb, yolo_model)
        print("Using YOLO detections")
    
    # Run monocular depth estimation
    print("\nRunning monocular depth estimation...")
    depth_map = run_depth(frame_rgb)
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Load LiDAR ground truth
    print("\nLoading LiDAR data...")
    lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
    if not os.path.exists(lidar_npz):
        print(f"Warning: LiDAR data not found at {lidar_npz}")
        lidar_points = None
        lidar_depths = None
    else:
        lidar_data = np.load(lidar_npz, allow_pickle=True)
        data_dict = dict(lidar_data)
        
        if 'projected_points' in data_dict:
            points_2d = data_dict['projected_points']
            depths = np.linalg.norm(data_dict.get('points_3d', points_2d), axis=1)
        elif 'points_uv' in data_dict:
            points_2d = data_dict['points_uv']
            depths = data_dict.get('depths', np.linalg.norm(data_dict.get('points_xyz', points_2d), axis=1))
        else:
            points_2d = None
            depths = None
            print("Warning: Could not find projected points in LiDAR data")
        
        if points_2d is not None and depths is not None:
            # Interpolate LiDAR points to full image
            grid_x, grid_y = np.mgrid[0:H:1, 0:W:1]
            lidar_depth = griddata(points_2d, depths, (grid_x, grid_y), method='linear')
            print("Created interpolated LiDAR depth map")
        else:
            lidar_depth = None
            print("Warning: Could not create LiDAR depth map")
    
    # Load fused depth if available
    print("\nLoading fused depth...")
    fused_path = os.path.join(project_root, "data/fused_output/debug", f"{frame_number}_fused_depth.npy")
    if os.path.exists(fused_path):
        fused_depth = np.load(fused_path)
        print("Loaded fused depth")
    else:
        fused_depth = None
        print("Warning: Fused depth not found")
    
    # Create combined visualization with ECW bubble
    print("\nCreating combined visualization...")
    plt.figure(figsize=(16, 4))
    
    # 1. Original RGB with boxes
    plt.subplot(141)
    plt.imshow(frame_rgb)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Color based on class
        if cls_idx == 1:  # Person
            color = 'green'
        elif cls_idx == 3:  # Motorcycle/bicycle
            color = 'cyan'
        else:  # Vehicles
            color = 'magenta'
        
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        plt.text(x1, y1-10, f"{CLASSES[cls_idx]}", color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.title('RGB with Detections')
    plt.axis('off')
    
    # 2. Monocular depth
    plt.subplot(142)
    plt.imshow(frame_rgb)
    depth_vis = np.ma.masked_array(depth_map)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        box_depth = robust_box_depth(depth_map[y1:y2, x1:x2])
        depth_roi = depth_map[y1:y2, x1:x2]
        plt.imshow(depth_roi, extent=[x1, x2, y2, y1], cmap='magma', alpha=0.7)
        plt.text(x1, y1-10, f"D:{box_depth:.1f}m", color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    plt.colorbar(label='Monocular Depth (m)')
    plt.title('Monocular Depth')
    plt.axis('off')
    
    # 3. LiDAR depth
    plt.subplot(143)
    plt.imshow(frame_rgb)
    if lidar_depth is not None:
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            box_depth = robust_box_depth(lidar_depth[y1:y2, x1:x2])
            if np.isfinite(box_depth):
                depth_roi = lidar_depth[y1:y2, x1:x2]
                plt.imshow(depth_roi, extent=[x1, x2, y2, y1], cmap='magma', alpha=0.7)
                plt.text(x1, y1-10, f"D:{box_depth:.1f}m", color='white', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        if points_2d is not None:
            plt.scatter(points_2d[:, 0], points_2d[:, 1], c=depths, 
                       cmap='magma', alpha=0.5, s=1)
        plt.colorbar(label='LiDAR Depth (m)')
    plt.title('LiDAR Ground Truth')
    plt.axis('off')
    
    # 4. ECW visualization
    plt.subplot(144)
    plt.imshow(frame_rgb)
    
    # Create ECW region
    top_y = int(H * 0.55)
    bot_y = int(H * 0.90)
    top_w = int(W * 0.20)
    bot_w = int(W * 0.90)
    tilt_offset = int(W * 0.15)
    
    # Calculate trapezoid points
    top_left_x = W//2 - top_w//2 - tilt_offset//2
    top_right_x = W//2 + top_w//2 - tilt_offset//2
    bot_left_x = W//2 - bot_w//2 + tilt_offset//2
    bot_right_x = W//2 + bot_w//2 + tilt_offset//2
    points = np.array([
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bot_right_x, bot_y],
        [bot_left_x, bot_y]
    ])
    
    # Create ECW mask
    ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
    pts = points.astype(np.int32)
    cv2.fillPoly(ecw_mask_arr, [pts], 1)
    ecw_mask = ecw_mask_arr.astype(bool)
    
    # Show ECW region with more visible colors
    plt.imshow(np.ma.masked_where(~ecw_mask, np.ones_like(ecw_mask)), 
               cmap='RdBu', alpha=0.3)  # Using RdBu colormap for better visibility
    
    # Process each detection
    ttc_tracker = TTCTracker()
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Check if object is in ECW region
        box_mask = np.zeros_like(ecw_mask)
        box_mask[y1:y2, x1:x2] = 1
        in_ecw = np.any(np.logical_and(ecw_mask, box_mask))
        
        # Get warning threshold based on class
        if cls_idx in [1, 3]:  # Person or bike
            warn_threshold = T_WARN_VRU
            box_color = 'cyan' if cls_idx == 3 else 'green'
        else:  # Vehicle
            warn_threshold = T_WARN_VEHICLE
            box_color = 'magenta'
        
        # Add hysteresis to threshold
        warn_threshold += T_HYSTERESIS if in_ecw else 0
        
        # Get depths from each source
        depths = {}
        if depth_map is not None:
            depths['mono'] = robust_box_depth(depth_map[y1:y2, x1:x2])
        if lidar_depth is not None:
            depths['lidar'] = robust_box_depth(lidar_depth[y1:y2, x1:x2])
        if fused_depth is not None:
            depths['fused'] = robust_box_depth(fused_depth[y1:y2, x1:x2])
        
        # Compute TTC
        if len(depths) > 0:
            min_depth = min(depths.values())
            ttc = min_depth / 10.0  # Assume 10 m/s velocity for simplicity
            
            # Determine warning status
            if in_ecw and ttc < warn_threshold:
                warning_ratio = warn_threshold / ttc
                if warning_ratio > 2.0:
                    box_color = 'red'  # Critical
                else:
                    box_color = 'yellow'  # Warning
        
        # Draw box and label
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=box_color, linewidth=2))
        
        # Create multi-line label with depths
        label_lines = [f"{CLASSES[cls_idx]}"]
        for src, d in depths.items():
            label_lines.append(f"{src}: {d:.1f}m")
        if len(depths) > 0:
            label_lines.append(f"TTC: {ttc:.1f}s")
        
        # Draw label with black background for visibility
        y = y1-10
        for line in label_lines:
            plt.text(x1, y, line, color=box_color, fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            y -= 12
    
    plt.title('ECW with Multi-Source Depth')
    plt.axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(figures_dir, f"frame_{frame_number}_depth_fusion.png")
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved multi-source depth visualization: {viz_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=27400,
                       help='Frame number to process')
    args = parser.parse_args()
    
    create_visualization_figures(args.frame)