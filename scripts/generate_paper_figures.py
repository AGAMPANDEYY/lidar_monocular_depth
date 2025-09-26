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

def extract_frame(video_path, frame_number, output_path=None):
    """Extract specific frame from video and crop to front view using ROIs."""
    from scripts.extract_frames import load_rois, VIEW_NAMES_DEFAULT
    
    # Load ROIs
    rois_yaml = "scripts/config/default_rois.yaml"
    if not os.path.exists(rois_yaml):
        raise ValueError(f"ROIs YAML not found at {rois_yaml}. Run extract_frames.py first to generate ROIs.")
    
    rois, crop_dimensions = load_rois(rois_yaml, VIEW_NAMES_DEFAULT)
    front_roi = rois[VIEW_NAMES_DEFAULT.index("front")]  # Get front view ROI
    
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number}")
    
    # Crop to front view
    x1, y1, x2, y2 = front_roi
    frame_cropped = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    
    if output_path:
        Image.fromarray(frame_rgb).save(output_path)
        
    return frame_rgb
from main import (make_ecw_mask, paint_depth_background,  # Reuse ECW utilities
              T_WARN_VRU, T_WARN_VEHICLE, T_WARN_DEFAULT, T_HYSTERESIS,  # TTC thresholds
              MIN_VALID_PIXELS, MIN_PERSISTENCE_FRAMES,  # Warning parameters
              MIN_SIZE_M, MAX_SIZE_M)  # Object size thresholds

def extract_frame(video_path, frame_number, output_path=None):
    """Extract specific frame from video and crop to front view using ROIs."""
    from scripts.extract_frames import load_rois, VIEW_NAMES_DEFAULT
    
    # Load ROIs
    rois_yaml = "scripts/config/default_rois.yaml"
    if not os.path.exists(rois_yaml):
        raise ValueError(f"ROIs YAML not found at {rois_yaml}. Run extract_frames.py first to generate ROIs.")
    
    rois, crop_dimensions = load_rois(rois_yaml, VIEW_NAMES_DEFAULT)
    front_roi = rois[VIEW_NAMES_DEFAULT.index("front")]  # Get front view ROI
    
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number}")
    
    # Extract frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number}")
    
    # Crop to front view
    x1, y1, x2, y2 = front_roi
    frame_cropped = frame[y1:y2, x1:x2]
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2RGB)
    
    if output_path:
        Image.fromarray(frame_rgb).save(output_path)
        
    return frame_rgb

def create_visualization_figures(frame_number):
    """Create all visualization figures for the paper."""
    
    # Create figures directory
    figures_dir = os.path.join(project_root, "data", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\nCreating figures in: {figures_dir}")

    # Input paths
    video_path = os.path.join(project_root, "data", "input.avi")
    
    # 1. Extract and save original frame
    print("\n1. Extracting frame...")
    frame_path = os.path.join(figures_dir, f"frame_{frame_number}.png")
    frame_rgb = extract_frame(video_path, frame_number, frame_path)
    print(f"Saved original frame: {frame_path}")

    # Load models
    print("\n2. Loading models...")
    yolo_model = load_yolo_model(os.path.join(project_root, "detection/best-e150 (1).pt"))
    run_depth, device, depth_name = load_depth_backend('midas')
    print(f"Models loaded: YOLO and {depth_name} on {device}")

    # Get frame dimensions
    H, W = frame_rgb.shape[:2]

    # 3. Get object detections (manual annotations or YOLO)
    print("\n3. Getting object detections...")
    
    # Check for manual annotations
    annotations_path = os.path.join(project_root, 'data', 'manual_annotations', f'frame_{frame_number}.json')
    if os.path.exists(annotations_path):
        print("Using manual annotations")
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        pred_bboxes = data['boxes']
    else:
        print("No manual annotations found, running YOLO detection")
        pred_bboxes = run_obstacle_detection(frame_rgb, yolo_model)
    
    # Visualization with custom colors per class
    det_vis = frame_rgb.copy()
    # Colors match the image: car (magenta), bus (orange), motorcycle/bicycle (cyan), person (green)
    colors = {
        0: (255, 0, 255),   # Car - Magenta
        1: (0, 255, 0),     # Person - Green
        2: (255, 165, 0),   # Bus - Orange
        3: (0, 255, 255),   # Motorcycle/Bicycle - Cyan
        4: (255, 255, 0),   # Truck - Yellow
    }
    
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        color = colors.get(cls_idx, (0, 255, 0))  # Default to green if class not in colors
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[cls_idx]} ({conf:.2f})"
        cv2.putText(det_vis, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    det_path = os.path.join(figures_dir, f"frame_{frame_number}_detection.png")
    Image.fromarray(det_vis).save(det_path)
    print(f"Saved detection visualization: {det_path}")

    # 4. Run monocular depth estimation
    print("\n4. Running monocular depth estimation...")
    depth_map = run_depth(frame_rgb)
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Create matplotlib figure for depth visualization
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(121)
    plt.imshow(frame_rgb)
    plt.title("RGB Image")
    plt.axis('off')
    
    # Depth map with colorbar
    plt.subplot(122)
    depth_plot = plt.imshow(depth_map, cmap='magma')
    plt.colorbar(depth_plot, label='Depth (m)')
    plt.title("Monocular Depth Estimation")
    plt.axis('off')
    
    # Save with tight layout
    depth_path = os.path.join(figures_dir, f"frame_{frame_number}_mono_depth.png")
    plt.tight_layout()
    plt.savefig(depth_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved monocular depth visualization: {depth_path}")

    # 5. Depth + Detection overlay
    print("\n5. Creating depth + detection overlay...")
    plt.figure(figsize=(12, 5))
    
    # Create depth visualization
    depth_vis = np.uint8(plt.cm.magma(plt.Normalize()(depth_map)) * 255)
    
    plt.subplot(121)
    plt.imshow(frame_rgb)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2))
        box_depth = np.median(depth_map[y1:y2, x1:x2])
        plt.text(x1, y1-10, f"{CLASSES[cls_idx]} ({conf:.2f})", color='g', fontsize=8)
    plt.title("RGB with Detections")
    plt.axis('off')
    
    # Depth visualization with detections
    plt.subplot(122)
    depth_plot = plt.imshow(depth_map, cmap='magma')
    plt.colorbar(depth_plot, label='Depth (m)')
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        box_depth = np.median(depth_map[y1:y2, x1:x2])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2))
        plt.text(x1, y1-10, f"{CLASSES[cls_idx]} D:{box_depth:.1f}m", color='g', fontsize=8)
    plt.title("Depth Map with Detections")
    plt.axis('off')
    
    plt.tight_layout()
    depth_det_path = os.path.join(figures_dir, f"frame_{frame_number}_depth_detection.png")
    plt.savefig(depth_det_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved depth + detection visualization: {depth_det_path}")

    # 6. Selective depth overlay
    print("\n6. Creating selective depth overlay...")
    plt.figure(figsize=(12, 5))
    
    # Create masked depth map
    depth_mask = np.zeros((H, W), dtype=bool)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        depth_mask[y1:y2, x1:x2] = True
    
    masked_depth = np.ma.masked_where(~depth_mask, depth_map)
    
    # Plot RGB with masked depth overlay
    plt.subplot(121)
    plt.imshow(frame_rgb)
    depth_overlay = plt.imshow(masked_depth, cmap='magma', alpha=0.7)
    plt.colorbar(depth_overlay, label='Depth (m)')
    
    # Add bounding boxes and labels
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        box_depth = np.median(depth_map[y1:y2, x1:x2])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2))
        plt.text(x1, y1-10, f"{CLASSES[cls_idx]} D:{box_depth:.1f}m", color='g', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.title("RGB with Selective Depth")
    plt.axis('off')
    
    # Plot depth map with mask overlay
    plt.subplot(122)
    plt.imshow(depth_map, cmap='magma')
    plt.imshow(depth_mask, cmap='gray', alpha=0.3)
    plt.colorbar(depth_overlay, label='Depth (m)')
    
    # Add bounding boxes
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2))
    plt.title("Depth Map with Detection Regions")
    plt.axis('off')
    
    plt.tight_layout()
    selective_path = os.path.join(figures_dir, f"frame_{frame_number}_selective_depth.png")
    plt.savefig(selective_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved selective depth visualization: {selective_path}")

    # 7 & 8. LiDAR projection
    print("\n7 & 8. Processing LiDAR projection...")
    import subprocess
    
    # Use known LiDAR frame number for frame 27400 (at 18:17 min)
    lidar_frame = 10981
    
    # Process LiDAR data
    lidar_pcd = os.path.join(project_root, "data/lidar", f"input (Frame {lidar_frame}).pcd")
    if os.path.exists(lidar_pcd):
        cmd = [
            "python",
            os.path.join(project_root, "lidar_projection/project_lidar.py"),
            "--pcd", lidar_pcd,
            "--cam_yaml", os.path.join(project_root, "calibration/camera.yaml"),
            "--ext_yaml", os.path.join(project_root, "calibration/extrinsics_lidar_to_cam.yaml"),
            "--image", frame_path,
            "--out_npz", os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz"),
            "--debug_overlay", os.path.join(figures_dir, f"frame_{frame_number}_lidar_overlay.png")
        ]
        subprocess.run(cmd, check=True)
        lidar_proj_path = os.path.join(figures_dir, f"frame_{frame_number}_lidar_overlay.png")
    else:
        print(f"Warning: LiDAR PCD file not found at {lidar_pcd}")
        lidar_proj_path = os.path.join(project_root, "data/fused_output/debug", 
                                      f"{frame_number}_03_lidar_mask_on_image.png")
    
    if os.path.exists(lidar_proj_path):
        # Load LiDAR data
        try:
            lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
            if not os.path.exists(lidar_npz):
                raise FileNotFoundError(f"LiDAR NPZ file not found: {lidar_npz}")
            
            lidar_data = np.load(lidar_npz, allow_pickle=True)
            data_dict = dict(lidar_data)
            
            # Load depth map and mask from LiDAR projection
            Dlidar = data_dict['Dlidar']
            Mlidar = data_dict['Mlidar']
            
            # Create 2D points from valid depth pixels
            y_idx, x_idx = np.where(Mlidar)
            points_2d = np.stack([x_idx, y_idx], axis=1)
            depths = Dlidar[Mlidar]
            
            if len(points_2d) == 0:
                raise KeyError("No valid LiDAR points found")
            
            # Create visualization with matplotlib
            plt.figure(figsize=(15, 5))
            
            # Full LiDAR projection
            plt.subplot(131)
            plt.imshow(frame_rgb)
            scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], 
                                c=depths, cmap='magma', alpha=0.7, s=1)
            plt.colorbar(scatter, label='Depth (m)')
            plt.title('LiDAR Points Projection')
            plt.axis('off')
            
            # LiDAR depth interpolation
            plt.subplot(132)
            from scipy.interpolate import griddata
            grid_x, grid_y = np.mgrid[0:H:1, 0:W:1]
            lidar_depth = griddata(points_2d, depths, (grid_x, grid_y), method='linear')
            lidar_vis = plt.imshow(lidar_depth, cmap='magma')
            plt.colorbar(lidar_vis, label='Depth (m)')
            plt.title('LiDAR Depth Interpolation')
            plt.axis('off')
            
            # Selective LiDAR (only in detection regions)
            plt.subplot(133)
            plt.imshow(frame_rgb)
            masked_lidar = np.ma.masked_where(~depth_mask, lidar_depth)
            lidar_overlay = plt.imshow(masked_lidar, cmap='magma', alpha=0.7)
            plt.colorbar(lidar_overlay, label='Depth (m)')
            
            # Add detection boxes
            for box in pred_bboxes:
                x1, y1, x2, y2 = map(int, box[:4])
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                                fill=False, color='g', linewidth=2))
            plt.title('Selective LiDAR Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            lidar_path = os.path.join(figures_dir, f"frame_{frame_number}_lidar_viz.png")
            plt.savefig(lidar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved LiDAR visualizations")
            
        except Exception as e:
            print(f"Warning: Could not process LiDAR data: {e}")
            # Fallback to original LiDAR projection image
            lidar_proj = cv2.imread(lidar_proj_path)
            lidar_proj = cv2.cvtColor(lidar_proj, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.imshow(frame_rgb)
            plt.title('RGB Image')
            plt.axis('off')
            
            plt.subplot(122)
            plt.imshow(lidar_proj)
            plt.title('LiDAR Projection')
            plt.axis('off')
            
            plt.tight_layout()
            lidar_path = os.path.join(figures_dir, f"frame_{frame_number}_lidar.png")
            plt.savefig(lidar_path, dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved basic LiDAR visualization")
    else:
        print(f"Warning: LiDAR projection not found at {lidar_proj_path}")

    # 9. Combined visualization with fused depths
    print("\n9. Creating combined visualization...")
    
    # Load ECW annotation if available
    ecw_annotation_file = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{frame_number}.json')
    
    if os.path.exists(ecw_annotation_file):
        print("Loading ECW bubble from annotation file for combined visualization")
        with open(ecw_annotation_file, 'r') as f:
            data = json.load(f)
            if 'points' in data:
                points = np.array(data['points'])
                print(f"Successfully loaded ECW points: {points.tolist()}")
            else:
                raise ValueError("Points not found in annotation file")
    else:
        print("No ECW annotation found, using default trapezoid")
        # Create default trapezoid vertices
        top_y = int(H * 0.55)
        bot_y = int(H * 0.90)
        top_w = int(W * 0.20)
        bot_w = int(W * 0.90)
        tilt_offset = int(W * 0.15)
        
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
    
    # Load all depth sources
    fused_path = os.path.join(project_root, "data/fused_output/debug", f"{frame_number}_fused_depth.npy")
    lidar_path = os.path.join(project_root, "data/fused_output/debug", f"{frame_number}_lidar_depth.npy")
    
    plt.figure(figsize=(20, 5))
    
    # 1. RGB + Boxes + ECW
    plt.subplot(141)
    plt.imshow(frame_rgb)
    
    # Draw ECW region
    plt.imshow(np.ma.masked_where(~ecw_mask, np.ones_like(ecw_mask)), 
              cmap='Greens', alpha=0.2)
    
    ttc_tracker = TTCTracker()  # Initialize TTC tracker
    
    # Process each detection
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Check if in ECW
        box_mask = np.zeros_like(ecw_mask)
        box_mask[y1:y2, x1:x2] = 1
        in_ecw = np.any(np.logical_and(ecw_mask, box_mask))
        
        # Get depth from all sources
        depths = {
            'mono': np.median(depth_map[y1:y2, x1:x2])
        }
        if os.path.exists(lidar_path):
            lidar_depth = np.load(lidar_path)
            depths['lidar'] = np.median(lidar_depth[y1:y2, x1:x2])
        if os.path.exists(fused_path):
            fused_depth = np.load(fused_path)
            depths['fused'] = np.median(fused_depth[y1:y2, x1:x2])
        
        # Determine warning level
        if cls_idx in [1, 3]:  # Person or bike
            warn_threshold = T_WARN_VRU
            base_color = 'cyan' if cls_idx == 3 else 'green'
        else:  # Vehicle
            warn_threshold = T_WARN_VEHICLE
            base_color = 'magenta'
        
        # Add position-based criticality
        if in_ecw and 'fused' in depths:
            depth = depths['fused']
            # Calculate criticality based on vertical position within ECW region
            ecw_y_min = points[:, 1].min()  # Top of ECW region
            ecw_y_max = points[:, 1].max()  # Bottom of ECW region
            ecw_y_pos = (y2 - ecw_y_min) / (ecw_y_max - ecw_y_min)
            criticality = min(1.0, ecw_y_pos * 1.5)
            adjusted_threshold = warn_threshold * (1.0 - criticality * 0.5)
            
            if depth < adjusted_threshold:
                severity = (adjusted_threshold - depth) / adjusted_threshold
                color = 'red' if severity > 0.5 else 'yellow'
            else:
                color = base_color
        else:
            color = base_color
        
        # Draw box and label
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        
        # Create multi-line label
        label = [f"{CLASSES[cls_idx]}"]
        for src, d in depths.items():
            label.append(f"{src}: {d:.1f}m")
        
        y_text = y1-10
        for line in label:
            plt.text(x1, y_text, line, color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            y_text -= 12
    
    plt.title('RGB with Detections and ECW')
    plt.axis('off')
    
    # 2. MiDaS Depth
    plt.subplot(142)
    plt.imshow(depth_map, cmap='magma')
    plt.colorbar(label='Depth (m)')
    plt.title('MiDaS Depth')
    plt.axis('off')
    
    # 3. LiDAR Depth
    plt.subplot(143)
    if os.path.exists(lidar_path):
        lidar_depth = np.load(lidar_path)
        plt.imshow(lidar_depth, cmap='magma')
        plt.colorbar(label='Depth (m)')
    else:
        plt.text(0.5, 0.5, 'No LiDAR data', ha='center', va='center')
    plt.title('LiDAR Depth')
    plt.axis('off')
    
    # 4. Fused Depth with RGB background
    plt.subplot(144)
    if os.path.exists(fused_path):
        fused_depth = np.load(fused_path)
        
        # First show RGB image
        plt.imshow(frame_rgb)
        
        # Then overlay depth with high transparency
        depth_overlay = plt.imshow(fused_depth, cmap='magma', alpha=0.5)
        plt.colorbar(depth_overlay, label='Depth (m)')
        
        # Overlay ECW outline
        plt.fill(points[:, 0], points[:, 1], 'g', alpha=0.2)
        plt.plot(points[:, 0], points[:, 1], 'g-', linewidth=2)
        plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 'g-', linewidth=2)
        
        # Add detection boxes and labels with more information
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_idx = int(box[4])
            conf = float(box[5])
            
            # Get depth information from different sources
            mono_depth = np.median(depth_map[y1:y2, x1:x2])
            lidar_depth = np.median(fused_depth[y1:y2, x1:x2])  # Using fused as representative
            
            # Choose box color based on class
            if cls_idx in [1, 3]:  # Person or bike
                color = 'cyan' if cls_idx == 3 else 'lime'
            else:  # Vehicles
                color = 'magenta'
            
            # Draw bounding box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            fill=False, color=color, linewidth=2))
            
            # Create detailed multi-line label
            label_lines = [
                f"{CLASSES[cls_idx]} ({conf:.2f})",
                f"Mono: {mono_depth:.1f}m",
                f"Fused: {lidar_depth:.1f}m"
            ]
            
            # Position label above box
            y_text = y1 - 5
            for line in reversed(label_lines):
                plt.text(x1, y_text, line, color=color, fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                y_text -= 12
            
    else:
        plt.text(0.5, 0.5, 'No fused depth', ha='center', va='center')
    plt.title('Fused Depth with ECW')
    plt.axis('off')
    
    plt.tight_layout()
    combined_path = os.path.join(figures_dir, f"frame_{frame_number}_all_depths.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined depth visualization: {combined_path}")
            
    print(f"\nSaved depth visualization: {combined_path}")
    
    # 10. ECW visualization
    print("\n10. Creating ECW visualization...")
    # Load manual ECW annotation
    ecw_annotation_file = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{frame_number}.json')
    
    if os.path.exists(ecw_annotation_file):
        print("Loading ECW bubble from annotation file")
        with open(ecw_annotation_file, 'r') as f:
            data = json.load(f)
            if 'points' in data:
                points = np.array(data['points'])
                print(f"Successfully loaded ECW points: {points.tolist()}")
            else:
                raise ValueError("Points not found in annotation file")
    else:
        print(f"No ECW annotation found at {ecw_annotation_file}, using default trapezoid")
        # Create default trapezoid vertices
        top_y = int(H * 0.55)
        bot_y = int(H * 0.90)
        top_w = int(W * 0.20)
        bot_w = int(W * 0.90)
        
        # Add left tilt
        tilt_offset = int(W * 0.15)
        
        # Calculate points
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
    
    # Create ECW mask from annotated points
    ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
    pts = points.astype(np.int32)
    cv2.fillPoly(ecw_mask_arr, [pts], 1)
    ecw_mask = ecw_mask_arr.astype(bool)
    
    print(f"Created ECW mask from points: {points.tolist()}")
    
    # Initialize TTC tracker
    ttc_tracker = TTCTracker()
    
    plt.figure(figsize=(15, 5))
    
    # Original image with ECW overlay
    plt.subplot(131)
    plt.imshow(frame_rgb)
    
    # Draw ECW region
    # Fill with semi-transparent color
    plt.fill(points[:, 0], points[:, 1], 'g', alpha=0.2)
    
    # Draw outline
    plt.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, label='ECW Region')
    plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 'g-', linewidth=2)
    
    # Add points for verification
    plt.scatter(points[:, 0], points[:, 1], c='g', s=50)
    
    # Compute ECW bubble based on depth sources
    if os.path.exists(fused_path):
        fused_depth = np.load(fused_path)
        # Compute ECW bubble using default warning thresholds from main.py
        # Create ECW bubble using fused depth
        H, W = frame_rgb.shape[:2]
        ecw_bubble = np.zeros((H, W), dtype=bool)
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            depth = np.median(fused_depth[y1:y2, x1:x2])
            if depth < T_WARN_VRU if box[4] in [1, 3] else T_WARN_VEHICLE:
                box_mask = np.zeros((H, W), dtype=bool)
                box_mask[y1:y2, x1:x2] = True
                ecw_bubble |= box_mask & ecw_mask
        plt.imshow(np.ma.masked_where(~ecw_bubble, np.ones_like(ecw_bubble)), 
                  cmap='Reds', alpha=0.3)
    else:
        plt.imshow(np.ma.masked_where(~ecw_mask, np.ones_like(ecw_mask)), 
                  cmap='Greens', alpha=0.3)
    plt.title('RGB with ECW Bubble')
    plt.axis('off')
    
    # Multi-source depth comparison with enhanced visualization
    plt.subplot(132)
    plt.imshow(frame_rgb)
    
    if os.path.exists(fused_path):
        fused_depth = np.load(fused_path)
        # Load LiDAR data if available
        lidar_depth = np.load(os.path.join(project_root, "data/fused_output/debug", 
                                          f"{frame_number}_lidar_depth.npy"))
                                          
        # Create depth visualization with more transparency
        depth_vis = np.ma.masked_where(~ecw_mask, fused_depth)
        norm = plt.Normalize(vmin=fused_depth.min(), vmax=fused_depth.max())
        cmap = plt.cm.magma
        depth_colors = cmap(norm(fused_depth))
        depth_colors[..., 3] = 0.4  # Set alpha for all pixels
        
        plt.imshow(depth_vis, cmap='magma', alpha=0.4)
        cbar = plt.colorbar(label='Depth (m)')
        cbar.ax.set_title('m', pad=10)
    
    # Add detections with TTC from each source
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Get object region and check if in ECW
        box_mask = np.zeros_like(ecw_mask)
        box_mask[y1:y2, x1:x2] = 1
        in_ecw = np.any(np.logical_and(ecw_mask, box_mask))
        
        # Compute TTC and warning threshold based on class
        if cls_idx in [1, 3]:  # Person or bike
            warn_threshold = T_WARN_VRU
            box_color = 'cyan' if cls_idx == 3 else 'green'  # Cyan for bike, green for person
        else:  # Vehicle
            warn_threshold = T_WARN_VEHICLE
            box_color = 'magenta'  # Magenta for vehicles
            
        # Process depth and warning information
        color = box_color  # Default to class color
        status = "OK"
        depth_info = ""
        
        if os.path.exists(fused_path):
            depth = np.median(fused_depth[y1:y2, x1:x2])
            
            # Define class-specific thresholds
            class_thresholds = {
                0: 8.0,    # Cars: warning if closer than 8m
                1: 5.0,    # People: warning if closer than 5m
                2: 12.0,   # Bus: warning if closer than 12m (larger vehicle)
                3: 5.0,    # Motorcycle: warning if closer than 5m
                4: 10.0    # Truck: warning if closer than 10m
            }
            dist_threshold = class_thresholds.get(cls_idx, 8.0)
            
            # Calculate position-based criticality
            ecw_y_min = points[:, 1].min()  # Top of ECW region
            ecw_y_max = points[:, 1].max()  # Bottom of ECW region
            ecw_y_pos = (y2 - ecw_y_min) / (ecw_y_max - ecw_y_min)
            criticality = min(1.0, ecw_y_pos * 1.5)
            adjusted_threshold = dist_threshold * (1.0 - criticality * 0.5)
            
            # Determine warning status and color
            if depth < adjusted_threshold:
                severity_ratio = (adjusted_threshold - depth) / adjusted_threshold
                if severity_ratio > 0.5:
                    color = 'red'      # Critical warning
                    status = "CRITICAL"
                else:
                    color = 'orange'   # Warning
                    status = "WARNING"
            
            depth_info = f"D:{depth:.1f}m"
            
        # Draw bounding box with appropriate color
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        
        # Add label with class, status, and depth info
        label = f"{CLASSES[cls_idx]}"
        if status != "OK":
            label += f"\n{status}"
        if depth_info:
            label += f"\n{depth_info}"
            
            # Calculate relative position in ECW
            if in_ecw:
                # Calculate vertical position in ECW (0 = top, 1 = bottom)
                ecw_y_min = points[:, 1].min()  # Top of ECW region
                ecw_y_max = points[:, 1].max()  # Bottom of ECW region
                ecw_y_pos = (y2 - ecw_y_min) / (ecw_y_max - ecw_y_min)
                # Objects closer to bottom of ECW are more critical
                criticality = min(1.0, ecw_y_pos * 1.5)  # Increase weight of lower positions
                # Adjust threshold based on position (stricter for closer objects)
                dist_threshold *= (1.0 - criticality * 0.5)  # Up to 50% stricter for close objects
            
            # Determine warning status
            if in_ecw and depth < dist_threshold:
                status = "WARNING"
                # Add severity level based on how much closer than threshold
                severity_ratio = (dist_threshold - depth) / dist_threshold
                if severity_ratio > 0.5:  # More than 50% closer than threshold
                    status = "CRITICAL"
                else:
                    status = "OK"
            
            label = f"{CLASSES[cls_idx]}\n{status}\nD:{depth:.1f}m"
        else:
            label = f"{CLASSES[cls_idx]}\n{'OK'}"
        
        plt.text(x1, y1-10, label, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.title('ECW with Detections')
    plt.axis('off')
    
    # Depth map with ECW
    if os.path.exists(fused_path):
        plt.subplot(133)
        fused_depth = np.load(fused_path)
        
        # Create depth visualization with ECW outline
        plt.imshow(fused_depth, cmap='magma')
        plt.colorbar(label='Depth (m)')
        
        # Plot ECW region outline
        plt.fill(points[:, 0], points[:, 1], 'g', alpha=0.2)
        plt.plot(points[:, 0], points[:, 1], 'g-', linewidth=2)
        plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 'g-', linewidth=2)
        
        plt.title('Depth Map with ECW Region')
        plt.axis('off')
    
    plt.tight_layout()
    ecw_path = os.path.join(figures_dir, f"frame_{frame_number}_ecw.png")
    plt.savefig(ecw_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ECW visualization: {ecw_path}")

    print("\nAll visualizations completed!")
    print(f"Output directory: {figures_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=27400,
                    help='Frame number to extract and visualize')
    args = parser.parse_args()
    
    create_visualization_figures(args.frame)