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

def create_single_visualization(frame_rgb, depth_map, pred_bboxes, lidar_data=None, figures_dir=None, object_tracks=None, frame_idx=None, fps=25):
    """Create a single visualization combining original image, depth, detections, and LiDAR."""
    H, W = frame_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame_rgb)
    depth_norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
    depth_colored = plt.cm.magma(depth_norm(depth_map))
    depth_colored[..., 3] = 0.3
    ax.imshow(depth_colored)
    # 3. Draw bounding boxes and labels
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        mono_depth = np.median(depth_map[y1:y2, x1:x2])
        lidar_depth = np.nan
        fused_depth = np.nan
        if lidar_data is not None and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
            lidar_mask_bbox = lidar_data['Mlidar'][y1:y2, x1:x2]
            lidar_vals = lidar_data['Dlidar'][y1:y2, x1:x2][lidar_mask_bbox]
            if lidar_vals.size > 0:
                lidar_depth = float(np.median(lidar_vals))
                w_lidar, w_mono = 0.8, 0.2
                fused_depth = w_lidar * lidar_depth + w_mono * mono_depth
            else:
                fused_depth = mono_depth
        color = 'cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta'
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        # --- Accurate TTC Calculation using tracked object speed over last 75 frames ---
        ego_speed = 10.33  # m/s (replace with actual ego speed if available)
        object_speed = 0.0
        track_id = int(box[7]) if len(box) > 7 else None
        # Estimate object speed using last 75 frames (or fewer if not enough history)
        if object_tracks is not None and track_id is not None and track_id in object_tracks:
            track = object_tracks[track_id]
            window = min(75, len(track))
            if window >= 2:
                t0 = track[-window]
                tN = track[-1]
                # Use depth for Z, bbox center for X,Y
                x0, y0, z0 = t0['center'][0], t0['center'][1], t0['depth']
                xN, yN, zN = tN['center'][0], tN['center'][1], tN['depth']
                # Physical distance moved (Euclidean)
                dist = np.sqrt((xN-x0)**2 + (yN-y0)**2 + (zN-z0)**2)
                dt = (tN['frame'] - t0['frame']) / fps
                if dt > 0:
                    object_speed = dist / dt
        # Compute relative speed (ego - object), clamp to avoid division by zero
        rel_speed = max(ego_speed - object_speed, 1e-3)
        ttc = np.nan
        if np.isfinite(fused_depth) and fused_depth > 0 and rel_speed > 0:
            ttc = fused_depth / rel_speed
        # --- End TTC calculation ---
        label_lines = [
            f"{CLASSES[cls_idx]} ({conf:.2f})",
            f"MiDaS: {mono_depth:.1f}m"
        ]
        if np.isfinite(lidar_depth):
            label_lines.append(f"LiDAR: {lidar_depth:.1f}m")
        if np.isfinite(fused_depth):
            label_lines.append(f"Fused: {fused_depth:.1f}m")
        if np.isfinite(ttc):
            label_lines.append(f"TTC: {ttc:.1f}s")
        y_text = y1 - 5
        for line in reversed(label_lines):
            if line.startswith("TTC:"):
                plt.text(x1, y_text, line, color='black', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            else:
                plt.text(x1, y_text, line, color=color, fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            y_text -= 12
    # 4. Overlay LiDAR points if available
    if lidar_data is not None and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
        y_idx, x_idx = np.where(lidar_data['Mlidar'])
        points_2d = np.stack([x_idx, y_idx], axis=1)
        depths = lidar_data['Dlidar'][lidar_data['Mlidar']]
        if len(points_2d) > 0:
            scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], 
                                c=depths, cmap='plasma', alpha=0.9, s=5,
                                norm=plt.Normalize(vmin=0, vmax=50),
                                edgecolor='white', linewidth=0.5)
            plt.colorbar(scatter, label='LiDAR Points Depth (m)', pad=0.01)
    plt.colorbar(plt.cm.ScalarMappable(norm=depth_norm, cmap='magma'), 
                ax=ax, label='Depth (m)')
    ax.set_title('Combined Visualization')
    ax.axis('off')
    if figures_dir is not None:
        plt.tight_layout()
        output_path = os.path.join(figures_dir, 'combined_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        plt.close()

def scale_midas_with_lidar(depth_map, lidar_data):
    """Scale MiDaS depth map using LiDAR data."""
    if lidar_data is None or 'Dlidar' not in lidar_data or 'Mlidar' not in lidar_data:
        return depth_map
        
    Dlidar = lidar_data['Dlidar']
    Mlidar = lidar_data['Mlidar']
    
    # Create valid mask for LiDAR points (1m to 50m range)
    valid_mask = Mlidar & (Dlidar > 1.0) & (Dlidar < 50.0)
    if valid_mask.sum() < 100:
        print("Warning: Not enough valid LiDAR points for scaling")
        return depth_map
        
    # Get corresponding depth values
    mono_points = depth_map[valid_mask]
    lidar_points = Dlidar[valid_mask]
    
    # Compute scaling factor using median ratio
    scale = np.median(lidar_points / (mono_points + 1e-6))
    if not (np.isfinite(scale) and scale > 0):
        print("Warning: Invalid scaling factor computed")
        return depth_map
        
    print(f"Applying depth scaling factor: {scale:.3f}")
    scaled_depth = depth_map * scale
    
    # Clip to reasonable range
    scaled_depth = np.clip(scaled_depth, 0.1, 80.0)
    
    return scaled_depth

def create_visualization_figures(frame_number):
    """Create all visualization figures for the paper."""
    
    # Create figures directory
    figures_dir = os.path.join(project_root, "data", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\nCreating figures in: {figures_dir}")
    # --- Missed LiDAR Depth-Driven Hazard Detection in ECW Region ---
    def visualize_missed_hazards_ecw(frame_rgb, depth_map, pred_bboxes, fused_depth, figures_dir, frame_number):
        # --- ECW mask creation (from annotation) ---
        H, W = frame_rgb.shape[:2]
        ecw_json_path = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{frame_number}.json')
        ecw_points = None
        ecw_mask = np.zeros((H, W), dtype=bool)
        if os.path.exists(ecw_json_path):
            with open(ecw_json_path, 'r') as f:
                ecw_data = json.load(f)
            if 'points' in ecw_data:
                ecw_points = np.array(ecw_data['points'], dtype=np.int32)
            elif 'ecw_zone' in ecw_data:
                ecw_points = np.array(ecw_data['ecw_zone'], dtype=np.int32)
        if ecw_points is not None and ecw_points.shape[0] >= 3:
            ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
            pts = ecw_points.astype(np.int32)
            cv2.fillPoly(ecw_mask_arr, [pts], 1)
            ecw_mask = ecw_mask_arr.astype(bool)
        # --- bbox_mask creation (for all bboxes) ---
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            bbox_mask[y1:y2, x1:x2] = 1
        # --- Step 4: Overlay missed blobs on missed hazard ECW visualization ---
        # Only do this if LiDAR and ECW mask are available
        lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
        combined_path = os.path.join(figures_dir, f"frame_{frame_number}_missed_hazards_ecw_with_blobs.png")
        if os.path.exists(lidar_npz) and np.any(ecw_mask):
            lidar_data = np.load(lidar_npz, allow_pickle=True)
            Dlidar = lidar_data['Dlidar']
            Mlidar = lidar_data['Mlidar']
            mask_missed = ecw_mask & (bbox_mask == 0) & Mlidar
            y_idx, x_idx = np.where(mask_missed)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(frame_rgb)
            # Overlay ECW bubble zone
            if ecw_points is not None and ecw_points.shape[0] >= 3:
                poly = plt.Polygon(ecw_points, closed=True, color=(0.1,0.1,0.3,0.5), edgecolor='yellow', linewidth=2)
                ax.add_patch(poly)
                centroid = np.mean(ecw_points, axis=0)
                ax.text(centroid[0], centroid[1], 'ECW Zone', color='yellow', fontsize=16, weight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            # Overlay fused depth in ECW
            fused_ecw = np.ma.masked_where(~ecw_mask, fused_depth)
            depth_overlay = ax.imshow(fused_ecw, cmap='magma', alpha=0.45)
            fig.colorbar(depth_overlay, ax=ax, label='Fused Depth (m)')
            # Draw detection boxes and highlight missed hazards
            for box in pred_bboxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cls_idx = int(box[4])
                conf = float(box[5])
                box_mask = np.zeros_like(ecw_mask)
                box_mask[y1:y2, x1:x2] = 1
                in_ecw = np.any(np.logical_and(ecw_mask, box_mask))
                obj_fused_depth = np.median(fused_depth[y1:y2, x1:x2])
                class_thresholds = {0: 8.0, 1: 5.0, 2: 12.0, 3: 5.0, 4: 10.0}
                warn_threshold = class_thresholds.get(cls_idx, 8.0)
                missed = in_ecw and obj_fused_depth < warn_threshold
                color = 'red' if missed else ('cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta')
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
                label = f"{CLASSES[cls_idx]} ({conf:.2f})\nFused: {obj_fused_depth:.1f}m"
                if missed:
                    label += "\nMISSED HAZARD"
                ax.text(x1, y1-10, label, color=color, fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            # Overlay missed blobs
            if len(x_idx) > 0:
                from sklearn.cluster import DBSCAN
                import matplotlib.colors as mcolors
                coords = np.stack([x_idx, y_idx], axis=1)
                clustering = DBSCAN(eps=10, min_samples=8).fit(coords)
                labels = clustering.labels_
                unique_labels = [l for l in set(labels) if l != -1]
                # Use a colormap for distinct colors
                cmap = plt.get_cmap('tab10')
                n_blobs = 0
                for i, label in enumerate(unique_labels):
                    blob = coords[labels == label]
                    if len(blob) > 20:
                        color = cmap(i % 10)
                        ax.scatter(blob[:,0], blob[:,1], c=[color], s=12, alpha=0.7, label=f'Missed Blob {n_blobs+1}')
                        n_blobs += 1
                if n_blobs > 0:
                    ax.legend()
            ax.set_title('Missed Hazards & Missed Object Blobs in ECW')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined missed hazard + blob visualization: {combined_path}")
        # --- ECW mask creation (from annotation) ---
        ecw_json_path = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{frame_number}.json')
        ecw_points = None
        ecw_mask = np.zeros((H, W), dtype=bool)
        if os.path.exists(ecw_json_path):
            with open(ecw_json_path, 'r') as f:
                ecw_data = json.load(f)
            if 'points' in ecw_data:
                ecw_points = np.array(ecw_data['points'], dtype=np.int32)
            elif 'ecw_zone' in ecw_data:
                ecw_points = np.array(ecw_data['ecw_zone'], dtype=np.int32)
        if ecw_points is not None and ecw_points.shape[0] >= 3:
            ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
            pts = ecw_points.astype(np.int32)
            cv2.fillPoly(ecw_mask_arr, [pts], 1)
            ecw_mask = ecw_mask_arr.astype(bool)
        # --- Step 1: Visualize ECW mask and bbox mask ---
        # Create union mask for all bboxes
        bbox_mask = np.zeros((H, W), dtype=np.uint8)
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            bbox_mask[y1:y2, x1:x2] = 1
        # Save bbox mask visualization
        plt.figure(figsize=(8, 5))
        plt.imshow(frame_rgb)
        plt.imshow(bbox_mask, cmap='Reds', alpha=0.3)
        plt.title('Bounding Box Mask')
        plt.axis('off')
        bbox_mask_path = os.path.join(figures_dir, f"frame_{frame_number}_bbox_mask.png")
        plt.savefig(bbox_mask_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved bbox mask visualization: {bbox_mask_path}")

        # --- Step 2: Find LiDAR points in ECW zone not in any bbox ---
        # Check for LiDAR data
        lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
        if os.path.exists(lidar_npz):
            lidar_data = np.load(lidar_npz, allow_pickle=True)
            Dlidar = lidar_data['Dlidar']
            Mlidar = lidar_data['Mlidar']
            mask_missed = ecw_mask & (bbox_mask == 0) & Mlidar
            y_idx, x_idx = np.where(mask_missed)
            # Save missed points visualization
            plt.figure(figsize=(8, 5))
            plt.imshow(frame_rgb)
            plt.scatter(x_idx, y_idx, c='orange', s=8, alpha=0.7, label='Missed LiDAR Points')
            plt.title('LiDAR Points in ECW (Not in BBox)')
            plt.axis('off')
            missed_pts_path = os.path.join(figures_dir, f"frame_{frame_number}_missed_lidar_points.png")
            plt.savefig(missed_pts_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved missed LiDAR points visualization: {missed_pts_path}")

        # --- Step 3: Cluster missed points and plot blobs ---
        if len(x_idx) > 0:
            from sklearn.cluster import DBSCAN
            coords = np.stack([x_idx, y_idx], axis=1)
            clustering = DBSCAN(eps=10, min_samples=8).fit(coords)
            labels = clustering.labels_
            plt.figure(figsize=(8, 5))
            plt.imshow(frame_rgb)
            n_blobs = 0
            for label in set(labels):
                if label == -1:
                    continue  # noise
                blob = coords[labels == label]
                if len(blob) > 20:  # density threshold
                    plt.scatter(blob[:,0], blob[:,1], c='orange', s=12, alpha=0.7, label=f'Missed Blob {n_blobs+1}')
                    n_blobs += 1
            plt.title(f'Missed Object Blobs in ECW (n={n_blobs})')
            plt.axis('off')
            plt.legend()
            blob_path = os.path.join(figures_dir, f"frame_{frame_number}_missed_blobs.png")
            plt.savefig(blob_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved missed object blob visualization: {blob_path}")
        H, W = frame_rgb.shape[:2]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(frame_rgb)
        # --- Overlay ECW bubble zone from annotation ---
        ecw_json_path = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{frame_number}.json')
        ecw_points = None
        ecw_mask = np.zeros((H, W), dtype=bool)
        if os.path.exists(ecw_json_path):
            with open(ecw_json_path, 'r') as f:
                ecw_data = json.load(f)
            if 'points' in ecw_data:
                ecw_points = np.array(ecw_data['points'], dtype=np.int32)
            elif 'ecw_zone' in ecw_data:
                ecw_points = np.array(ecw_data['ecw_zone'], dtype=np.int32)
        if ecw_points is not None and ecw_points.shape[0] >= 3:
            ax.set_xlim([0, W])
            ax.set_ylim([H, 0])
            poly = plt.Polygon(ecw_points, closed=True, color=(0.1,0.1,0.3,0.5), edgecolor='yellow', linewidth=2)
            ax.add_patch(poly)
            centroid = np.mean(ecw_points, axis=0)
            ax.text(centroid[0], centroid[1], 'ECW Zone', color='yellow', fontsize=16, weight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            # Create mask for ECW region
            ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
            pts = ecw_points.astype(np.int32)
            cv2.fillPoly(ecw_mask_arr, [pts], 1)
            ecw_mask = ecw_mask_arr.astype(bool)
            # Overlay fused depth in ECW
            fused_ecw = np.ma.masked_where(~ecw_mask, fused_depth)
            depth_overlay = ax.imshow(fused_ecw, cmap='magma', alpha=0.45)
            fig.colorbar(depth_overlay, ax=ax, label='Fused Depth (m)')
        else:
            print(f"[WARNING] ECW annotation for frame {frame_number} is missing or invalid.")
        # Draw detection boxes and highlight missed hazards
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_idx = int(box[4])
            conf = float(box[5])
            # Check if box is in ECW
            box_mask = np.zeros_like(ecw_mask)
            box_mask[y1:y2, x1:x2] = 1
            in_ecw = np.any(np.logical_and(ecw_mask, box_mask))
            # Get fused depth for this object
            obj_fused_depth = np.median(fused_depth[y1:y2, x1:x2])
            # Class-specific warning threshold
            class_thresholds = {0: 8.0, 1: 5.0, 2: 12.0, 3: 5.0, 4: 10.0}
            warn_threshold = class_thresholds.get(cls_idx, 8.0)
            # Missed hazard: in ECW and fused depth < threshold
            missed = in_ecw and obj_fused_depth < warn_threshold
            color = 'red' if missed else ('cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta')
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
            label = f"{CLASSES[cls_idx]} ({conf:.2f})\nFused: {obj_fused_depth:.1f}m"
            if missed:
                label += "\nMISSED HAZARD"
            ax.text(x1, y1-10, label, color=color, fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.set_title('Missed LiDAR Depth-Driven Hazards in ECW Region')
        ax.axis('off')
        out_path = os.path.join(figures_dir, f"frame_{frame_number}_missed_hazards_ecw.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved missed hazard ECW visualization: {out_path}")

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

    # 3. Get object detections (manual annotations ONLY)
    print("\n3. Getting object detections...")
    annotations_path = os.path.join(project_root, 'data', 'manual_annotations', f'frame_{frame_number}.json')
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Manual annotation file not found: {annotations_path}")
    print("Using manual annotations ONLY for bounding boxes")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    pred_bboxes = data['boxes']
        
    # 4. Run monocular depth estimation with LiDAR-based scaling
    print("\n4. Running monocular depth estimation...")
    depth_map = run_depth(frame_rgb)
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Convert relative to absolute depth using LiDAR
    lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
    if os.path.exists(lidar_npz):
        print("Calibrating depth with LiDAR data...")
        lidar_data = np.load(lidar_npz, allow_pickle=True)
        data_dict = dict(lidar_data)
        Dlidar = data_dict['Dlidar']
        Mlidar = data_dict['Mlidar']
        
        # Scale using valid LiDAR points
        valid_mask = Mlidar & (Dlidar > 1.0) & (Dlidar < 50.0)
        if valid_mask.sum() >= 100:
            mono_points = depth_map[valid_mask]
            lidar_points = Dlidar[valid_mask]
            scale = np.median(lidar_points / (mono_points + 1e-6))
            if np.isfinite(scale) and scale > 0:
                depth_map = np.clip(depth_map * scale, 0.1, 80.0)
                print(f"Applied depth scaling: {scale:.3f}")
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    # Load LiDAR data if available
    lidar_npz = os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz")
    lidar_data = None
    if os.path.exists(lidar_npz):
        lidar_data = dict(np.load(lidar_npz, allow_pickle=True))
        
    # Create combined visualization
    combined_path = create_single_visualization(frame_rgb, depth_map, pred_bboxes, 
                                             lidar_data=lidar_data, figures_dir=figures_dir)
    print(f"Saved combined visualization: {combined_path}")
    
    # Visualization with custom colors per class
    # --- Compute TTC for all bboxes and display only averaged TTC ---
    ttc_list = []
    ego_speed = 10.33  # m/s (replace with actual ego speed if available)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        mono_depth = np.median(depth_map[y1:y2, x1:x2])
        lidar_depth = np.nan
        fused_depth = np.nan
        if lidar_data is not None and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
            lidar_mask_bbox = lidar_data['Mlidar'][y1:y2, x1:x2]
            lidar_vals = lidar_data['Dlidar'][y1:y2, x1:x2][lidar_mask_bbox]
            if lidar_vals.size > 0:
                lidar_depth = float(np.median(lidar_vals))
                w_lidar, w_mono = 0.8, 0.2
                fused_depth = w_lidar * lidar_depth + w_mono * mono_depth
            else:
                fused_depth = mono_depth
        else:
            fused_depth = mono_depth
        rel_speed = max(ego_speed, 1e-3)
        ttc = np.nan
        if np.isfinite(fused_depth) and fused_depth > 0 and rel_speed > 0:
            ttc = fused_depth / rel_speed
        ttc_list.append(ttc)
    avg_ttc = np.nanmean(ttc_list) if len(ttc_list) > 0 else np.nan
    # Draw bboxes and show only averaged TTC
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        color = 'cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta'
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        # Only show averaged TTC for all boxes
        label = f"TTC (avg): {avg_ttc:.1f}s" if np.isfinite(avg_ttc) else "TTC: N/A"
        plt.text(x1, y1-10, label, color='black', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Color based on class
        if cls_idx in [1, 3]:  # Person or bike
            color = 'cyan' if cls_idx == 3 else 'lime'
        else:  # Vehicles
            color = 'magenta'
        
        # Draw box and label
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        plt.text(x1, y1-10, f"{CLASSES[cls_idx]} ({conf:.2f})", color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.title("Object Detection")
    plt.axis('off')
    
    # 3. Depth visualization with detection overlay
    ax3 = plt.subplot(133)
    # Show original image
    plt.imshow(frame_rgb)
    
    # Create depth overlay with transparency
    depth_norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
    depth_colored = plt.cm.magma(depth_norm(depth_map))
    depth_colored[..., 3] = 0.5  # Set alpha for transparency
    depth_img = plt.imshow(depth_colored)
    
    # Add detection boxes and depth values
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Get depth for this object
        box_depth = np.median(depth_map[y1:y2, x1:x2])
        
        # Color based on class
        if cls_idx in [1, 3]:  # Person or bike
            color = 'cyan' if cls_idx == 3 else 'lime'
        else:  # Vehicles
            color = 'magenta'
        
        # Draw box and label with depth
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        label = f"{CLASSES[cls_idx]}\nDepth: {box_depth:.1f}m"
        plt.text(x1, y1-10, label, color=color, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add colorbar with proper axes reference
    plt.colorbar(depth_img, ax=ax3, label='Depth (m)')
    plt.title("Depth Map with Detections")
    plt.axis('off')
    
    # Save the visualization
    depth_path = os.path.join(figures_dir, f"frame_{frame_number}_mono_depth_visualization.png")
    plt.tight_layout()
    # Fix: Cap figure size to avoid ValueError on savefig
    # --- Figure Saving Utility ---
    def safe_savefig(path, dpi=300, **kwargs):
        """Safely save figure by capping size to avoid matplotlib limits."""
        fig = plt.gcf()
        w_inches, h_inches = fig.get_size_inches()
        max_pixels = 65535
        w_pixels = w_inches * dpi
        h_pixels = h_inches * dpi
        if w_pixels > max_pixels or h_pixels > max_pixels:
            scale_w = max_pixels / w_pixels if w_pixels > max_pixels else 1.0
            scale_h = max_pixels / h_pixels if h_pixels > max_pixels else 1.0
            scale = min(scale_w, scale_h)
            new_w = w_inches * scale
            new_h = h_inches * scale
            print(f"Warning: Figure too large ({w_pixels:.0f}x{h_pixels:.0f}). Resizing to ({new_w*dpi:.0f}x{new_h*dpi:.0f})")
            fig.set_size_inches(new_w, new_h)
        plt.savefig(path, dpi=dpi, **kwargs)

    # --- Visualization routines ---
    def create_mono_depth_visualization(frame_rgb, depth_map, pred_bboxes, figures_dir, frame_number, dpi=300):
        """Create monocular depth visualization with proper size management."""
        plt.figure(figsize=(15, 8))  # Fixed reasonable size
        plt.subplot(121)
        plt.imshow(frame_rgb)
        plt.title("RGB Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(frame_rgb)
        depth_norm = plt.Normalize(vmin=depth_map.min(), vmax=depth_map.max())
        depth_colored = plt.cm.magma(depth_norm(depth_map))
        depth_colored[..., 3] = 0.5
        depth_img = plt.imshow(depth_colored)
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_idx = int(box[4])
            conf = float(box[5])
            box_depth = np.median(depth_map[y1:y2, x1:x2])
            color = 'cyan' if cls_idx == 3 else ('lime' if cls_idx == 1 else 'magenta')
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
            label = f"{CLASSES[cls_idx]}\nDepth: {box_depth:.1f}m"
            plt.text(x1, y1-10, label, color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        plt.colorbar(depth_img, label='Depth (m)', shrink=0.8)
        plt.title("Depth Map with Detections")
        plt.axis('off')
        plt.tight_layout()
        depth_path = os.path.join(figures_dir, f"frame_{frame_number}_mono_depth_visualization.png")
        safe_savefig(depth_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return depth_path

    # ...existing code...
    plt.close()
    print(f"Saved depth visualization: {depth_path}")
    
    # Original image
    plt.subplot(121)
    plt.imshow(frame_rgb)
    plt.title("RGB Image")
    plt.axis('off')
    
    # Original vs Calibrated Depth
    plt.subplot(122)
    
    if 'Dlidar' in locals():
        # Create scatter plot comparing LiDAR vs calibrated mono depth
        valid_points = Mlidar & np.isfinite(depth_map)
        if valid_points.sum() > 0:
            lidar_vals = Dlidar[valid_points]
            mono_vals = depth_map[valid_points]
            
            # Compute error metrics
            abs_rel_error = np.mean(np.abs(mono_vals - lidar_vals) / lidar_vals)
            rmse = np.sqrt(np.mean((mono_vals - lidar_vals) ** 2))
            
            # Plot depth map with metrics
            depth_plot = plt.imshow(depth_map, cmap='magma')
            plt.colorbar(depth_plot, label='Absolute Depth (m)')
            plt.title(f"MiDAS Absolute Depth with LiDAR for scaling")
        else:
            depth_plot = plt.imshow(depth_map, cmap='magma')
            plt.colorbar(depth_plot, label='Depth (m)')
            plt.title("Monocular Depth (No LiDAR Overlap)")
    else:
        depth_plot = plt.imshow(depth_map, cmap='magma')
        plt.colorbar(depth_plot, label='Depth (m)')
        plt.title("Monocular Depth (Uncalibrated)")
    
    plt.axis('off')
    
    # Save with tight layout
    depth_path = os.path.join(figures_dir, f"frame_{frame_number}_mono_depth.png")
    plt.tight_layout()
    plt.savefig(depth_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved monocular depth visualization: {depth_path}")

    # 5. Depth + Detection overlay with multiple depth sources
    print("\n5. Creating depth + detection overlay...")
    plt.figure(figsize=(12, 5))
    
    # Create depth visualization
    depth_vis = np.uint8(plt.cm.magma(plt.Normalize()(depth_map)) * 255)
    
    # Create fused depth with weighted confidence fusion
    if 'Dlidar' in locals():
        # First, scale mono depth to absolute using LiDAR overlap
        valid_mask = Mlidar & np.isfinite(depth_map) & (depth_map > 0)
        if valid_mask.sum() >= 50:
            scale = float(np.median(Dlidar[valid_mask] / (depth_map[valid_mask] + 1e-6)))
            if np.isfinite(scale) and scale > 0:
                depth_map = np.clip(depth_map * scale, 0.1, 80.0)
                print(f"Applied global scale to mono depth: {scale:.3f}")
        
        # Initialize confidence weights
        Wlidar = np.zeros_like(Dlidar)
        Wmono = np.ones_like(depth_map) * 0.3  # Base confidence for mono depth
        
        # Process each detected object for local scaling and confidence
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            box_mask = np.zeros_like(Mlidar, dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            
            # Check for LiDAR points in box
            lidar_in_box = box_mask & Mlidar
            if np.any(lidar_in_box):
                # High confidence for LiDAR points in detected objects
                Wlidar[lidar_in_box] = 0.9
                
                # Increase mono confidence near LiDAR points in same object
                mono_region = box_mask & ~Mlidar & np.isfinite(depth_map)
                if mono_region.any():
                    lidar_vals = Dlidar[lidar_in_box]
                    if lidar_vals.size > 0:
                        box_lidar_depth = np.median(lidar_vals)
                        mono_vals = depth_map[mono_region]
                        if mono_vals.size > 0:
                            # Local scaling for mono depth in object
                            local_scale = box_lidar_depth / (np.median(mono_vals) + 1e-6)
                            depth_map[mono_region] = depth_map[mono_region] * local_scale
                            # Higher confidence for scaled mono depth in objects
                            Wmono[mono_region] = 0.5
            
        # Add distance-based confidence decay for LiDAR
        lidar_conf = np.exp(-Dlidar / 50.0)  # Confidence decreases with distance
        Wlidar[Mlidar] *= lidar_conf[Mlidar]
        
        # Add distance-based confidence decay for mono
        mono_conf = np.exp(-depth_map / 30.0)  # Faster decay for mono
        Wmono *= mono_conf
        
        # Normalize weights
        Wsum = Wlidar + Wmono
        valid_weights = Wsum > 0
        if np.any(valid_weights):
            Wlidar[valid_weights] /= Wsum[valid_weights]
            Wmono[valid_weights] /= Wsum[valid_weights]
        
        # Compute fused depth using normalized weights
        fused_depth = np.zeros_like(depth_map)
        fused_depth += Wlidar * Dlidar
        fused_depth += Wmono * depth_map
        
        # Define valid mask
        fused_mask = Mlidar | np.isfinite(depth_map)
        
        # Ensure reasonable depth range
        fused_depth = np.clip(fused_depth, 0.1, 80.0)
    else:
        fused_depth = depth_map.copy()
        fused_mask = np.isfinite(depth_map)
    
    plt.subplot(121)
    plt.imshow(frame_rgb)
    for box in pred_bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cls_idx = int(box[4])
        conf = float(box[5])
        
        # Get depths from different sources
        box_mask = np.ones((y2-y1, x2-x1), dtype=bool)
        mono_vals = depth_map[y1:y2, x1:x2][box_mask]
        mono_depth = float(np.median(mono_vals[np.isfinite(mono_vals)])) if mono_vals.size > 0 else np.nan
        
        if 'Dlidar' in locals():
            lidar_vals = Dlidar[y1:y2, x1:x2][Mlidar[y1:y2, x1:x2]]
            lidar_depth = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan
            
            fused_vals = fused_depth[y1:y2, x1:x2][fused_mask[y1:y2, x1:x2]]
            fused_depth_val = float(np.median(fused_vals)) if fused_vals.size > 0 else np.nan
        else:
            lidar_depth = np.nan
            fused_depth_val = mono_depth
        
        # Color based on class following main.py conventions
        if cls_idx in [1, 3]:  # Person or bike
            color = 'cyan' if cls_idx == 3 else 'lime'
        else:  # Vehicles
            color = 'magenta'
        
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        
        # Create multi-line label with all depth measurements
        label_lines = [
            f"{CLASSES[cls_idx]} ({conf:.2f})",
            f"MDE: {mono_depth:.1f}m",
            f"LiDAR: {lidar_depth:.1f}m" if np.isfinite(lidar_depth) else "LiDAR: N/A",
            f"Fused: {fused_depth_val:.1f}m" if np.isfinite(fused_depth_val) else "Fused: N/A"
        ]
        
        # Position label above box with multiple lines
        y_text = y1 - 5
        for line in reversed(label_lines):
            plt.text(x1, y_text, line, color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            y_text -= 12
    
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
        
        # Get depths from all sources for this box
        box_mask = np.ones((y2-y1, x2-x1), dtype=bool)
        
        # MDE (Midas) depth
        mono_vals = depth_map[y1:y2, x1:x2][box_mask]
        mono_depth = float(np.median(mono_vals[np.isfinite(mono_vals)])) if mono_vals.size > 0 else np.nan
        
        # LiDAR depth if available
        if 'Dlidar' in locals():
            lidar_patch = Dlidar[y1:y2, x1:x2]
            mask_patch = Mlidar[y1:y2, x1:x2]
            lidar_vals = lidar_patch[mask_patch]
            lidar_depth = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan
        else:
            lidar_depth = np.nan
        
        # Fused depth
        if 'fused_depth' in locals():
            fused_vals = fused_depth[y1:y2, x1:x2]
            fused_vals = fused_vals[np.isfinite(fused_vals)]
            fused_depth_val = float(np.median(fused_vals)) if fused_vals.size > 0 else np.nan
        else:
            fused_depth_val = mono_depth
            
        # Set color based on class (following main.py convention)
        if cls_idx in [1, 3]:  # Person or bike/motorcycle
            color = 'cyan' if cls_idx == 3 else 'lime'
        else:  # Vehicles (car, bus, truck)
            color = 'magenta'
        
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                        fill=False, color=color, linewidth=2))
        
        # Create multi-line label with all depth measurements
        label_lines = [
            f"{CLASSES[cls_idx]} ({conf:.2f})",
            f"MDE: {mono_depth:.1f}m",
            f"LiDAR: {lidar_depth:.1f}m" if np.isfinite(lidar_depth) else "LiDAR: N/A",
            f"Fused: {fused_depth_val:.1f}m" if np.isfinite(fused_depth_val) else "Fused: N/A"
        ]
        
        # Position label above box with better visibility
        y_text = y1 - 5
        for line in reversed(label_lines):
            plt.text(x1, y_text, line, color=color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            y_text -= 12
    
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
            #os.path.join(project_root, "lidar_projection/project_lidar.py"),
            os.path.join(project_root, "diagnostic_tool.py"),
            "--pcd", lidar_pcd,
            "--cam_yaml", os.path.join(project_root, "optimized_calibration/camera_optimized.yaml"),
            "--ext_yaml", os.path.join(project_root, "optimized_calibration/extrinsics_optimized.yaml"),
            "--image", frame_path,
            "--out_npz", os.path.join(figures_dir, f"frame_{frame_number}_lidar.npz"),
            "--debug_overlay", os.path.join(figures_dir, f"frame_{frame_number}_lidar_overlay.png"),
            "--manual_ty", "1.2", "--manual_tz", "0.1", "--manual_pitch", "1.9", "--manual_yaw", "1.2",
            "--manual_cx", "-5", "--manual_cy", "0.9"
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
    
    # Create ECW region
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
    
    # Draw ECW region with more visible color
    plt.imshow(np.ma.masked_where(~ecw_mask, np.ones_like(ecw_mask)), 
              cmap='RdBu', alpha=0.3)  # Using RdBu colormap for better visibility
    
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
            ecw_y_pos = (y2 - top_y) / (bot_y - top_y)
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
        vertices = np.array([
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bot_right_x, bot_y],
            [bot_left_x, bot_y]
        ])
        plt.gca().add_patch(plt.Polygon(vertices, fill=False, color='g', linewidth=2))
        
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
    # --- Missed LiDAR Depth-Driven Hazard Detection in ECW Region ---
    # Only run if fused depth and ECW mask are available
    if os.path.exists(fused_path):
        fused_depth = np.load(fused_path)
        # Recreate ECW mask (same as above)
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
        ecw_mask_arr = np.zeros((H, W), dtype=np.uint8)
        pts = points.astype(np.int32)
        cv2.fillPoly(ecw_mask_arr, [pts], 1)
        ecw_mask = ecw_mask_arr.astype(bool)
    visualize_missed_hazards_ecw(frame_rgb, depth_map, pred_bboxes, fused_depth, figures_dir, frame_number)
        
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
    # Fill with more visible semi-transparent color
    plt.fill(points[:, 0], points[:, 1], 'lightblue', alpha=0.4)
    
    # Draw outline with more visible color
    plt.plot(points[:, 0], points[:, 1], 'blue', linewidth=2, label='ECW Region')
    plt.plot([points[-1, 0], points[0, 0]], [points[-1, 1], points[0, 1]], 'blue', linewidth=2)
    
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
            ecw_y_pos = (y2 - top_y) / (bot_y - top_y)
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
                ecw_y_pos = (y2 - top_y) / (bot_y - top_y)
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
        from matplotlib.path import Path
        vertices = np.array([
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bot_right_x, bot_y],
            [bot_left_x, bot_y]
        ])
        path = Path(vertices)
        patch = plt.Polygon(vertices, fill=False, color='g', linewidth=2)
        plt.gca().add_patch(patch)
        
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