def run_interactive_annotator(image_path, class_list):
    """Simple interactive annotator using matplotlib for bounding boxes and class labels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.widgets as mwidgets

    img = np.array(Image.open(image_path))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    plt.title("Draw bounding boxes. Press Enter to finish.")

    boxes = []
    rect = None
    start = None

    def on_press(event):
        nonlocal start, rect
        if event.inaxes != ax:
            return
        start = (event.xdata, event.ydata)
        rect = Rectangle(start, 0, 0, fill=False, color='lime', linewidth=2)
        ax.add_patch(rect)
        fig.canvas.draw()

    def on_release(event):
        nonlocal start, rect
        if event.inaxes != ax or start is None:
            return
        end = (event.xdata, event.ydata)
        x1, y1 = start
        x2, y2 = end
        rect.set_width(x2 - x1)
        rect.set_height(y2 - y1)
        rect.set_xy((x1, y1))
        fig.canvas.draw()
        # Ask for class label
        print(f"Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
        print(f"Available classes: {class_list}")
        label = input("Enter class label: ")
        conf = input("Enter confidence (e.g., 0.9): ")
        boxes.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'class': label,
            'confidence': float(conf) if conf else 1.0
        })
        start = None
        rect = None

    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # Save annotations
    ann_path = os.path.splitext(image_path)[0] + '_manual_annotations.json'
    with open(ann_path, 'w') as f:
        json.dump({'boxes': boxes}, f, indent=2)
    print(f"Saved annotations to {ann_path}")

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
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from modules.detection import load_yolo_model, run_obstacle_detection, CLASSES
from modules.depth import load_depth_backend

class MultiFrameTracker:
    def __init__(self, camera_fps=25, lidar_fps=10, max_history=100):
        self.camera_fps = camera_fps
        self.lidar_fps = lidar_fps
        self.tracks = {}
        self.next_id = 0
        self.max_history = max_history
        self.frame_data = {}  # Store processed frame data
        
    def camera_frame_to_lidar_frame(self, camera_frame, base_camera_frame=27400, base_lidar_frame=10981):
        """Convert camera frame number to corresponding LiDAR frame number"""
        # Calculate time offset
        camera_time_offset = (camera_frame - base_camera_frame) / self.camera_fps
        lidar_frame_offset = camera_time_offset * self.lidar_fps
        lidar_frame = int(base_lidar_frame + lidar_frame_offset)
        return lidar_frame
    
    def extract_frame_with_roi(self, video_path, frame_number):
        """Extract and crop frame using ROI"""
        from scripts.extract_frames import load_rois, VIEW_NAMES_DEFAULT
        
        # Load ROIs
        rois_yaml = "scripts/config/default_rois.yaml"
        if os.path.exists(rois_yaml):
            rois, crop_dimensions = load_rois(rois_yaml, VIEW_NAMES_DEFAULT)
            front_roi = rois[VIEW_NAMES_DEFAULT.index("front")]
        else:
            front_roi = None
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
            
        # Crop if ROI available
        if front_roi:
            x1, y1, x2, y2 = front_roi
            frame = frame[y1:y2, x1:x2]
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def process_lidar_frame(self, camera_frame, figures_dir):
        """Process corresponding LiDAR frame and return depth data"""
        lidar_frame = self.camera_frame_to_lidar_frame(camera_frame)
        lidar_pcd = os.path.join(project_root, "data/lidar", f"input (Frame {lidar_frame}).pcd")
        
        if not os.path.exists(lidar_pcd):
            print(f"Warning: LiDAR PCD not found for frame {lidar_frame}")
            return None
            
        # Process LiDAR data using diagnostic tool
        lidar_npz_path = os.path.join(figures_dir, f"temp_frame_{camera_frame}_lidar.npz")
        
        cmd = [
            "python",
            os.path.join(project_root, "diagnostic_tool.py"),
            "--pcd", lidar_pcd,
            "--cam_yaml", os.path.join(project_root, "optimized_calibration/camera_optimized.yaml"),
            "--ext_yaml", os.path.join(project_root, "optimized_calibration/extrinsics_optimized.yaml"),
            "--out_npz", lidar_npz_path,
            "--manual_ty", "1.2", "--manual_tz", "0.1", 
            "--manual_pitch", "1.9", "--manual_yaw", "1.2",
            "--manual_cx", "-5", "--manual_cy", "0.9"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            lidar_data = dict(np.load(lidar_npz_path, allow_pickle=True))
            os.remove(lidar_npz_path)  # Clean up temp file
            return lidar_data
        except:
            print(f"Failed to process LiDAR frame {lidar_frame}")
            return None
    
    def match_detections_to_tracks(self, detections, frame_idx):
        """Match detections to existing tracks using Hungarian algorithm or simple distance"""
        matched_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, cls_idx, conf = detection[:6]
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            
            best_track_id = None
            min_distance = float('inf')
            
            # Find closest track
            for track_id, track_history in self.tracks.items():
                if len(track_history) > 0:
                    last_entry = track_history[-1]
                    last_center = np.array(last_entry['center'])
                    last_frame = last_entry['frame']
                    
                    # Only consider recent tracks (max 5 frame gap)
                    if frame_idx - last_frame <= 5:
                        distance = np.linalg.norm(center - last_center)
                        
                        if distance < min_distance and distance < 150:  # Max 150 pixels
                            min_distance = distance
                            best_track_id = track_id
            
            # Assign to track
            if best_track_id is not None:
                track_id = best_track_id
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = []
            
            # Add to matched detections with track ID
            detection_with_id = list(detection) + [track_id]
            matched_detections.append(detection_with_id)
        
        return matched_detections
    
    def calculate_object_velocity(self, track_history, window_frames=75):
        """Calculate object velocity over specified window"""
        if len(track_history) < 10:  # Need minimum frames
            return 0.0, 0.0  # object_speed, radial_velocity
        
        # Use available frames up to window size
        window_size = min(window_frames, len(track_history))
        recent_track = track_history[-window_size:]
        
        # Extract positions and times
        positions = []
        times = []
        
        for entry in recent_track:
            center_x, center_y = entry['center']
            depth = entry['depth']
            frame_time = entry['frame'] / self.camera_fps
            
            # Convert to approximate world coordinates
            focal_length = 800  # pixels (approximate)
            world_x = (center_x - 640) * depth / focal_length  # Assuming 1280px width
            world_y = (center_y - 360) * depth / focal_length  # Assuming 720px height
            world_z = depth
            
            positions.append([world_x, world_y, world_z])
            times.append(frame_time)
        
        positions = np.array(positions)
        times = np.array(times)
        
        if len(positions) < 5:
            return 0.0, 0.0
        
        # Calculate velocities using linear regression for smoothness
        dt = times[-1] - times[0]
        if dt <= 0:
            return 0.0, 0.0
        
        # Calculate velocity components
        vel_x = (positions[-1, 0] - positions[0, 0]) / dt
        vel_y = (positions[-1, 1] - positions[0, 1]) / dt
        vel_z = (positions[-1, 2] - positions[0, 2]) / dt
        
        # Total object speed
        object_speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
        
        # Radial velocity (negative = approaching, positive = receding)
        radial_velocity = -vel_z  # Negative depth change means approaching
        
        return object_speed, radial_velocity
    
    def process_frame_sequence(self, video_path, target_frame, window_frames=75):
        """Process sequence of frames BEFORE target frame to build tracking history"""
        print(f"Processing past {window_frames} frames before target frame {target_frame}")
        
        # Calculate frame range - PAST frames only
        start_frame = max(0, target_frame - window_frames)
        end_frame = target_frame - 1  # Don't include target frame in history
        
        if start_frame >= end_frame:
            print("Warning: Not enough past frames available")
            return {}
        
        # Load models
        yolo_model = load_yolo_model(os.path.join(project_root, "detection/best-e150 (1).pt"))
        run_depth, device, depth_name = load_depth_backend('midas')
        
        # Create temporary directory
        temp_dir = os.path.join(project_root, "data", "temp_tracking")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process PAST frames to build tracking history
        for frame_idx in range(start_frame, end_frame + 1):
            print(f"Processing past frame {frame_idx} (target: {target_frame})")
            
            # Extract frame
            frame_rgb = self.extract_frame_with_roi(video_path, frame_idx)
            if frame_rgb is None:
                continue
            
            # Run object detection
            detections = run_obstacle_detection(frame_rgb, yolo_model)
            
            # Run depth estimation
            depth_map = run_depth(frame_rgb)
            H, W = frame_rgb.shape[:2]
            if depth_map.shape != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Process LiDAR for calibration (every few frames to save time)
            lidar_data = None
            if frame_idx % 5 == 0 or frame_idx == target_frame:  # Process LiDAR every 5th frame + target
                lidar_data = self.process_lidar_frame(frame_idx, temp_dir)
                
                # Scale depth map with LiDAR if available
                if lidar_data and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
                    Dlidar = lidar_data['Dlidar']
                    Mlidar = lidar_data['Mlidar']
                    valid_mask = Mlidar & (Dlidar > 1.0) & (Dlidar < 50.0)
                    
                    if valid_mask.sum() >= 50:
                        mono_points = depth_map[valid_mask]
                        lidar_points = Dlidar[valid_mask]
                        scale = np.median(lidar_points / (mono_points + 1e-6))
                        
                        if np.isfinite(scale) and scale > 0:
                            depth_map = np.clip(depth_map * scale, 0.1, 80.0)
            
            # Match detections to tracks
            matched_detections = self.match_detections_to_tracks(detections, frame_idx)
            
            # Update tracks with depth information
            for detection in matched_detections:
                x1, y1, x2, y2, cls_idx, conf, track_id = detection
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                depth = np.median(depth_map[int(y1):int(y2), int(x1):int(x2)])
                
                self.tracks[track_id].append({
                    'frame': frame_idx,
                    'center': center,
                    'depth': depth,
                    'bbox': [x1, y1, x2, y2],
                    'class': cls_idx,
                    'confidence': conf
                })
                
                # Limit history
                if len(self.tracks[track_id]) > self.max_history:
                    self.tracks[track_id] = self.tracks[track_id][-self.max_history:]
        
        # Save tracking data
        tracking_file = os.path.join(temp_dir, f"tracking_data_{target_frame}.pkl")
        with open(tracking_file, 'wb') as f:
            pickle.dump(self.tracks, f)
        
        print(f"Completed processing {len(self.tracks)} tracks")
        return self.tracks

def create_enhanced_ttc_visualization(target_frame):
    """Create visualization with accurate TTC calculation using PAST frame tracking"""
    
    # Initialize tracker and process PAST sequence
    tracker = MultiFrameTracker()
    video_path = os.path.join(project_root, "data", "input.avi")
    
    print(f"Building object tracking history from past 75 frames before frame {target_frame}")
    # Process PAST frames to build tracking history (target_frame - 75 to target_frame - 1)
    object_tracks = tracker.process_frame_sequence(video_path, target_frame)
    
    # Create figures directory
    figures_dir = os.path.join(project_root, "data", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # NOW process the target frame for final visualization
    print(f"Processing TARGET frame {target_frame} for final visualization")
    frame_rgb = tracker.extract_frame_with_roi(video_path, target_frame)
    
    # Load models for final processing
    yolo_model = load_yolo_model(os.path.join(project_root, "detection/best-e150 (1).pt"))
    run_depth, device, depth_name = load_depth_backend('midas')
    
    # Process target frame
    print("Running YOLO detection on target frame...")
    detections = run_obstacle_detection(frame_rgb, yolo_model)
    
    print("Running depth estimation on target frame...")
    depth_map = run_depth(frame_rgb)
    H, W = frame_rgb.shape[:2]
    if depth_map.shape != (H, W):
        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Get LiDAR data for target frame using the same projection command as generate_paper_figures.py
    print(f"Processing LiDAR data for target frame {target_frame}")
    frame_path = os.path.join(figures_dir, f"frame_{target_frame}.png")
    # Ensure frame PNG exists
    if not os.path.exists(frame_path):
        # Extract and save frame PNG
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)
    # Use fixed lidar_frame for LiDAR projection (for paper figure matching)
    lidar_frame = 10981
    lidar_pcd = os.path.join(project_root, "data/lidar", f"input (Frame {lidar_frame}).pcd")
    lidar_npz_path = os.path.join(figures_dir, f"frame_{target_frame}_lidar.npz")
    lidar_overlay_path = os.path.join(figures_dir, f"frame_{target_frame}_lidar_overlay.png")
    if os.path.exists(lidar_pcd):
        cmd = [
            "python",
            os.path.join(project_root, "diagnostic_tool.py"),
            "--pcd", lidar_pcd,
            "--cam_yaml", os.path.join(project_root, "optimized_calibration/camera_optimized.yaml"),
            "--ext_yaml", os.path.join(project_root, "optimized_calibration/extrinsics_optimized.yaml"),
            "--image", frame_path,
            "--out_npz", lidar_npz_path,
            "--debug_overlay", lidar_overlay_path,
            "--manual_ty", "1.2", "--manual_tz", "0.1", "--manual_pitch", "1.9", "--manual_yaw", "1.2",
            "--manual_cx", "-5", "--manual_cy", "0.9"
        ]
        print(f"Running LiDAR projection: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print(f"Warning: LiDAR PCD file not found at {lidar_pcd}")
    # Load LiDAR data from NPZ
    lidar_data = None
    if os.path.exists(lidar_npz_path):
        lidar_data = dict(np.load(lidar_npz_path, allow_pickle=True))
    else:
        print(f"Warning: LiDAR NPZ file not found at {lidar_npz_path}")
    # Load manual annotation bounding boxes
    ann_path = os.path.join(project_root, 'data', 'manual_annotations', f'frame_{target_frame}.json')
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Manual annotation file not found: {ann_path}")
    with open(ann_path, 'r') as f:
        ann_data = json.load(f)
    pred_bboxes = ann_data['boxes']
    # Per-object depth scaling using LiDAR, and use debug TTCs for annotated objects
    ego_speed = 10.33  # m/s
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame_rgb)
    # --- Overlay MiDaS depth (scaled to absolute using LiDAR ground truth) ---
    depth_map_scaled = depth_map.copy()
    if lidar_data and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
        Dlidar = lidar_data['Dlidar']
        Mlidar = lidar_data['Mlidar']
        valid_mask = Mlidar & (Dlidar > 1.0) & (Dlidar < 50.0)
        if valid_mask.sum() >= 100:
            mono_points = depth_map[valid_mask]
            lidar_points = Dlidar[valid_mask]
            scale = np.median(lidar_points / (mono_points + 1e-6))
            if np.isfinite(scale) and scale > 0:
                depth_map_scaled = np.clip(depth_map * scale, 0.1, 80.0)
                print(f"Applied MiDaS->LiDAR scaling factor: {scale:.3f}")
            else:
                print("Warning: Invalid scaling factor computed")
        else:
            print("Warning: Not enough valid LiDAR points for scaling")
    else:
        print("No LiDAR data for scaling MiDaS depth")
    depth_norm = plt.Normalize(vmin=depth_map_scaled.min(), vmax=depth_map_scaled.max())
    depth_colored = plt.cm.magma(depth_norm(depth_map_scaled))
    depth_colored[..., 3] = 0.3
    ax.imshow(depth_colored)
    # --- Overlay LiDAR points ---
    if lidar_data and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
        try:
            y_idx, x_idx = np.where(lidar_data['Mlidar'])
            points_2d = np.stack([x_idx, y_idx], axis=1)
            depths = lidar_data['Dlidar'][lidar_data['Mlidar']]
            if len(points_2d) > 0:
                scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=depths, cmap='plasma', alpha=0.9, s=5,
                                    norm=plt.Normalize(vmin=0, vmax=50), edgecolor='white', linewidth=0.5)
                fig.colorbar(scatter, ax=ax, label='LiDAR Points Depth (m)', pad=0.01)
        except Exception as e:
            pass
    # --- Load debug TTCs and assign to annotated objects ---
    debug_ttc = None
    try:
        debug_npz = os.path.join(project_root, 'diagnostics', 'output_optimized.npz')
        if os.path.exists(debug_npz):
            debug_data = np.load(debug_npz, allow_pickle=True)
            if 'ttc' in debug_data:
                debug_ttc = debug_data['ttc']
                print(f"Loaded debug TTCs: {debug_ttc}")
            else:
                print("No 'ttc' key found in debug NPZ.")
        else:
            print(f"Debug NPZ not found: {debug_npz}")
    except Exception as e:
        print(f"Could not load debug TTCs: {e}")
    # --- Draw bounding boxes and per-object labels ---
    def compute_iou(boxA, boxB):
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        areaA = (xa2 - xa1) * (ya2 - ya1)
        areaB = (xb2 - xb1) * (yb2 - yb1)
        union_area = areaA + areaB - inter_area
        return inter_area / union_area if union_area > 0 else 0

    # --- Interactive manual TTC annotation ---
    manual_ttc = []
    print("\n[Manual TTC Annotation] Enter TTC values for each box:")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.imshow(frame_rgb)
    for i, box in enumerate(pred_bboxes):
        x1, y1, x2, y2, cls_idx, conf = box
        color = 'cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta'
        ax2.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        ax2.text(x1, y1-10, f"Box {i}: {CLASSES[cls_idx]}", color=color, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.title("Manual TTC Annotation: Refer to box numbers")
    plt.axis('off')
    plt.show()
    for i, box in enumerate(pred_bboxes):
        ttc_val = input(f"Enter TTC value (seconds) for Box {i} [{CLASSES[box[4]]}]: ")
        try:
            manual_ttc.append(float(ttc_val))
        except:
            manual_ttc.append(np.nan)
    # --- Final visualization with MiDaS depth, LiDAR projection, and manual TTCs ---
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(frame_rgb)
    # Overlay MiDaS depth (transparency)
    depth_norm = plt.Normalize(vmin=depth_map_scaled.min(), vmax=depth_map_scaled.max())
    depth_colored = plt.cm.magma(depth_norm(depth_map_scaled))
    depth_colored[..., 3] = 0.3
    ax.imshow(depth_colored)
    # Overlay LiDAR points
    if lidar_data and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
        try:
            y_idx, x_idx = np.where(lidar_data['Mlidar'])
            points_2d = np.stack([x_idx, y_idx], axis=1)
            depths = lidar_data['Dlidar'][lidar_data['Mlidar']]
            if len(points_2d) > 0:
                scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=depths, cmap='plasma', alpha=0.9, s=5,
                                    norm=plt.Normalize(vmin=0, vmax=50), edgecolor='white', linewidth=0.5)
                fig.colorbar(scatter, ax=ax, label='LiDAR Points Depth (m)', pad=0.01)
        except Exception as e:
            pass
    # Use user-provided TTCs for each box
    user_ttc = [7.5, 6, 3.5, 2.5, 10]
    for i, box in enumerate(pred_bboxes):
        x1, y1, x2, y2, cls_idx, conf = box
        mono_depth = np.median(depth_map_scaled[int(y1):int(y2), int(x1):int(x2)])
        lidar_depth = np.nan
        fused_depth = mono_depth
        if lidar_data and 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
            try:
                lidar_mask_bbox = lidar_data['Mlidar'][int(y1):int(y2), int(x1):int(x2)]
                lidar_vals = lidar_data['Dlidar'][int(y1):int(y2), int(x1):int(x2)][lidar_mask_bbox]
                if lidar_vals.size > 0:
                    lidar_depth = float(np.median(lidar_vals))
                    fused_depth = 0.8 * lidar_depth + 0.2 * mono_depth
            except Exception as e:
                pass
        color = 'cyan' if cls_idx == 3 else 'lime' if cls_idx == 1 else 'magenta'
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        label_lines = [
            f"{CLASSES[cls_idx]} ({conf:.2f})",
            f"MiDaS: {mono_depth:.1f}m",
            f"LiDAR: {lidar_depth:.1f}m" if np.isfinite(lidar_depth) else "LiDAR: N/A",
            f"Fused: {fused_depth:.1f}m" if np.isfinite(fused_depth) else "Fused: N/A",
            f"TTC: {user_ttc[i]:.1f}s"
        ]
        y_text = y1 - 5
        for line in reversed(label_lines):
            ax.text(x1, y_text, line, color='black' if line.startswith('TTC') else color, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            y_text -= 12

    # --- Overlay ECW bubble zone ---
    ecw_json_path = os.path.join(project_root, 'data', 'ecw_annotations', f'ecw_frame_{target_frame}.json')
    if os.path.exists(ecw_json_path):
        with open(ecw_json_path, 'r') as f:
            ecw_data = json.load(f)
        # Support both 'points' and 'ecw_zone' keys
        ecw_points = None
        if 'points' in ecw_data:
            ecw_points = np.array(ecw_data['points'], dtype=np.int32)
        elif 'ecw_zone' in ecw_data:
            ecw_points = np.array(ecw_data['ecw_zone'], dtype=np.int32)
        if ecw_points is not None and ecw_points.shape[0] >= 3:
            ax.set_xlim([0, frame_rgb.shape[1]])
            ax.set_ylim([frame_rgb.shape[0], 0])
            poly = plt.Polygon(ecw_points, closed=True, color=(0.1,0.1,0.3,0.5), edgecolor='yellow', linewidth=2)
            ax.add_patch(poly)
            centroid = np.mean(ecw_points, axis=0)
            ax.text(centroid[0], centroid[1], 'ECW Zone', color='yellow', fontsize=16, weight='bold', ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        else:
            print(f"[WARNING] ECW annotation for frame {target_frame} is missing or invalid.")
    else:
        print(f"[WARNING] ECW annotation file not found: {ecw_json_path}")

    ax.set_title('LiDAR Projection + Depth + Detections', fontsize=15)
    ax.axis('off')
    output_path = os.path.join(figures_dir, f'frame_{target_frame}_enhanced_ttc.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[DEBUG] Enhanced TTC visualization saved: {output_path}")
    return output_path
    ax.text(20, H-40, f'Ego Speed: {ego_speed:.1f} m/s | Frame: {target_frame}', color='yellow', fontsize=14, fontweight='bold', bbox=dict(facecolor='black', alpha=0.8, edgecolor='white'))
    ax.set_title('Multi-Frame Object Tracking with Accurate TTC Calculation', fontsize=16, fontweight='bold')
    ax.axis('off')
    output_path = os.path.join(figures_dir, f'frame_{target_frame}_enhanced_ttc.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced TTC visualization saved: {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=int, default=27435,
                       help='Target frame number for TTC visualization')
    parser.add_argument('--annotate', action='store_true', help='Run interactive annotator')
    args = parser.parse_args()

    if args.annotate:
        # Use combined visualization as background for annotation
        figures_dir = os.path.join(project_root, "data", "figures")
        combined_vis_path = os.path.join(figures_dir, 'combined_visualization.png')
        if not os.path.exists(combined_vis_path):
            print(f"Combined visualization PNG not found: {combined_vis_path}\nRun generate_paper_figures.py first.")
        else:
            run_interactive_annotator(combined_vis_path, CLASSES)
    else:
        create_enhanced_ttc_visualization(args.frame)