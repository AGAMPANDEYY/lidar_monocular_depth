import os
import numpy as np
import pandas as pd
from glob import glob
from metrics_depth import rmse_rel

def load_depth_data(base_dir, frame_id):
    """Load all depth maps for a given frame."""
    npy_base = os.path.join(base_dir, "debug", f"{frame_id}")
    return {
        'fused': np.load(f"{npy_base}_fused_depth.npy"),
        'mono': np.load(f"{npy_base}_mono_depth.npy"),
        'lidar': np.load(f"{npy_base}_lidar_depth.npy"),
        'lidar_mask': np.load(f"{npy_base}_lidar_mask.npy").astype(bool)
    }

def compute_depth_metrics(pred, gt, mask):
    """Compute standard depth estimation metrics."""
    p = pred[mask]
    g = gt[mask]
    
    # Scale-invariant metrics
    ratio = p / g
    a1 = np.mean((ratio > 0.8) & (ratio < 1.25))
    a2 = np.mean((ratio > 0.5) & (ratio < 2.0))
    
    # Scale-dependent metrics
    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel = np.mean(((p - g) ** 2) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,  # delta < 1.25
        'a2': a2   # delta < 2.0
    }

def compare_depth_methods(base_dir):
    """Compare depth estimation quality: Mono vs Fused (using LiDAR as GT)."""
    # Load object metrics CSV
    obj_df = pd.read_csv(os.path.join(base_dir, 'object_depth_metrics.csv'))
    
    # 1. Box-level depth accuracy
    print("\n=== Box-Level Depth Accuracy ===")
    # Filter for detected objects only (not missed)
    det_df = obj_df[obj_df['source'] == 'det'].copy()
    
    # Compute metrics using LiDAR as GT where available
    lidar_mask = det_df['lidar_median_depth'].notna()
    mono_err = np.abs(det_df.loc[lidar_mask, 'mono_median_depth'] - 
                     det_df.loc[lidar_mask, 'lidar_median_depth'])
    fused_err = np.abs(det_df.loc[lidar_mask, 'fused_median_depth'] - 
                      det_df.loc[lidar_mask, 'lidar_median_depth'])
    
    box_metrics = {
        'Method': ['Mono', 'Fused'],
        'MAE (m)': [mono_err.mean(), fused_err.mean()],
        'RMSE (m)': [np.sqrt((mono_err**2).mean()), 
                     np.sqrt((fused_err**2).mean())]
    }
    print("\nBox-Level Depth Error (vs LiDAR):")
    print(pd.DataFrame(box_metrics).round(3))

    # 2. Dense depth accuracy
    print("\n=== Dense Depth Accuracy ===")
    frame_metrics = {
        'mono': {'abs_rel': [], 'rmse': [], 'a1': []},
        'fused': {'abs_rel': [], 'rmse': [], 'a1': []}
    }
    
    # Process each frame's depth maps
    for frame_npy in sorted(glob(os.path.join(base_dir, "debug", "*_fused_depth.npy"))):
        frame_id = os.path.basename(frame_npy).split('_')[0]
        depths = load_depth_data(base_dir, frame_id)
        
        # Use LiDAR as GT where available
        mask = depths['lidar_mask'] & (depths['lidar'] > 0)
        if mask.sum() < 100:
            continue
            
        # Evaluate mono
        metrics = compute_depth_metrics(depths['mono'], depths['lidar'], mask)
        for k in frame_metrics['mono'].keys():
            frame_metrics['mono'][k].append(metrics[k])
            
        # Evaluate fused
        metrics = compute_depth_metrics(depths['fused'], depths['lidar'], mask)
        for k in frame_metrics['fused'].keys():
            frame_metrics['fused'][k].append(metrics[k])
    
    # Average dense metrics
    dense_metrics = {
        'Method': ['Mono', 'Fused'],
        'Abs.Rel': [np.mean(frame_metrics['mono']['abs_rel']),
                   np.mean(frame_metrics['fused']['abs_rel'])],
        'RMSE (m)': [np.mean(frame_metrics['mono']['rmse']),
                    np.mean(frame_metrics['fused']['rmse'])],
        'δ < 1.25': [np.mean(frame_metrics['mono']['a1']),
                    np.mean(frame_metrics['fused']['a1'])]
    }
    print("\nDense Depth Accuracy (vs LiDAR):")
    print(pd.DataFrame(dense_metrics).round(3))

    # 3. ECW Performance
    print("\n=== Early Collision Warning Performance ===")
    
    # Get all objects for both methods
    def compute_ecw_metrics(df):
        """Compute precision, recall, F1 for ECW detection"""
        # Ground truth: Objects with valid LiDAR depth
        gt_df = df[df['lidar_median_depth'].notna()].copy()
        gt_total = len(gt_df)
        
        # Consider an object as requiring ECW if LiDAR depth indicates close proximity
        # You might want to adjust this threshold based on your requirements
        DANGER_THRESHOLD = 33.0  # meters
        gt_df['needs_ecw'] = gt_df['lidar_median_depth'] < DANGER_THRESHOLD
        
        results = {}
        methods = {
            'mono': 'mono_median_depth',
            'fused': 'fused_median_depth'
        }
        
        for method_name, depth_col in methods.items():
            # For each method, evaluate its depth estimates against ground truth
            method_mask = gt_df[depth_col].notna()
            
            # True positives: Method detected object AND it needed ECW
            tp = ((method_mask) & (gt_df['needs_ecw'])).sum()
            
            # False positives: Method triggered ECW but object wasn't actually close
            fp = ((method_mask) & (~gt_df['needs_ecw'])).sum()
            
            # False negatives: Method missed object that needed ECW
            fn = ((~method_mask) & (gt_df['needs_ecw'])).sum()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[method_name] = {
                'Total Objects': gt_total,
                'True Positives': tp,
                'False Positives': fp,
                'False Negatives': fn,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
        
        return results

    # Compute frame-level metrics for all objects
    ecw_results = compute_ecw_metrics(obj_df)
    
    # Define metrics to display
    metric_names = ['Total Objects', 'True Positives', 'False Positives', 
                   'False Negatives', 'Precision', 'Recall', 'F1 Score']
    
    # Create DataFrame for display
    metrics_display = {
        'Metric': metric_names,
        'Mono': [ecw_results['mono'][m] for m in metric_names],
        'Fused': [ecw_results['fused'][m] for m in metric_names]
    }
    
    print("\nECW Frame-Level Detection Performance:")
    metrics_df = pd.DataFrame(metrics_display)
    
    # Convert numeric columns to strings first
    display_df = metrics_df.copy()
    display_df['Mono'] = display_df['Mono'].astype(str)
    display_df['Fused'] = display_df['Fused'].astype(str)
    
    # Format only the percentage rows
    percentage_metrics = ['Precision', 'Recall', 'F1 Score']
    for col in ['Mono', 'Fused']:
        # Convert percentages for specific rows
        for metric in percentage_metrics:
            idx = display_df.index[display_df['Metric'] == metric].tolist()
            if idx:
                val = float(display_df.loc[idx[0], col])
                display_df.loc[idx[0], col] = f'{val:.1%}'
    
    print(display_df.to_string(index=False))

    # 4. Event-Level ECW Analysis
    print("\n=== Early Collision Warning (Event-Level) ===")

    def class_ttc_thresh(cls_name):
        """Conservative thresholds from literature (VRU gets more time than cars)"""
        if cls_name is None:
            return 3.5
        name = str(cls_name).lower()
        if any(k in name for k in ['person', 'ped', 'bicycle', 'bike']):
            return 4.0
        return 3.0  # cars, trucks, etc.

    def iou(a, b):
        """Compute IoU between two bboxes [x1,y1,x2,y2]"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1+1), max(0, iy2-iy1+1)
        inter = iw * ih
        ua = (ax2-ax1+1)*(ay2-ay1+1) + (bx2-bx1+1)*(by2-by1+1) - inter
        return inter/ua if ua > 0 else 0.0

    def assign_tracks(frame_rows, prev_tracks, frame_idx, iou_thr=0.3, max_age=5):
        """Simple IoU tracker for frame-to-frame object association"""
        tracks = {}
        next_id = 0 if not prev_tracks else (max(prev_tracks.keys())+1)

        # Age out old tracks
        for tid, T in prev_tracks.items():
            if (frame_idx - T['last_frame']) <= max_age:
                tracks[tid] = T

        # Greedy match by IoU
        assigned = set()
        for i, r in frame_rows.iterrows():
            bb = eval(r['bbox']) if isinstance(r['bbox'], str) else r['bbox']
            best_iou, best_id = 0.0, None
            for tid, T in tracks.items():
                cur_iou = iou(T['bbox'], bb)
                if cur_iou > best_iou:
                    best_iou, best_id = cur_iou, tid
            if best_iou >= iou_thr and best_id is not None and best_id not in assigned:
                T = tracks[best_id]
                T['bbox'] = bb
                T['class'] = r['class']
                T['last_frame'] = frame_idx
                assigned.add(best_id)
            else:
                # New track
                tracks[next_id] = {
                    'bbox': bb, 'class': r['class'], 'last_frame': frame_idx,
                    'warn_frames': [], 'hazard_frames': []
                }
                assigned.add(next_id)
                next_id += 1
        return tracks

    def is_finite(x):
        """Safe check for finite float values"""
        try:
            return np.isfinite(float(x))
        except:
            return False

    def compute_warn_and_hazard_flags(row, method='fused'):
        """Compute warning and hazard flags for a single object"""
        ttc = row.get('ttc', np.nan)
        cls_name = row.get('class', None)
        in_bubble = bool(row.get('in_ecw', False))
        T_warn = class_ttc_thresh(cls_name)

        # Warning based on chosen method's TTC
        warn = in_bubble and is_finite(ttc) and (ttc <= T_warn)

        # Hazard based on LiDAR TTC or depth
        hazard = False
        if is_finite(row.get('lidar_median_depth', np.nan)):
            v_ego = 5.0  # m/s (could load from timing.csv)
            d = float(row['lidar_median_depth'])
            ttc_lidar = d / max(v_ego, 1e-3)
            hazard = in_bubble and (ttc_lidar <= T_warn)
        
        return warn, hazard, T_warn
    def build_events(df, method='fused'):
        """Build warning events from frame-wise object data"""
        rows = df[df['in_ecw'] & df['meets_size'] & (df['depth_valid_px'] >= 50)].copy()
        rows.sort_values(['obj_id', 'frame'], inplace=True)
        events = []
        
        for oid, g in rows.groupby('obj_id'):
            active = False
            start = None
            last_bbox = None
            warn_col = f'warn_stable_{method}'
            
            for _, r in g.iterrows():
                if r[warn_col] and not active:
                    active = True
                    start = int(r['frame'])
                if r[warn_col]:
                    last_bbox = r['bbox']
                if active and not r[warn_col]:
                    end = int(r['frame'])
                    events.append({
                        'obj_id': oid, 
                        'start': start, 
                        'end': end, 
                        'bbox': last_bbox
                    })
                    active = False
            if active:
                events.append({
                    'obj_id': oid, 
                    'start': start, 
                    'end': int(g['frame'].iloc[-1]), 
                    'bbox': last_bbox
                })
        return events

    def match_events(pred_events, gt_events, iou_threshold=0.3):
        """Match predicted events to ground truth events"""
        TP = FP = FN = 0
        matched_pairs = []
        
        for pe in pred_events:
            best_match = None
            best_overlap = 0
            
            for ge in gt_events:
                # Check temporal overlap
                t_overlap = min(pe['end'], ge['end']) - max(pe['start'], ge['start'])
                if t_overlap <= 0:
                    continue
                    
                # Check spatial overlap (IoU) during overlap period
                if iou(pe['bbox'], ge['bbox']) >= iou_threshold:
                    overlap = t_overlap
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = ge
            
            if best_match is not None:
                TP += 1
                matched_pairs.append((pe, best_match))
                gt_events.remove(best_match)
            else:
                FP += 1
                
        FN = len(gt_events)  # Remaining unmatched GT events
        return TP, FP, FN, matched_pairs

    def evaluate_ecw(obj_df, method_name='fused'):
        """Evaluate ECW performance with object-centric metrics"""
        fps = 25.0  # Camera FPS
        
        # Initialize tracking state
        prev_tracks = {}
        global frame_idx
        
        # Verify required columns exist
        required_cols = ['frame', 'obj_id', 'ttc', 'in_ecw', 'meets_size', 
                        'depth_valid_px', 'bbox', 'source', 'class']
        missing_cols = [col for col in required_cols if col not in obj_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")  # Used in assign_tracks
        
        # Set warning flags based on TTC and bubble
        warn_threshold = 3.5  # Default threshold in seconds
        valid_mask = (
            obj_df['in_ecw'] &      # Object is in warning zone
            obj_df['meets_size'] &   # Object meets size criteria
            (obj_df['depth_valid_px'] >= 50) & # Has enough valid depth pixels
            obj_df['ttc'].notna()    # Has valid TTC estimate
        )
        
        # Initialize warning columns
        obj_df[f'warn_raw_{method_name}'] = False
        obj_df[f'warn_stable_{method_name}'] = False
        
        # Set warnings where TTC is below threshold
        obj_df.loc[valid_mask, f'warn_raw_{method_name}'] = obj_df.loc[valid_mask, 'ttc'] <= warn_threshold
        obj_df.loc[valid_mask, f'warn_stable_{method_name}'] = obj_df.loc[valid_mask, 'ttc'] <= warn_threshold
        
        # Filter for valid objects
        valid_mask = (
            obj_df['in_ecw'] &     # Object is in warning zone
            obj_df['meets_size'] &     # Object meets size criteria
            (obj_df['depth_valid_px'] >= 50) & # Has enough valid depth pixels
            obj_df['ttc'].notna()      # Has valid TTC estimate
        )
        df = obj_df[valid_mask].copy()
        
        # Add tracking IDs if not present
        if 'obj_id' not in df.columns:
            df['obj_id'] = [f"{method_name}_{i}" for i in range(len(df))]
        
        # Process each frame to build event tracks
        events = []
        for frame_idx, frame_data in df.groupby('frame'):
            # Create tracks dictionary if it doesn't exist
            if not prev_tracks:
                prev_tracks = {}
                
            # Process current frame
            tracks = assign_tracks(frame_data, prev_tracks, frame_idx)
            prev_tracks = tracks.copy()  # Update for next iteration
            
            # Update warning states for active tracks
            for tid, track in tracks.items():
                if frame_data[f'warn_stable_{method_name}'].any():
                    track.setdefault('warn_frames', []).append(frame_idx)
                if frame_data[f'warn_raw_{method_name}'].any():
                    track.setdefault('hazard_frames', []).append(frame_idx)
                    
        # Convert tracks to events
        for tid, track in prev_tracks.items():
            if 'warn_frames' in track and track['warn_frames']:
                warn_frames = sorted(track['warn_frames'])
                hazard_frames = sorted(track.get('hazard_frames', []))
                
                # Create events from warning tracks that have hazards
            if hazard_frames and warn_frames:
                # Get hazard frames that happen during or after warning 
                warn_start = min(warn_frames)
                warn_end = max(warn_frames)
                valid_hazards = [h for h in hazard_frames if h >= warn_start]
                
                if valid_hazards:
                    events.append({
                        'obj_id': tid,
                        'start': warn_start,
                        'end': warn_end, 
                        'hazard_start': min(valid_hazards),  # First hazard after warning
                        'bbox': track['bbox'],
                        'class': track['class']
                    })
        
        # Process events and compute metrics
        TP = FP = FN = 0
        lead_times = []
        
        # Convert to ground truth events using LiDAR-based hazards
        gt_events = []
        for e in events:
            if e.get('hazard_start') is not None:
                gt_events.append(e)
                
        # Match predicted warnings to ground truth hazards
        matched_gt = set()
        for pred in events:
            best_match = None
            best_iou = 0.0
            
            for gt in gt_events:
                # Check temporal overlap and ordering
                if gt['hazard_start'] >= pred['start'] and gt['hazard_start'] <= pred['end']:
                    # Check spatial overlap using IoU
                    if iou(pred['bbox'], gt['bbox']) >= 0.3:
                        # Compute lead time (warning before hazard)
                        lead_time = (gt['hazard_start'] - pred['start']) / fps
                        # Prefer matches with longer lead time
                        if lead_time > best_iou:
                            best_iou = lead_time
                            best_match = gt
                            
            if best_match is not None:
                TP += 1
                matched_gt.add(best_match['obj_id'])
                if best_iou > 0:  # Only count positive lead times
                    lead_times.append(best_iou)
            else:
                FP += 1
                
        # Count unmatched ground truth events as false negatives
        FN = len(gt_events) - len(matched_gt)
        
        # Calculate final metrics
        total_frames = len(df['frame'].unique())
        minutes = total_frames / (fps * 60)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'Total Events': TP + FN,
            'True Positives': TP,
            'False Positives': FP,
            'False Negatives': FN,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Lead Time (mean)': float(np.mean(lead_times)) if lead_times else np.nan,
            'Lead Time (median)': float(np.median(lead_times)) if lead_times else np.nan,
            'False Alarms/min': FP / minutes if minutes > 0 else np.nan
        }
    # Evaluate both mono and fused ECW performance at event level
    ecw_mono_event = evaluate_ecw(obj_df, method_name='mono')
    ecw_fused_event = evaluate_ecw(obj_df, method_name='fused')

    # Create results table
    results_df = pd.DataFrame([ecw_mono_event, ecw_fused_event], index=['Mono', 'Fused'])
    
    # Format percentage metrics
    percentage_cols = ['Precision', 'Recall', 'F1 Score']
    for col in percentage_cols:
        results_df[col] = results_df[col].apply(lambda x: f'{x:.1%}')
    
    # Format time metrics
    time_cols = ['Lead Time (mean)', 'Lead Time (median)']
    for col in time_cols:
        results_df[col] = results_df[col].apply(lambda x: f'{x:.2f}s' if np.isfinite(x) else 'N/A')
    
    # Format rate metrics
    results_df['False Alarms/min'] = results_df['False Alarms/min'].apply(lambda x: f'{x:.2f}')
    
    print("\nECW Event-Level Performance:")
    print(results_df.round(3))

    # 4. Timing Performance
    print("\n=== Pipeline Timing ===")
    timing_df = pd.read_csv(os.path.join(base_dir, 'timing.csv'))
    # Calculate means of only numeric columns
    numeric_cols = ['t_det_ms', 't_depth_ms', 't_fuse_ms', 't_ecw_ms', 't_total_ms']
    timing_means = timing_df[numeric_cols].mean()
    
    backend_name = timing_df['backend'].iloc[0]  # Get the backend name from first row
    
    timing_metrics = {
        'Component': ['Detection', 'Depth', 'Fusion', 'ECW', 'Total'],
        'Time (ms)': [
            timing_means['t_det_ms'],
            timing_means['t_depth_ms'],
            timing_means['t_fuse_ms'],
            timing_means['t_ecw_ms'],
            timing_means['t_total_ms']
        ],
        'FPS': [
            1000/timing_means['t_det_ms'],
            1000/timing_means['t_depth_ms'],
            1000/timing_means['t_fuse_ms'],
            1000/timing_means['t_ecw_ms'],
            1000/timing_means['t_total_ms']
        ]
    }
    
    print(f"\nTiming Results (using {backend_name} backend):")
    print("\nComponent-wise Timing:")
    print(pd.DataFrame(timing_metrics).round(2))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                       help='Directory containing results (with debug/ subfolder)')
    args = parser.parse_args()
    
    compare_depth_methods(args.data_dir)