import os
import numpy as np
from typing import Dict, List

# Small constant to avoid division by zero
EPS = 1e-6

def load_depth_data(base_dir: str, frame_id: str) -> Dict:
    """Load depth data for a given frame."""
    lidar_path = os.path.join(base_dir, 'lidar', f'{frame_id}_lidar.npy')
    mono_path = os.path.join(base_dir, 'mono', f'{frame_id}_depth.npy')
    fused_path = os.path.join(base_dir, 'fused', f'{frame_id}_depth.npy')
    mask_path = os.path.join(base_dir, 'lidar', f'{frame_id}_mask.npy')
    
    return {
        'lidar': np.load(lidar_path),
        'mono': np.load(mono_path),
        'fused': np.load(fused_path),
        'lidar_mask': np.load(mask_path)
    }

def process_dense_metrics(range_bins: List[str], frames_to_process: List[str], base_dir: str) -> List[Dict]:
    """Process dense depth metrics with confidence intervals."""
    from .metrics_with_ci import compute_metrics_with_ci
    
    # Initialize storage
    range_data = {rb: {'mono': [], 'fused': [], 'gt': []} for rb in range_bins}
    valid_frame_count = 0
    total_valid_points = 0
    dense_results = []
    
    for f in frames_to_process:
        frame_id = os.path.basename(f).split('_')[0]
        try:
            d = load_depth_data(base_dir, frame_id)
            
            # Create validity mask
            valid_mask = (d['lidar_mask'] & 
                        (d['lidar'] > EPS) & np.isfinite(d['lidar']) &
                        (d['mono'] > EPS) & np.isfinite(d['mono']) &
                        (d['fused'] > EPS) & np.isfinite(d['fused']))
            
            n_valid = valid_mask.sum()
            if n_valid < 100:
                continue
                
            # Extract valid depths
            gt_depths = d['lidar'][valid_mask]
            mono_depths = d['mono'][valid_mask]
            fused_depths = d['fused'][valid_mask]
            
            # Add to appropriate range bins
            for rb in range_bins:
                if rb == '0-10m':
                    mask = gt_depths <= 10
                elif rb == '10-25m':
                    mask = (gt_depths > 10) & (gt_depths <= 25)
                elif rb == '25-50m':
                    mask = (gt_depths > 25) & (gt_depths <= 50)
                else:  # >50m
                    mask = gt_depths > 50
                
                if mask.sum() > 0:
                    range_data[rb]['gt'].extend(gt_depths[mask])
                    range_data[rb]['mono'].extend(mono_depths[mask])
                    range_data[rb]['fused'].extend(fused_depths[mask])
            
            valid_frame_count += 1
            total_valid_points += n_valid
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {str(e)}")
            continue
    
    print(f"\nProcessed {valid_frame_count} frames with {total_valid_points} total valid points")
    
    # Compute metrics for each range bin
    for rb in range_bins:
        for method in ['mono', 'fused']:
            pred = np.array(range_data[rb][method])
            gt = np.array(range_data[rb]['gt'])
            
            if len(pred) > 100:
                metrics = compute_metrics_with_ci(pred, gt)
                ci_abs_rel = (metrics['abs_rel_ci'][1] - metrics['abs_rel_ci'][0]) / 2
                ci_rmse = (metrics['rmse_ci'][1] - metrics['rmse_ci'][0]) / 2
                
                dense_results.append({
                    'Method': method.capitalize(),
                    'Range': rb,
                    'AbsRel': f"{metrics['abs_rel_mean']:.3f} ± {ci_abs_rel:.3f}",
                    'RMSE': f"{metrics['rmse_mean']:.3f} ± {ci_rmse:.3f}",
                    'N': len(pred)
                })
            else:
                print(f"WARNING: Insufficient points for {method} at {rb} (N={len(pred)})")
    
    return dense_results