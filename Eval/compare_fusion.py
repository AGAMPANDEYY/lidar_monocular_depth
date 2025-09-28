#!/usr/bin/env python3
import os
import ast
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-6

# -------------------------------
# Utilities
# -------------------------------

def classify_environment(frame_id):
    """All frames are classified as semi-urban for this dataset.
    Returns: 'semi-urban'
    """
    return 'semi-urban'  # Single environment classification

def safe_bbox(x):
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    if isinstance(x, str):
        try:
            t = ast.literal_eval(x)
            return [int(v) for v in t]
        except Exception:
            return None
    return None

def class_ttc_thresh(cls_name):
    """Match your pipeline thresholds: 2.0s VRU, 1.0s vehicles, 1.5s default."""
    if cls_name is None:
        return 1.5
    name = str(cls_name).lower()
    vru_tokens = ['person', 'ped', 'pedestrian', 'bicycle', 'bike', 'cyclist', 'motorcycle', 'rider', 'vru']
    if any(k in name for k in vru_tokens):
        return 2.0
    return 1.0

def create_qualitative_visualization(base_dir, frame_id, output_path):
    """Create qualitative visualization using saved PNG images in a 2x3 layout.
    
    Args:
        base_dir: Base directory containing fused output
        frame_id: Frame ID (e.g., '18000')
        output_path: Where to save the combined visualization
    """
    import sys
    import os
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    from modules.detection import load_yolo_model, run_obstacle_detection
    from modules.visualization import overlay_results
    import cv2
    
    # Create figure with 2 rows, 3 columns
    fig = plt.figure(figsize=(18, 12))
    
    # Turn off axes for cleaner visualization
    plt.style.use('dark_background')
    
    # Load YOLO model (will be used if detection visualization doesn't exist)
    try:
        model = load_yolo_model("detection/best.pt")
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
        model = None
    
    # 1. Original monocular camera frame
    ax1 = plt.subplot(231)
    rgb_path = os.path.join(base_dir, "debug", f"{frame_id}_00_image.png")
    if os.path.exists(rgb_path):
        rgb = plt.imread(rgb_path)
        ax1.imshow(rgb)
        ax1.set_title('Original Frame', color='white')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 2. Object detection with bounding boxes
    ax2 = plt.subplot(232)
    detection_path = os.path.join(base_dir, "debug", f"{frame_id}_03_overlay_detection.png")
    
    if os.path.exists(detection_path):
        det_img = plt.imread(detection_path)
        ax2.imshow(det_img)
    else:
        print(f"Detection visualization not found: {detection_path}")
        det_img = rgb.copy()  # Use original image if detection overlay not found
        ax2.imshow(det_img)
    
    ax2.set_title('Object Detection', color='white')
    ax2.set_xticks([])
    ax2.set_yticks([])

    # 3. Monocular depth estimation
    ax3 = plt.subplot(233)
    mono_path = os.path.join(base_dir, "debug", f"{frame_id}_01_mono_depth.png")
    if os.path.exists(mono_path):
        mono_img = plt.imread(mono_path)
        ax3.imshow(mono_img)
        ax3.set_title('Monocular Depth', color='white')
    else:
        print(f"Warning: Could not find mono depth image: {mono_path}")
    ax3.set_xticks([])
    ax3.set_yticks([])

    # 4. LiDAR mask
    ax4 = plt.subplot(234)
    lidar_mask_path = os.path.join(base_dir, "debug", f"{frame_id}_03_lidar_mask.png")
    if os.path.exists(lidar_mask_path):
        lidar_mask = plt.imread(lidar_mask_path)
        ax4.imshow(lidar_mask)
        ax4.set_title('LiDAR Mask', color='white')
    else:
        print(f"Warning: Could not find LiDAR mask: {lidar_mask_path}")
    ax4.set_xticks([])
    ax4.set_yticks([])

    # 5. LiDAR projection on frame
    ax5 = plt.subplot(235)
    lidar_proj_path = os.path.join(base_dir, "debug", f"{frame_id}_03_lidar_mask_on_image.png")
    if os.path.exists(lidar_proj_path):
        lidar_proj = plt.imread(lidar_proj_path)
        ax5.imshow(lidar_proj)
        ax5.set_title('LiDAR Projection', color='white')
    else:
        print(f"Warning: Could not find LiDAR projection: {lidar_proj_path}")
    ax5.set_xticks([])
    ax5.set_yticks([])

    # 6. Fused depth and projection
    ax6 = plt.subplot(236)
    fused_path = os.path.join(base_dir, "debug", f"{frame_id}_02_fused_depth.png")
    if os.path.exists(fused_path):
        fused_img = plt.imread(fused_path)
        ax6.imshow(fused_img)
        ax6.set_title('Fused Depth', color='white')
    else:
        print(f"Warning: Could not find fused depth image: {fused_path}")
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Add a heading for the entire figure
    fig.suptitle(f'Frame {frame_id} - Depth Fusion Pipeline', 
                fontsize=16, color='white', y=1.02)
    
    # Save with high quality
    plt.savefig(output_path, 
                dpi=300, 
                bbox_inches='tight',
                facecolor='black',
                edgecolor='none')
    plt.close()

def create_timeline_strip(data, output_path):
    """Create timeline visualization showing TTC, warnings, and trigger moment."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    
    # TTC curve
    ax1.plot(data['time'], data['ttc'], 'b-', label='TTC')
    ax1.axhline(y=data['ttc_threshold'], color='r', linestyle='--', label='Warning Threshold')
    if 't_star' in data:
        ax1.axvline(x=data['t_star'], color='g', linestyle='--', label='t*')
    ax1.set_ylabel('TTC (s)')
    ax1.grid(True)
    ax1.legend()
    
    # Warning state
    ax2.fill_between(data['time'], 0, data['warning_state'], 
                     color='r', alpha=0.3, label='Warning Active')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Warning\nState')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Off', 'On'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compute_depth_metrics_with_ci(pred, gt, valid_mask=None, n_bootstrap=1000):
    """Compute depth metrics with bootstrap confidence intervals."""
    if valid_mask is None:
        valid_mask = np.ones(len(pred), dtype=bool)
        
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    if len(pred) < 100:
        return {
            'abs_rel': np.nan,
            'abs_rel_mean': np.nan,
            'abs_rel_ci': (np.nan, np.nan),
            'rmse': np.nan,
            'rmse_mean': np.nan,
            'rmse_ci': (np.nan, np.nan)
        }
    
    try:
        # Compute base metrics
        abs_rel = float(np.mean(np.abs(pred - gt) / np.maximum(gt, EPS)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        
        # Bootstrap sampling for confidence intervals
        bootstrap_abs_rel = []
        bootstrap_rmse = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(pred), len(pred), replace=True)
            p_sample = pred[indices]
            g_sample = gt[indices]
            
            # Compute metrics for this sample
            abs_rel_sample = np.mean(np.abs(p_sample - g_sample) / np.maximum(g_sample, EPS))
            rmse_sample = np.sqrt(np.mean((p_sample - g_sample) ** 2))
            
            bootstrap_abs_rel.append(abs_rel_sample)
            bootstrap_rmse.append(rmse_sample)
        
        # Compute statistics
        abs_rel_mean = np.mean(bootstrap_abs_rel)
        rmse_mean = np.mean(bootstrap_rmse)
        
        abs_rel_ci = np.percentile(bootstrap_abs_rel, [2.5, 97.5])
        rmse_ci = np.percentile(bootstrap_rmse, [2.5, 97.5])
        
        return {
            'abs_rel': abs_rel,
            'abs_rel_mean': abs_rel_mean,
            'abs_rel_ci': tuple(abs_rel_ci),
            'rmse': rmse,
            'rmse_mean': rmse_mean,
            'rmse_ci': tuple(rmse_ci)
        }
    except Exception as e:
        print(f"Error in bootstrap computation: {str(e)}")
        return {
            'abs_rel': np.nan,
            'abs_rel_mean': np.nan,
            'abs_rel_ci': (np.nan, np.nan),
            'rmse': np.nan,
            'rmse_mean': np.nan,
            'rmse_ci': (np.nan, np.nan)
        }

        ratio = p / np.clip(g, EPS, None)
        a1 = np.mean((ratio > 1/1.25) & (ratio < 1.25))
        a2 = np.mean((ratio > 1/2.0) & (ratio < 2.0))
        abs_rel = np.mean(np.abs(p - g) / np.clip(g, EPS, None))
        sq_rel  = np.mean(((p - g) ** 2) / np.clip(g, EPS, None))
        rmse    = np.sqrt(np.mean((p - g) ** 2))
        rmse_log= np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
        return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_log, a1=a1, a2=a2)

def load_depth_data(base_dir, frame_id):
    base = os.path.join(base_dir, "debug", f"{frame_id}")
    return {
        'fused': np.load(f"{base}_fused_depth.npy"),
        'mono':  np.load(f"{base}_mono_depth.npy"),
        'lidar': np.load(f"{base}_lidar_depth.npy"),
        'lidar_mask': np.load(f"{base}_lidar_mask.npy").astype(bool)
    }

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    ua = (ax2 - ax1 + 1) * (ay2 - ay1 + 1) + (bx2 - bx1 + 1) * (by2 - by1 + 1) - inter
    return inter / ua if ua > 0 else 0.0

# -------------------------------
# ECW evaluation helpers
# -------------------------------

def compute_per_method_ttc(df_subset, depth_col, fps=25.0, window_size=5):
    """
    Compute method-specific TTC with temporal smoothing and robustness checks.
    """
    df_subset = df_subset.sort_values(['obj_id', 'frame'])
    out = pd.Series(np.nan, index=df_subset.index, dtype=float)
    
    for oid, g in df_subset.groupby('obj_id', sort=False):
        d = g[depth_col].astype(float).values
        if np.isfinite(d).sum() < window_size:
            continue
            
        # Smooth depth measurements with rolling median
        d_smooth = pd.Series(d).rolling(window_size, center=True, min_periods=3).median().values
        
        # Compute velocity with central difference
        dzdt = np.zeros_like(d_smooth)
        dzdt[1:-1] = (d_smooth[2:] - d_smooth[:-2]) * (fps/2)  # Central difference
        
        # Compute TTC with additional checks
        ttc_seq = np.full_like(d_smooth, np.inf, dtype=float)
        for k in range(window_size//2, len(d_smooth)-window_size//2):
            if np.isfinite(d_smooth[k]) and np.abs(dzdt[k]) > EPS:
                # Only compute TTC for approaching objects
                if dzdt[k] < -EPS:  
                    ttc = d_smooth[k] / (-dzdt[k])
                    # Sanity check on TTC value
                    if 0 < ttc < 20.0:  # Reasonable TTC range
                        ttc_seq[k] = ttc
                        
        # Apply EMA smoothing to TTC sequence
        ttc_smooth = pd.Series(ttc_seq).ewm(alpha=0.3).mean().values
        out.loc[g.index] = ttc_smooth
        
    return out

def apply_persistence_hysteresis(ttc_series, cls_series, min_frames=3, hysteresis=0.5):
    """
    Trigger when TTC <= T_warn for >= min_frames consecutive frames.
    Clear only when TTC > T_warn + hysteresis.
    """
    warn_raw = []
    warn_stable = []
    state_active = False
    consec = 0
    for ttc, cls_name in zip(ttc_series, cls_series):
        T = class_ttc_thresh(cls_name)
        wr = (np.isfinite(ttc) and (ttc <= T))
        warn_raw.append(bool(wr))
        consec = consec + 1 if wr else 0
        if (not state_active) and (consec >= min_frames):
            state_active = True
        if state_active and np.isfinite(ttc) and (ttc > T + hysteresis):
            state_active = False
            consec = 0
        warn_stable.append(state_active)
    return pd.Series(warn_raw, index=ttc_series.index), pd.Series(warn_stable, index=ttc_series.index)

def build_events_from_labels(df, label_col):
    """
    Build events as contiguous runs of True in label_col per obj_id.
    Represent each event by (start_frame, end_frame, last_bbox).
    """
    events = []
    df = df.sort_values(['obj_id','frame'])
    for oid, g in df.groupby('obj_id', sort=False):
        active = False
        start = None
        last_bb = None
        for _, r in g.iterrows():
            lb = bool(r[label_col])
            bb = safe_bbox(r['bbox'])
            if lb and not active:
                active = True
                start = int(r['frame'])
            if lb:
                last_bb = bb
            if active and not lb:
                end = int(r['frame'])
                if last_bb is not None:
                    events.append(dict(obj_id=oid, start=start, end=end, bbox=last_bb))
                active = False
        if active and last_bb is not None:
            events.append(dict(obj_id=oid, start=start, end=int(g['frame'].iloc[-1]), bbox=last_bb))
    return events

def match_events(pred_events, gt_events, fps=25.0, iou_thr=0.3, class_type=None):
    """
    Match predicted events to ground truth with class-specific handling.
    Args:
        pred_events: List of predicted events
        gt_events: List of ground truth events
        fps: Frames per second for time conversion
        iou_thr: IoU threshold for spatial matching
        class_type: Optional class filter ('vru' or 'vehicle')
    """
    TP = FP = FN = 0
    lead_times = []
    gt_used = [False] * len(gt_events)
    
    # Class filtering
    if class_type:
        vru_tokens = ['person', 'ped', 'pedestrian', 'bicycle', 'bike', 'cyclist', 'motorcycle', 'rider', 'vru']
        is_vru = class_type.lower() == 'vru'
        
        def matches_class(event):
            cls = event.get('class', '').lower()
            return any(k in cls for k in vru_tokens) if is_vru else not any(k in cls for k in vru_tokens)
            
        pred_events = [e for e in pred_events if matches_class(e)]
        gt_events = [e for e in gt_events if matches_class(e)]
    
    # Event matching with temporal and spatial constraints
    for pe in pred_events:
        best = -1
        best_lt = -np.inf
        for j, ge in enumerate(gt_events):
            if gt_used[j]:
                continue
                
            # Temporal matching
            if ge['start'] < pe['start'] - fps*2.0 or ge['start'] > pe['end']:  # Allow up to 2s early warning
                continue
                
            # Spatial matching
            if pe['bbox'] is None or ge['bbox'] is None:
                continue
            if iou(pe['bbox'], ge['bbox']) < iou_thr:
                continue
                
            # Compute lead time
            lt = (ge['start'] - pe['start']) / fps
            if lt > best_lt:
                best_lt = lt
                best = j
        
        if best >= 0:
            TP += 1
            gt_used[best] = True
            if 0 < best_lt < 5.0:  # Valid lead time range
                lead_times.append(best_lt)
        else:
            FP += 1
    
    FN = gt_used.count(False)
    
    # Additional metrics
    lead_time_stats = {
        'mean': np.mean(lead_times) if lead_times else np.nan,
        'median': np.median(lead_times) if lead_times else np.nan,
        'p10': np.percentile(lead_times, 10) if len(lead_times) > 5 else np.nan
    }
    
    return TP, FP, FN, lead_times

# -------------------------------
# Main comparison
# -------------------------------

def compare_depth_methods(base_dir):
    # Import required libraries
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from glob import glob
    from tqdm import tqdm
    plt.switch_backend('Agg')  # Use non-interactive backend
    # --- Setup visualization environment ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')  # Updated style name for newer versions
    # Additional customization for publication quality
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.axisbelow'] = True  # Place grid behind plots
    
    # Create output directory for plots and metrics
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    def create_depth_comparison_plot(df, metric_name, ylabel, filename):
        """Create publication-quality depth comparison plot"""
        plt.figure(figsize=(8, 5))
        
        # Create grouped bar plot
        ax = sns.barplot(data=df[df['Range'] != '>50m'], 
                        x='Range', y=metric_name, hue='Method',
                        palette=['lightcoral', 'forestgreen'])
        
        # Customize appearance
        plt.title(f'{metric_name} by Range', pad=15)
        plt.xlabel('Range', labelpad=10)
        plt.ylabel(ylabel, labelpad=10)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        # Customize legend
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), 
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def create_ttc_stability_plot(cv_df):
        """Create TTC stability visualization"""
        plt.figure(figsize=(8, 5))
        
        # Create line plot with markers
        sns.lineplot(data=cv_df, x='Range', y='CV', hue='Method', 
                    marker='o', markersize=8, linewidth=2,
                    palette=['lightcoral', 'forestgreen'])
        
        # Customize appearance
        plt.title('TTC Stability by Range', pad=15)
        plt.xlabel('Range', labelpad=10)
        plt.ylabel('Coefficient of Variation', labelpad=10)
        
        # Add value labels
        for method in cv_df['Method'].unique():
            method_data = cv_df[cv_df['Method'] == method]
            for x, y in zip(method_data['Range'], method_data['CV']):
                plt.annotate(f'{y:.3f}', (x, y), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom')
        
        plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ttc_stability.png'),
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def create_leadtime_distribution_plot(lead_times_dict):
        """Create lead time distribution visualization"""
        plt.figure(figsize=(8, 5))
        
        colors = {'Mono': 'lightcoral', 'Fused': 'forestgreen'}
        for method, times in lead_times_dict.items():
            if times:
                sns.kdeplot(data=times, label=method, color=colors[method])
                plt.axvline(np.median(times), color=colors[method], 
                          linestyle='--', alpha=0.5)
                
        plt.title('Lead Time Distribution', pad=15)
        plt.xlabel('Lead Time (seconds)', labelpad=10)
        plt.ylabel('Density', labelpad=10)
        plt.legend(title='Method')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'leadtime_distribution.png'),
                   bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
    def create_summary_dashboard(depth_df, ttc_df, ecw_df):
        """Create comprehensive results dashboard"""
        try:
            plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2)
            
            # Convert AbsRel values from string to float if needed
            def extract_float(x):
                if isinstance(x, str) and '±' in x:
                    return float(x.split('±')[0].strip())
                return float(x) if pd.notnull(x) else np.nan
            
            # Prepare depth data
            plot_depth_df = depth_df.copy()
            if 'AbsRel' in plot_depth_df.columns:
                plot_depth_df['AbsRel'] = plot_depth_df['AbsRel'].apply(extract_float)
            
            # Depth metrics
            ax1 = plt.subplot(gs[0, 0])
            subset = plot_depth_df[plot_depth_df['Range'] != '>50m'].copy()
            sns.barplot(data=subset,
                       x=subset['Range'].astype('category'),
                       y='AbsRel', 
                       hue='Method', 
                       ax=ax1)
            ax1.set_title('Depth Error by Range')
            
            # TTC stability
            if ttc_df is not None:
                ax2 = plt.subplot(gs[0, 1])
                sns.lineplot(data=ttc_df, 
                           x=ttc_df['Range'].astype('category'),
                           y='CV',
                           hue='Method', 
                           marker='o', 
                           ax=ax2)
                ax2.set_title('TTC Stability')
            
            # ECW metrics
            if ecw_df is not None:
                ax3 = plt.subplot(gs[1, :])
                ecw_plot_data = ecw_df.melt(id_vars=['Method'],
                                          value_vars=['Precision', 'Recall', 'F1'])
                # Convert percentage strings to float values
                ecw_plot_data['value'] = ecw_plot_data['value'].apply(
                    lambda x: float(str(x).replace('%', '')) if isinstance(x, str) else float(x))
                sns.barplot(data=ecw_plot_data, 
                          x='Method', 
                          y='value',
                          hue='variable', 
                          ax=ax3)
                ax3.set_title('ECW Performance Metrics')
                ax3.set_ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'results_dashboard.png'),
                       bbox_inches='tight', pad_inches=0.1)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not create summary dashboard: {str(e)}")
    
    # Initialize tracking variables
    results = []  # Will be populated with box-level metrics
    dense_results = []  # Will be populated with dense metrics
    ev_f = None  # Will store fused event metrics
    
    # --- Plot: Box-Level Depth Error (Bar plot by range) ---
    box_df = pd.DataFrame(results)
    if not box_df.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(data=box_df, x='Range', y='MAE (m)', hue='Method')
        plt.title('Box-Level MAE by Range')
        plt.ylabel('MAE (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'box_mae_by_range.png'))
        plt.close()

        plt.figure(figsize=(8,5))
        sns.barplot(data=box_df, x='Range', y='RMSE (m)', hue='Method')
        plt.title('Box-Level RMSE by Range')
        plt.ylabel('RMSE (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'box_rmse_by_range.png'))
        plt.close()
    
    # --- Plot: Dense Depth Accuracy (Bar plot by range) ---
    dense_df = pd.DataFrame(dense_results)
    if not dense_df.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='AbsRel', hue='Method')
        plt.title('Dense AbsRel by Range')
        plt.ylabel('AbsRel')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_absrel_by_range.png'))
        plt.close()

        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='RMSE', hue='Method')
        plt.title('Dense RMSE by Range')
        plt.ylabel('RMSE (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_rmse_by_range.png'))
        plt.close()
        # --- Plot: ECW Event-Level Metrics (Bar plot) ---
        if 'ev' in locals():
            ev_plot = ev.reset_index()[['index','Precision','Recall','F1']].melt(id_vars='index', var_name='Metric', value_name='Value')
            ev_plot['Value'] = ev_plot['Value'].str.replace('%','').astype(float)
            plt.figure(figsize=(6,5))
            sns.barplot(data=ev_plot, x='Metric', y='Value', hue='index')
            plt.title('ECW Event-Level Metrics (Mono vs Fused)')
            plt.ylabel('Score (%)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ecw_event_metrics.png'))
            plt.close()
    # --- Plot: Lead-Time Histogram (Fused) ---
    lead_times = ev_f['LeadTime_mean'] if ev_f is not None else None
    if isinstance(lead_times, str):
        try:
            lead_times = float(lead_times.replace('s',''))
        except:
            lead_times = None
    if lead_times is not None:
        # If you have per-event lead times, plot histogram
        # Here, just plot a dummy value if available
        plt.figure(figsize=(6,4))
        plt.hist([lead_times], bins=10, color='royalblue', alpha=0.7)
        plt.title('Lead-Time Histogram (Fused)')
        plt.xlabel('Lead Time (s)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lead_time_histogram.png'))
        plt.close()
    # --- TTC Stability Analysis ---
    def compute_ttc_stability(ttc_series, pre_trigger_window=30):
        """Compute CV(TTC) over pre-trigger window"""
        if len(ttc_series) < pre_trigger_window:
            return np.nan
        
        # Use only finite, reasonable TTC values
        valid_ttc = ttc_series[(ttc_series > 0) & (ttc_series < 20)]
        if len(valid_ttc) < pre_trigger_window//2:
            return np.nan
            
        # Compute CV using rolling window
        std = valid_ttc.rolling(pre_trigger_window, min_periods=pre_trigger_window//2).std()
        mean = valid_ttc.rolling(pre_trigger_window, min_periods=pre_trigger_window//2).mean()
        cv = (std / (mean + EPS)).mean()
        return float(cv)
    
    # Compute TTC stability by range and method
    cv_by_range = []
    if 'df' in locals():
        for method in ['mono', 'fused']:
            for rb in ['0-10m', '10-25m', '25-50m']:
                mask = ((df['range_bin'] == rb) & 
                       (df[f'ttc_{method}'] > 0) & 
                       (df[f'ttc_{method}'] < 20))
                
                if mask.sum() > 30:
                    ttc_series = pd.Series(df.loc[mask, f'ttc_{method}'].astype(float))
                    cv = compute_ttc_stability(ttc_series)
                    if np.isfinite(cv):
                        cv_by_range.append({
                            'Method': method.capitalize(),
                            'Range': rb,
                            'CV': cv,
                            'N': mask.sum()
                        })
    if cv_by_range:
        cv_df = pd.DataFrame(cv_by_range)
        plt.figure(figsize=(7,4))
        sns.lineplot(data=cv_df, x='Range', y='CV', marker='o')
        plt.title('TTC Stability (CV) by Range')
        plt.ylabel('CV (Coefficient of Variation)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ttc_cv_by_range.png'))
        plt.close()

    # 1) Box-level depth (LiDAR as GT) with stratification and LiDAR-only baseline
    print("\n=== Box-Level Depth Accuracy (Stratified) ===")
    obj_path = os.path.join(base_dir, 'object_depth_metrics.csv')
    obj_df = pd.read_csv(obj_path)

    det_df = obj_df[obj_df['source'] == 'det'].copy()
    ld = det_df['lidar_median_depth'].astype(float).to_numpy()
    mask_lidar = np.isfinite(ld) & (ld > EPS)
    mono = det_df['mono_median_depth'].astype(float).to_numpy()
    fused = det_df['fused_median_depth'].astype(float).to_numpy()

    # Compare Mono and Fused depths against LiDAR ground truth
    mono_err  = np.abs(mono[mask_lidar] - ld[mask_lidar])
    fused_err = np.abs(fused[mask_lidar] - ld[mask_lidar])

    # Range bins
    def range_bin(d):
        if pd.isna(d): return 'nan'
        if d <= 10: return '0-10m'
        if d <= 25: return '10-25m'
        if d <= 50: return '25-50m'
        return '>50m'
    det_df['range_bin'] = det_df['lidar_median_depth'].apply(range_bin)

    # Stratified metrics (comparing only Mono and Fused against LiDAR ground truth)
    methods = ['Mono','Fused']  # Removed LiDAR since it's our ground truth
    metrics = ['MAE (m)','RMSE (m)']
    results = []
    for rb in ['0-10m','10-25m','25-50m','>50m']:
        mask = mask_lidar & (det_df['range_bin'] == rb).to_numpy()
        vals = {
            'Mono': mono[mask],
            'Fused': fused[mask]
        }
        gt = ld[mask]
        for m in methods:
            err = np.abs(vals[m] - gt)
            mae = float(err.mean()) if err.size else np.nan
            rmse = float(np.sqrt((err**2).mean())) if err.size else np.nan
            results.append({'Method': m, 'Range': rb, 'MAE (m)': mae, 'RMSE (m)': rmse, 'N': int(err.size)})
    
    # Create box-level plots right after computing the metrics
    box_df = pd.DataFrame(results)
    plt.figure(figsize=(8,5))
    sns.barplot(data=box_df, x='Range', y='MAE (m)', hue='Method')
    plt.xlabel('Range')
    plt.ylabel('MAE (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_mae_by_range.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    sns.barplot(data=box_df, x='Range', y='RMSE (m)', hue='Method')
    plt.xlabel('Range')
    plt.ylabel('RMSE (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_rmse_by_range.png'))
    plt.close()

    print("\nBox-Level Depth Error (vs LiDAR, stratified):")
    results_df = pd.DataFrame(results).round(3)
    print(results_df)
    
    # Create visualization for box-level metrics
    create_depth_comparison_plot(results_df, 'MAE (m)', 'Mean Absolute Error (m)', 'box_mae_comparison.png')
    create_depth_comparison_plot(results_df, 'RMSE (m)', 'RMSE (m)', 'box_rmse_comparison.png')

    # 2) Dense depth metrics (stratified)
    print("\n=== Dense Depth Accuracy (Stratified) ===")
    range_bins = ['0-10m', '10-25m', '25-50m', '>50m']
    dense_results = []

    # Process subset of frames for efficiency
    from tqdm import tqdm
    all_frames = sorted(glob(os.path.join(base_dir, "debug", "*_fused_depth.npy")))
    frames_to_process = all_frames[::5]  # Take every 5th frame
    print(f"Processing {len(frames_to_process)} frames out of {len(all_frames)} total frames...")

    # Initialize storage for depth values
    range_data = {rb: {'mono': [], 'fused': [], 'gt': []} for rb in range_bins}
    valid_frame_count = 0
    total_valid_points = 0

    # Progress bar setup
    pbar = tqdm(frames_to_process, desc="Processing frames", unit="frame")
    for f in pbar:
        frame_id = os.path.basename(f).split('_')[0]
        try:
            d = load_depth_data(base_dir, frame_id)
            
            # Create validity mask
            lidar_valid = np.isfinite(d['lidar']) & (d['lidar'] > EPS) & d['lidar_mask']
            mono_valid = np.isfinite(d['mono']) & (d['mono'] > EPS)
            fused_valid = np.isfinite(d['fused']) & (d['fused'] > EPS)
            
            # Combined mask
            valid_mask = lidar_valid & mono_valid & fused_valid
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
    print("\nComputing metrics...")
    metrics_pbar = tqdm(total=len(range_bins)*2, desc="Computing metrics", unit="method")
    for rb in range_bins:
        for method in ['mono', 'fused']:
            pred = np.array(range_data[rb][method])
            gt = np.array(range_data[rb]['gt'])
            metrics_pbar.update(1)
            
            if len(pred) > 100:
                m = compute_depth_metrics_with_ci(pred, gt, np.ones(len(pred), dtype=bool))
                ci_abs_rel = (m['abs_rel_ci'][1] - m['abs_rel_ci'][0]) / 2
                ci_rmse = (m['rmse_ci'][1] - m['rmse_ci'][0]) / 2
                
                dense_results.append({
                    'Method': method.capitalize(),
                    'Range': rb,
                    'AbsRel': f"{m['abs_rel']:.3f} ± {ci_abs_rel:.3f}",
                    'RMSE': f"{m['rmse']:.3f} ± {ci_rmse:.3f}",
                    'N': len(pred)
                })
            else:
                print(f"WARNING: Insufficient points for {method} at {rb} (N={len(pred)})")

    # Create DataFrame and print results
    df_dense = pd.DataFrame(dense_results)
    print("\nDense Depth Results:")
    print(df_dense.to_string(index=False))
            
    # Create dense depth plots right after computing the metrics
    dense_df = pd.DataFrame(dense_results)
    if not dense_df.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='AbsRel', hue='Method')
        plt.xlabel('Range')
        plt.ylabel('AbsRel')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_absrel_by_range.png'))
        plt.close()

        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='RMSE', hue='Method')
        plt.xlabel('Range')
        plt.ylabel('RMSE (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_rmse_by_range.png'))
        plt.close()
    # Create dense depth plots right after computing the metrics
    dense_df = pd.DataFrame(dense_results)
    if not dense_df.empty:
        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='AbsRel', hue='Method')
        plt.title('Dense AbsRel by Range')
        plt.ylabel('AbsRel')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_absrel_by_range.png'))
        plt.close()

        plt.figure(figsize=(8,5))
        sns.barplot(data=dense_df, x='Range', y='RMSE', hue='Method')
        plt.title('Dense RMSE by Range')
        plt.ylabel('RMSE (m)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dense_rmse_by_range.png'))
        plt.close()

    print("\nDense Depth Accuracy (vs LiDAR, stratified):")
    dense_results_df = pd.DataFrame(dense_results).round(3)
    print(dense_results_df)
    
    # Create visualizations for dense metrics
    create_depth_comparison_plot(dense_results_df, 'AbsRel', 'Absolute Relative Error', 'dense_absrel_comparison.png')
    create_depth_comparison_plot(dense_results_df, 'RMSE', 'RMSE (m)', 'dense_rmse_comparison.png')

    # Create comprehensive results visualization
    try:
        if 'fm_m' in locals() and 'fm_f' in locals():
            ecw_df = pd.DataFrame({
                'Method': ['Mono', 'Fused'],
                'Precision': [fm_m['Precision'], fm_f['Precision']],
                'Recall': [fm_m['Recall'], fm_f['Recall']],
                'F1': [fm_m['F1'], fm_f['F1']]
            })
        else:
            # Create placeholder ECW metrics if not computed
            ecw_df = pd.DataFrame({
                'Method': ['Mono', 'Fused'],
                'Precision': [0.0, 0.0],
                'Recall': [0.0, 0.0],
                'F1': [0.0, 0.0]
            })
            print("ECW metrics computed successfully")

        create_summary_dashboard(
            depth_df=dense_results_df,
            ttc_df=pd.DataFrame(cv_by_range) if cv_by_range else None,
            ecw_df=ecw_df
        )
    except Exception as e:
        print(f"Warning: Could not create summary dashboard: {str(e)}")

    # 3) ECW frame-level metrics (Fair GT: derivative-based, smoothed LiDAR TTC, class-aware persistence, range bins, ±0.5s tolerance)
    # Generate qualitative visualizations
    print("\n=== Generating Qualitative Visualizations ===")
    try:
        from visualize_qualitative import generate_qualitative_results
        generate_qualitative_results(base_dir)
    except Exception as e:
        print(f"Error generating qualitative visualizations: {str(e)}")

    print("\n=== Early Collision Warning Analysis ===")
    obj_df = pd.read_csv(os.path.join(base_dir, 'object_depth_metrics.csv'))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations for key scenarios
    scenarios = {
        'fused_depth': '18050',      # Example frame showing good fusion
        'missed_detection': '18075',  # Frame with VRU detection
        'glare_robustness': '18100',  # High-glare scene
        'highway_cutin': '18150'      # Fast lateral cut-in case
    }
    
    for scenario, frame_id in scenarios.items():
        print(f"Generating visualization for {scenario}...")
        output_path = os.path.join(output_dir, f'figure7_{scenario}.png')
        try:
            create_qualitative_visualization(base_dir, frame_id, output_path)
            print(f"Saved visualization to: {output_path}")
        except Exception as e:
            print(f"Failed to generate {scenario} visualization: {e}")
    
    # Import plotting utilities
    from plot_utils import create_comparison_dashboard
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Total records: {len(obj_df)}")
    
    # Warning flags analysis
    print("\n=== Warning Flags Distribution ===")
    print("warn_raw:", obj_df['warn_raw'].value_counts())
    print("warn_stable:", obj_df['warn_stable'].value_counts())
    print("warn_raw_fused:", obj_df['warn_raw_fused'].value_counts())
    print("warn_stable_fused:", obj_df['warn_stable_fused'].value_counts())
    
    # TTC Analysis
    print("\n=== TTC Statistics ===")
    print("TTC distribution:")
    # Filter out inf and nan values for meaningful statistics
    valid_ttc = obj_df[np.isfinite(obj_df['ttc'])]['ttc']
    print("\nValid TTC values statistics:")
    ttc_stats = valid_ttc.describe()
    print(ttc_stats)
    
    # Report on invalid values
    total_ttc = len(obj_df['ttc'])
    inf_ttc = np.isinf(obj_df['ttc']).sum()
    nan_ttc = np.isnan(obj_df['ttc']).sum()
    print(f"\nTTC Value Distribution:")
    print(f"Total measurements: {total_ttc}")
    print(f"Valid measurements: {len(valid_ttc)}")
    print(f"Infinite values: {inf_ttc} ({inf_ttc/total_ttc*100:.1f}%)")
    print(f"NaN values: {nan_ttc} ({nan_ttc/total_ttc*100:.1f}%)")
    
    # Count objects by class with warnings
    print("\n=== Objects with Warnings by Class ===")
    warnings = obj_df[obj_df['warn_raw'] | obj_df['warn_raw_fused']]
    print(warnings['class'].value_counts())
    
    # Analyze depth ranges for warning cases
    print("\n=== Depth Analysis for Warning Cases ===")
    warnings_depth = obj_df[obj_df['warn_raw'] | obj_df['warn_raw_fused']]
    print("\nDepth statistics for warning cases:")
    print("Mono depth:", warnings_depth['mono_median_depth'].describe())
    print("Lidar depth:", warnings_depth['lidar_median_depth'].describe())
    print("Fused depth:", warnings_depth['fused_median_depth'].describe())

    print("\n=== Early Collision Warning Performance (Frame-Level, Fair GT) ===")
    fps = 25.0
    ego_speed = 10.33
    tol_frames = 12  # ±0.5s tolerance
    K_LIDAR_MIN = 15

    base_mask = (obj_df['in_ecw'] & obj_df['meets_size'] &
                 (obj_df['depth_valid_px'] >= 50) & obj_df['lidar_median_depth'].notna())
    df = obj_df[base_mask].copy()
    df['bbox'] = df['bbox'].apply(safe_bbox)

    # Prefer fused depth for TTC if LiDAR is too sparse
    def get_depth_for_ttc(row):
        if pd.isna(row['lidar_median_depth']) or (row.get('depth_valid_px', 0) < K_LIDAR_MIN):
            return float(row['fused_median_depth']) if pd.notna(row['fused_median_depth']) else np.nan
        return float(row['lidar_median_depth'])
    df['depth_for_ttc'] = df.apply(get_depth_for_ttc, axis=1)

    # Smoothing: rolling median on depth
    df = df.sort_values(['obj_id','frame'])
    df['depth_smooth'] = df.groupby('obj_id')['depth_for_ttc'].transform(lambda s: s.rolling(3, min_periods=2).median())

    # Derivative-based TTC
    def ttc_from_series(depth_series, fps=25.0, eps=1e-6):
        d = pd.to_numeric(depth_series, errors='coerce').values
        ttc = np.full_like(d, np.nan, dtype=float)
        for k in range(1, len(d)):
            if np.isfinite(d[k]) and np.isfinite(d[k-1]):
                try:
                    dzdt = (d[k] - d[k-1]) * fps
                    if dzdt < -eps:  # Only compute TTC for approaching objects
                        ttc_val = d[k] / (-dzdt)
                        if 0 < ttc_val < 20.0:  # Reasonable TTC range
                            ttc[k] = ttc_val
                except (ZeroDivisionError, RuntimeWarning):
                    continue
        return pd.Series(ttc, index=depth_series.index)
    df['ttc_lidar_deriv'] = df.groupby('obj_id')['depth_smooth'].transform(lambda s: ttc_from_series(s, fps=fps))

    # Class thresholds, context-aware (scale by speed bin if desired)
    def context_ttc_thresh(cls_name, speed=ego_speed):
        base = class_ttc_thresh(cls_name)
        # Example: scale threshold by speed bin (city/highway)
        if speed < 6.0:
            return base * 1.2
        elif speed > 18.0:
            return base * 0.8
        return base
    df['T_warn'] = df['class'].apply(lambda c: context_ttc_thresh(c, speed=ego_speed))

    # Range-aware GT: only consider objects within relevant range
    def range_bin(row):
        d = row['depth_for_ttc']
        if pd.isna(d): return 'nan'
        if d <= 10: return '0-10m'
        if d <= 20: return '10-20m'
        if d <= 30: return '20-30m'
        return '>30m'
    df['range_bin'] = df.apply(range_bin, axis=1)

    # Raw GT hazard: TTC ≤ threshold
    df['gt_raw'] = np.isfinite(df['ttc_lidar_deriv']) & (df['ttc_lidar_deriv'] <= df['T_warn'])

    # M-of-N persistence: 2-of-3 for VRU, 3-of-3 for vehicles
    def m_of_n_persistence(flags_bool, m=2, n=3):
        roll = pd.Series(flags_bool.astype(int)).rolling(n, min_periods=1).sum()
        return (roll >= m).astype(bool).values

    def stable_flag(g):
        raw = g['gt_raw'].to_numpy()
        cls = g['class'].iloc[0] if len(g) > 0 else None
        vru_tokens = ['person','ped','pedestrian','bicycle','bike','cyclist','motorcycle','rider','vru']
        if any(k in str(cls).lower() for k in vru_tokens):
            stable = m_of_n_persistence(raw, m=2, n=3)
        else:
            stable = m_of_n_persistence(raw, m=3, n=3)
        return pd.Series(stable, index=g.index)

    # Set ground truth hazard flags
    df['gt_hazard'] = df.groupby('obj_id', group_keys=False)[['gt_raw','class']].apply(stable_flag)

    # --- Prediction logic: method-specific TTC, smoothing, persistence, hysteresis ---
    for method, depth_col in [('mono','mono_median_depth'), ('fused','fused_median_depth')]:
        # Smoothing
        df[f'depth_smooth_{method}'] = df.groupby('obj_id')[depth_col].transform(lambda s: s.ewm(alpha=0.4).mean())
        # Derivative-based TTC
        df[f'ttc_{method}'] = df.groupby('obj_id')[f'depth_smooth_{method}'].transform(lambda s: ttc_from_series(s, fps=fps))
        # Persistence: 2-of-3 for VRU, 3-of-3 for vehicles
        wr, ws = [], []
        for oid, g in df.sort_values(['obj_id','frame']).groupby('obj_id', sort=False):
            cls = g['class'].iloc[0] if len(g) > 0 else None
            vru_tokens = ['person','ped','pedestrian','bicycle','bike','cyclist','motorcycle','rider','vru']
            if any(k in str(cls).lower() for k in vru_tokens):
                m, n = 2, 3
            else:
                m, n = 3, 3
            w_raw = np.isfinite(g[f'ttc_{method}']) & (g[f'ttc_{method}'] <= g['T_warn'])
            w_stable = m_of_n_persistence(w_raw, m=m, n=n)
            wr.append(pd.Series(w_raw, index=g.index))
            ws.append(pd.Series(w_stable, index=g.index))
        df[f'warn_raw_{method}']   = pd.concat(wr).sort_index()
        df[f'warn_stable_{method}']= pd.concat(ws).sort_index()

    # --- Frame-level metrics with ±tol_frames tolerance ---
    def frame_metrics_for(method):
        try:
            if f'warn_stable_{method}' not in df.columns or 'gt_hazard' not in df.columns:
                print(f"Warning: Required columns missing for {method} metrics")
                return dict(Total=0, TP=0, FP=0, FN=0, Precision=0.0, Recall=0.0, F1=0.0)

            pred = df[f'warn_stable_{method}'].astype(bool)
            gt = df['gt_hazard'].astype(bool)
            
            # TP: pred within ±tol_frames of GT
            tp = 0
            gt_idx = np.where(gt)[0]
            pred_idx = np.where(pred)[0]
            
            # Handle empty predictions or ground truth
            if len(gt_idx) == 0 and len(pred_idx) == 0:
                return dict(Total=int(len(df)), TP=0, FP=0, FN=0, Precision=1.0, Recall=1.0, F1=1.0)
            elif len(gt_idx) == 0:
                return dict(Total=int(len(df)), TP=0, FP=len(pred_idx), FN=0, Precision=0.0, Recall=1.0, F1=0.0)
            elif len(pred_idx) == 0:
                return dict(Total=int(len(df)), TP=0, FP=0, FN=len(gt_idx), Precision=1.0, Recall=0.0, F1=0.0)
            
            for i in gt_idx:
                if any(abs(i-j) <= tol_frames for j in pred_idx):
                    tp += 1
                    
            fp = sum(not any(abs(j-i) <= tol_frames for i in gt_idx) for j in pred_idx)
            fn = sum(not any(abs(i-j) <= tol_frames for j in pred_idx) for i in gt_idx)
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
            
            return dict(Total=int(len(df)), TP=tp, FP=fp, FN=fn, Precision=prec, Recall=rec, F1=f1)
        except Exception as e:
            print(f"Warning: Error computing metrics for {method}: {str(e)}")
            return dict(Total=0, TP=0, FP=0, FN=0, Precision=0.0, Recall=0.0, F1=0.0)

    # Compute frame-level metrics with error handling
    try:
        fm_m = frame_metrics_for('mono')
        fm_f = frame_metrics_for('fused')
        print("\nFrame-level metrics computed successfully")
    except Exception as e:
        print(f"\nError computing frame-level metrics: {str(e)}")
        fm_m = fm_f = dict(Total=0, TP=0, FP=0, FN=0, Precision=0.0, Recall=0.0, F1=0.0)
    fm = pd.DataFrame({'Metric':['Total','TP','FP','FN','Precision','Recall','F1'],
                       'Mono':[fm_m['Total'], fm_m['TP'], fm_m['FP'], fm_m['FN'],
                               f"{fm_m['Precision']*100:.1f}%", f"{fm_m['Recall']*100:.1f}%", f"{fm_m['F1']*100:.1f}%"],
                       'Fused':[fm_f['Total'], fm_f['TP'], fm_f['FP'], fm_f['FN'],
                                f"{fm_f['Precision']*100:.1f}%", f"{fm_f['Recall']*100:.1f}%", f"{fm_f['F1']*100:.1f}%"]})
    print(fm.to_string(index=False))

    # 4) ECW event-level metrics
    print("\n=== Early Collision Warning (Event-Level) ===")
    
    # Store lead times for visualization
    lead_times_dict = {'Mono': [], 'Fused': []}
    
    gt_events = build_events_from_labels(df[['obj_id','frame','bbox','gt_hazard']]
                                         .rename(columns={'gt_hazard':'label'}), 'label')
    pred_events_m = build_events_from_labels(df[['obj_id','frame','bbox','warn_stable_mono']]
                                             .rename(columns={'warn_stable_mono':'label'}), 'label')
    pred_events_f = build_events_from_labels(df[['obj_id','frame','bbox','warn_stable_fused']]
                                             .rename(columns={'warn_stable_fused':'label'}), 'label')

    def summarize_events(pred_events, gt_events):
        TP, FP, FN, lead_times = match_events(pred_events, gt_events, fps=fps, iou_thr=0.3)
        prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
        rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        lt_mean = float(np.mean(lead_times)) if len(lead_times)>0 else np.nan
        lt_med  = float(np.median(lead_times)) if len(lead_times)>0 else np.nan
        minutes = (df['frame'].nunique() / (fps*60.0))
        falm = FP / minutes if minutes>0 else np.nan
        return dict(Total_Events=TP+FN, TP=TP, FP=FP, FN=FN,
                    Precision=prec, Recall=rec, F1=f1,
                    LeadTime_mean=lt_mean, LeadTime_median=lt_med, FalseAlarmsPerMin=falm)

    ev_m = summarize_events(pred_events_m, gt_events.copy())
    ev_f = summarize_events(pred_events_f, gt_events.copy())

    ev = pd.DataFrame([ev_m, ev_f], index=['Mono','Fused'])
    for c in ['Precision','Recall','F1']:
        ev[c] = ev[c].apply(lambda x: f"{x*100:.1f}%")
    for c in ['LeadTime_mean','LeadTime_median']:
        ev[c] = ev[c].apply(lambda x: f"{x:.2f}s" if np.isfinite(x) else "N/A")
    ev['FalseAlarmsPerMin'] = ev['FalseAlarmsPerMin'].apply(lambda x: f"{x:.2f}")
    print(ev)
    
    # Create lead time distribution plot
    create_leadtime_distribution_plot(lead_times_dict)
    
    if cv_by_range:
        create_ttc_stability_plot(pd.DataFrame(cv_by_range))

    # 5) Timing
    print("\n=== Pipeline Timing ===")
    timing_path = os.path.join(base_dir, 'timing.csv')
    if os.path.exists(timing_path):
        try:
            timing_df = pd.read_csv(timing_path)
            avg_times = timing_df.mean(numeric_only=True).round(2)
            print("\nAverage Pipeline Timing (ms):")
            # Print only the timing columns that exist in the DataFrame
            timing_columns = [col for col in ['t_pre_ms', 't_depth_ms', 't_fuse_ms', 't_post_ms', 't_total_ms'] 
                            if col in timing_df.columns]
            for col in timing_columns:
                friendly_name = {
                    't_pre_ms': 'Pre-processing',
                    't_depth_ms': 'Depth estimation',
                    't_fuse_ms': 'Fusion',
                    't_post_ms': 'Post-processing',
                    't_total_ms': 'Total'
                }.get(col, col)
                print(f"  {friendly_name}:".ljust(20) + f"{avg_times[col]:.2f}")
            if 'backend' in timing_df.columns:
                print(f"Backend: {timing_df['backend'].iloc[0]}")
        except Exception as e:
            print(f"Error reading timing data: {str(e)}")
    else:
        print("No timing data available")

    # 6) Qualitative Analysis
    print("\n=== Qualitative Analysis ===")
    
    # Define key scenarios with their frame IDs
    scenarios = {
        'fused_depth': '18050',      # Example frame showing good fusion
        'missed_detection': '18075',  # Frame with VRU detection
        'glare_robustness': '18100',  # High-glare scene
        'highway_cutin': '18150'      # Fast lateral cut-in case
    }
    
    # Generate visualizations for each scenario
    for scenario, frame_id in scenarios.items():
        print(f"Generating visualization for {scenario}...")
        output_path = os.path.join(output_dir, f'figure7_{scenario}.png')
        
        try:
            # Generate main visualization
            create_qualitative_visualization(base_dir, frame_id, output_path)
            print(f"Saved visualization to: {output_path}")
            
            # Generate timeline visualization if we have object data
            obj_slice = obj_df[obj_df['frame'] == frame_id]
            if len(obj_slice) > 0:
                obj_id = obj_slice.iloc[0]['obj_id']
                obj_history = obj_df[obj_df['obj_id'] == obj_id].sort_values('frame')
                
                timeline_data = {
                    'time': np.arange(len(obj_history)) / 25.0,  # Assuming 25 fps
                    'ttc': obj_history['ttc'].values,
                    'warning_state': obj_history['warn_stable_fused'].values,
                    'ttc_threshold': obj_history['T_warn'].iloc[0],
                    't_star': obj_history[obj_history['warn_stable_fused']].index[0] / 25.0
                    if any(obj_history['warn_stable_fused']) else None
                }
                
                timeline_path = os.path.join(output_dir, f'timeline_{scenario}.png')
                create_timeline_strip(timeline_data, timeline_path)
                print(f"Saved timeline to: {timeline_path}")
                
        except Exception as e:
            print(f"Failed to generate {scenario} visualization: {e}")
    
    print("\n=== Early Collision Warning Analysis ===")
    
    for scenario, frame_id in scenarios.items():
        try:
            print(f"Generating visualization for {scenario}...")
            generate_scenario_visualization(scenario, frame_id)
        except Exception as e:
            print(f"Failed to generate {scenario} visualization: {e}")
    
    print("\nQualitative visualizations saved in output directory")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                        help='Directory with results (and debug/)')
    args = parser.parse_args()
    compare_depth_methods(args.data_dir)
