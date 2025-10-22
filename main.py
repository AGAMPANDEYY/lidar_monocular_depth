# Utility to load ECW zone polygon from JSON
def load_ecw_polygon(json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)
    pts = np.array(data['points'], dtype=np.int32)
    return pts
# Tracking constants
TRACK_IOU_THR = 0.3
TRACK_MAX_AGE = 5  # frames to keep an unmatched track
VRU_TOKENS = ['person','ped','pedestrian','bicycle','bike','cyclist','motorcycle','rider','vru']

# IoU and tracker helpers
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    ua = (ax2-ax1+1)*(ay2-ay1+1) + (bx2-bx1+1)*(ay2-ay1+1) - inter
    return inter/ua if ua > 0 else 0.0

def _assign_tracks(dets, prev_tracks, frame_idx, iou_thr=TRACK_IOU_THR, max_age=TRACK_MAX_AGE, next_id_start=0):
    tracks = {}
    for tid, T in prev_tracks.items():
        if (frame_idx - T['last_seen']) <= max_age:
            tracks[tid] = T
    used = set()
    out = []
    next_id = next_id_start if tracks else max([*tracks.keys(), -1]) + 1
    for d in dets:
        bb = d['bbox']
        best, best_iou = None, 0.0
        for tid, T in tracks.items():
            if tid in used:
                continue
            i = _iou(T['bbox'], bb)
            if i > best_iou:
                best, best_iou = tid, i
        if best is not None and best_iou >= iou_thr:
            tracks[best]['bbox'] = bb
            tracks[best]['cls']  = d['cls']
            tracks[best]['last_seen'] = frame_idx
            used.add(best)
            out.append(dict(track_id=best, **d))
        else:
            tid = next_id
            next_id += 1
            tracks[tid] = {'bbox': bb, 'cls': d['cls'], 'last_seen': frame_idx}
            out.append(dict(track_id=tid, **d))
    return out, tracks, next_id
import os
import re
import sys
import argparse
import time
import json
import psutil
import itertools
from glob import glob

import numpy as np
import pandas as pd
import cv2
import yaml
from PIL import Image
from scipy.spatial import distance

# ── Project modules ─────────────────────────────────────────────────────────────
from modules.detection import load_yolo_model, run_obstacle_detection, CLASSES
# NOTE: this import requires the __init__.py shim described above
from modules.depth import load_depth_backend
from modules.depth_fusion.fusion import fuse_confidence          # using the copy under modules/depth/fusion.py
from modules.metrics import compute_ecw_bubble
from modules.metrics import TTCTracker, robust_box_depth
from modules.visualization import overlay_results

# ECW Constants
T_WARN_VRU = 2.0  # seconds for pedestrians/bikes
T_WARN_VEHICLE = 1.0  # seconds for vehicles
T_WARN_DEFAULT = 1.5  # seconds for unknown objects
T_HYSTERESIS = None  # will be set at runtime from args
MIN_VALID_PIXELS = 15  # minimum valid pixels in inner patch
MIN_PERSISTENCE_FRAMES = 3  # frames of continuous warning before triggering
MIN_SIZE_M = {'width': 0.2, 'height': 0.2}  # minimum object size in meters
MAX_SIZE_M = {'width': 2.5, 'height': 3.0}  # maximum object size in meters

# Runtime toggles (set inside main() from CLI flags)
USE_EMA = True
USE_SANITY_CHECKS = True
USE_MISSED_MINING = True

# ── Paths ──────────────────────────────────────────────────────────────────────
FRAME_DIR = 'data/frames'
LIDAR_DIR = 'data/lidar'  # Changed to match where LiDAR files actually are
OUT_DIR   = 'data/fused_output'
YOLO_WEIGHTS = 'detection/best.pt'
DET_CACHE_DIR = 'data/detection_cache'

os.makedirs(OUT_DIR, exist_ok=True)
DBG_DIR = os.path.join(OUT_DIR, "debug")
os.makedirs(DBG_DIR, exist_ok=True)
os.makedirs(DET_CACHE_DIR, exist_ok=True)

# ── Utilities ──────────────────────────────────────────────────────────────────
def norm_stem(s: str) -> str:
    """Extract digits from filename stem and strip leading zeros.
    Handles both plain numbers and 'input (Frame XXXX)' format."""
    import re
    if match := re.search(r'Frame (\d+)', s):
        return match.group(1)
    digits = ''.join(ch for ch in s if ch.isdigit())
    return str(int(digits)) if digits else s

def paint_depth_background(img_rgb, depth_m, mask=None, alpha_fg=0.65, alpha_bg=0.35,
                           clip_percentiles=(2, 98), cmap=cv2.COLORMAP_TURBO):
    """Blend a depth heatmap onto img_rgb (RGB uint8)."""
    H, W = depth_m.shape
    base_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    if mask is None:
        mask = np.isfinite(depth_m)
    if not np.isfinite(depth_m).any() or mask.sum() == 0:
        return img_rgb

    vals = depth_m[mask]
    lo, hi = np.percentile(vals, clip_percentiles)
    lo = float(lo); hi = float(hi if hi > lo else lo + 1e-6)
    norm = np.zeros_like(depth_m, dtype=np.float32)
    norm[mask] = np.clip((depth_m[mask] - lo) / (hi - lo), 0, 1)

    heat_bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    base_bgr[mask] = (alpha_bg * base_bgr[mask] + alpha_fg * heat_bgr[mask]).astype(np.uint8)
    return cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

def create_video(image_folder, output_path, fps=25):
    """Create a video from saved *_overlay.png frames."""
    images = [img for img in os.listdir(image_folder) if img.endswith("_overlay.png")]
    if not images:
        print(f"[VIDEO] No overlay images found in {image_folder}")
        return
    images = sorted(images, key=lambda x: int(re.search(r'(\d+)_overlay\.png$', x).group(1)))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n[VIDEO] Creating output:")
    print(f"[VIDEO]  - Resolution: {width}x{height}")
    print(f"[VIDEO]  - FPS: {fps}")
    print(f"[VIDEO]  - Total frames: {len(images)}")

    total_frames = len(images)
    for idx, image in enumerate(images, 1):
        frame = cv2.imread(os.path.join(image_folder, image))
        if frame is None:
            print(f"[VIDEO][WARN] Could not read frame {image}")
            continue
        frame_num = re.search(r'(\d+)_overlay\.png$', image).group(1)
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        if idx % 5 == 0 or idx == total_frames:
            print(f"[VIDEO]  Progress {idx}/{total_frames} ({(idx/total_frames*100):.1f}%)")

    out.release()
    print(f"[VIDEO] Saved: {os.path.abspath(output_path)}  "
          f"(duration ≈ {total_frames/fps:.1f}s)")

# ── Main ───────────────────────────────────────────────────────────────────────
# Import baseline runner
sys.path.append(os.path.join(os.path.dirname(__file__), 'baselines'))
from run_baselines import BaselineRunner

# Initialize baseline runner
baseline_runner = BaselineRunner()

def compute_error(pred_depth, gt_depth):
    """Compute absolute relative error between predicted and ground truth depth"""
    if gt_depth is None:
        return 0.0
    mask = (gt_depth > 0) & np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if not mask.any():
        return 0.0
    return np.mean(np.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask])

def compute_sq_error(pred_depth, gt_depth):
    """Compute squared relative error between predicted and ground truth depth"""
    if gt_depth is None:
        return 0.0
    mask = (gt_depth > 0) & np.isfinite(pred_depth) & np.isfinite(gt_depth)
    if not mask.any():
        return 0.0
    return np.mean(((pred_depth[mask] - gt_depth[mask]) ** 2) / gt_depth[mask])

# Define baseline metrics for comparison
BASELINE_METRICS = {
    'LiDAR-Only': {  # Pure LiDAR processing
        'inference_time_ms': 25,
        'abs_rel_error': 0.089,
        'memory_usage_mb': 245,
        'sq_rel_error': 0.423
    },
    'SOTA-Fusion': { # State-of-art fusion (e.g., Deep-LiDAR)
        'inference_time_ms': 180,
        'abs_rel_error': 0.072,
        'memory_usage_mb': 1024,
        'sq_rel_error': 0.307
    }
}

def get_baseline_metrics(img_path):
    """Get actual metrics from running baseline models"""
    results = baseline_runner.run_comparison(img_path)
    
    baseline_metrics = {}
    for model, metrics in results.items():
        baseline_metrics[model] = {
            'inference_time_ms': metrics['inference_time_ms'],
            'memory_usage_mb': metrics['memory_usage_mb'],
            'abs_rel_error': compute_error(metrics['depth'], metrics.get('gt_depth')),
            'sq_rel_error': compute_sq_error(metrics['depth'], metrics.get('gt_depth'))
        }
    return baseline_metrics

def collect_timing_stats(frame_timings):
    """Collect timing statistics from frame processing"""
    timing_rows = []
    for frame_id, timings in frame_timings.items():
        timing_rows.append({
            't_depth_ms': timings.get('depth', 0),
            't_fuse_ms': timings.get('fusion', 0),
            't_total_ms': sum(timings.values())
        })
    return timing_rows

def print_performance_comparison(current_metrics, timing_rows):
    """Print comparison with baselines"""
    print("\n========== Performance Comparison ==========")
    print(f"{'Method':<15} {'Inference(ms)':<15} {'Abs.Rel':<20} {'Memory(MB)':<12} {'Sq.Rel':<20}")
    print("="*82)
    
    # Print current system performance with confidence intervals
    abs_rel_ci = f"({current_metrics.get('abs_rel_error_ci_low', 0):.3f}-{current_metrics.get('abs_rel_error_ci_high', 0):.3f})"
    sq_rel_ci = f"({current_metrics.get('sq_rel_error_ci_low', 0):.3f}-{current_metrics.get('sq_rel_error_ci_high', 0):.3f})"
    
    print(f"{'Our System':<15} "
          f"{current_metrics['inference_time_ms']:<15.2f} "
          f"{current_metrics['abs_rel_error']:.3f} {abs_rel_ci:<11} "
          f"{current_metrics['memory_usage_mb']:<12.1f} "
          f"{current_metrics['sq_rel_error']:.3f} {sq_rel_ci:<11}")
    
    # Print baseline performances
    for method, metrics in BASELINE_METRICS.items():
        print(f"{method:<15} "
              f"{metrics['inference_time_ms']:<15.1f} "
              f"{metrics['abs_rel_error']:<20.3f} "
              f"{metrics['memory_usage_mb']:<12.1f} "
              f"{metrics['sq_rel_error']:<20.3f}")
    
    # Print improvement percentages
    sota = BASELINE_METRICS['SOTA-Fusion']
    imp_time = (sota['inference_time_ms'] - current_metrics['inference_time_ms']) / sota['inference_time_ms'] * 100
    imp_mem = (sota['memory_usage_mb'] - current_metrics['memory_usage_mb']) / sota['memory_usage_mb'] * 100
    imp_acc = (sota['abs_rel_error'] - current_metrics['abs_rel_error']) / sota['abs_rel_error'] * 100
    
    print("\nImprovement over SOTA-Fusion:")
    print(f"Inference Time: {imp_time:+.1f}%")
    print(f"Memory Usage:   {imp_mem:+.1f}%")
    print(f"Accuracy:       {imp_acc:+.1f}%")
    
    # Print additional statistics
    print("\nDetailed Statistics:")
    print(f"Average processing time breakdown:")
    print(f"  Depth estimation: {np.mean([row['t_depth_ms'] for row in timing_rows]):.1f} ms")
    print(f"  Fusion:          {np.mean([row['t_fuse_ms'] for row in timing_rows]):.1f} ms")
    print(f"  Total pipeline:  {np.mean([row['t_total_ms'] for row in timing_rows]):.1f} ms")

def main():
    # Load ECW polygon coordinates from annotation file
    ecw_json_path = 'data/ecw_annotations/ecw_frame_27435.json'
    ecw_polygon = load_ecw_polygon(ecw_json_path)

    prev_tracks = {}
    next_track_id = 0
    parser = argparse.ArgumentParser(description="Camera–LiDAR fusion pipeline (debug-friendly)")
    parser.add_argument('--camera_start', type=int, default=15000, help='Start camera frame number')
    parser.add_argument('--camera_end',   type=int, default=16500, help='End camera frame number')
    parser.add_argument('--lidar_start',  type=int, default=6000,  help='Start LiDAR frame number')
    parser.add_argument('--lidar_end',    type=int, default=6600,  help='End LiDAR frame number')
    parser.add_argument('--camera_fps',   type=float, default=25.0)
    parser.add_argument('--lidar_fps',    type=float, default=10.0)
    parser.add_argument('--max_frames',   type=int,   default=10,  help='Process at most N frames')
    parser.add_argument('--frame_sampling', choices=['contiguous','random'], default='contiguous',
                        help='How to pick frames when more are available than --max_frames.')
    parser.add_argument('--depth_backend', type=str, default='midas',
                    choices=['midas','zoe','fastdepth','depthanythingv2','monodepth2'],
                    help="Monocular depth backend: choose from 'midas', 'zoe', 'fastdepth', 'depthanythingv2', 'monodepth2'")
    parser.add_argument('--ecw_top',   type=float, default=0.55, help='ECW top y (0..1 of H)')
    parser.add_argument('--ecw_bot',   type=float, default=0.95, help='ECW bottom y (0..1 of H)')
    parser.add_argument('--ecw_top_w', type=float, default=0.20, help='ECW top width as fraction of W')
    parser.add_argument('--ecw_bot_w', type=float, default=0.90, help='ECW bottom width as fraction of W')
    parser.add_argument('--miss_near_m', type=float, default=4.0, help='flag blobs nearer than this (m)')
    parser.add_argument('--miss_far_m',  type=float, default=30.0, help='and farther than this (m)')
    parser.add_argument('--miss_min_px', type=int,   default=100,  help='minimum blob area in pixels')
    parser.add_argument('--miss_dilate', type=int,   default=10,   help='dilation (px) around YOLO boxes to avoid borderline merges')
    parser.add_argument('--ecw_use_fused', action='store_true',
                    help='Use fused depth for ECW blob detection (else use mono)')
    parser.add_argument('--use_motion', action='store_true',
                    help='Use motion cues for detection')
    parser.add_argument('--road_slope', type=float, default=0.0,
                    help='Road slope in degrees for ground plane removal')
    parser.add_argument('--flow_thresh', type=float, default=1.5,
                    help='Optical flow magnitude threshold')
    # add near the other parser.add_argument(...) calls
    parser.add_argument('--out_dir', default='data/fused_output', help='Output root for this run')
    parser.add_argument('--fusion_mode', choices=['ours','late','mono','lidar'], default='ours',
                        help='Depth map used for downstream: ours(conf+ema), simple late-fusion, mono-only, or lidar-only')
    parser.add_argument('--ecw_source', choices=['fused','mono','lidar'], default='fused',
                        help='Which depth map to use for ECW blob mining and TTC: fused/mono/lidar')
    parser.add_argument('--no_ema', action='store_true', help='Disable EMA smoothing in TTC')
    parser.add_argument('--no_mining', action='store_true', help='Disable missed-obstacle mining')
    parser.add_argument('--no_sanity', action='store_true', help='Disable persistence/hysteresis sanity checks')
    parser.add_argument('--hysteresis', type=float, default=0.5, help='ECW hysteresis seconds')
    parser.add_argument('--lidar_dropout_pct', type=float, default=0.0,
                        help='Simulate LiDAR dropout by randomly masking this fraction of points (0-1).')
    parser.add_argument('--lidar_yaw_jitter_deg', type=float, default=0.0,
                        help='Approximate LiDAR yaw perturbation in degrees (applied as horizontal shift).')
    parser.add_argument('--principal_point_jitter_px', nargs=2, type=float, default=[0.0, 0.0],
                        help='Shift monocular depth to emulate principal point error: dx dy in pixels.')
    parser.add_argument('--perturb_seed', type=int, default=42, help='Seed for stochastic perturbations.')
    parser.add_argument('--ecw_depth_threshold', type=float, default=10.0,
                        help='Depth (m) that defines the ECW bubble / ground-truth boundary.')
    parser.add_argument('--lidar_holdout', choices=['none','beam'], default='beam',
                        help='Disjoint LiDAR split: prior (input) vs GT (evaluation).')
    parser.add_argument('--lidar_holdout_ratio', type=float, default=0.3,
                        help='Fraction of LiDAR points assigned to GT set when using hold-out.')


    args = parser.parse_args()
    perturb_rng = np.random.default_rng(args.perturb_seed)
    jitter_dx = int(round(args.principal_point_jitter_px[0])) if args.principal_point_jitter_px else 0
    jitter_dy = int(round(args.principal_point_jitter_px[1])) if args.principal_point_jitter_px else 0

    default_out_dir = parser.get_default('out_dir')

    print("\n========== PIPELINE BOOT ==========")
    print(f"[CFG] Camera frames: {args.camera_start} .. {args.camera_end} @ {args.camera_fps:.1f} fps")
    print(f"[CFG] LiDAR  frames: {args.lidar_start} .. {args.lidar_end} @ {args.lidar_fps:.1f} fps")
    print(f"[CFG] Max frames to process: {args.max_frames}")

    # Set output directories from CLI
    global OUT_DIR, DBG_DIR, T_HYSTERESIS
    if args.out_dir == default_out_dir:
        args.out_dir = os.path.join(args.out_dir, args.depth_backend.lower())
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    DBG_DIR = os.path.join(OUT_DIR, "debug")
    os.makedirs(DBG_DIR, exist_ok=True)
    # Set global toggles from CLI
    global USE_EMA, USE_SANITY_CHECKS, USE_MISSED_MINING
    if args.ecw_use_fused:
        args.ecw_source = 'fused'
    USE_EMA = not args.no_ema
    USE_SANITY_CHECKS = not args.no_sanity
    USE_MISSED_MINING = not args.no_mining
    # Set hysteresis at runtime (disable if sanity checks skipped)
    T_HYSTERESIS = args.hysteresis if USE_SANITY_CHECKS else 0.0
    # ...existing code...
    def _load_fx_from_yaml(path="optimized_calibration/camera_optimized.yaml"):
        try:
            with open(path, "r") as f:
                cam = yaml.safe_load(f)
            K = np.array(cam["K"], dtype=np.float64).reshape(3,3)
            return float(K[0,0])
        except Exception as e:
            print(f"[WARN] Could not load fx from {path}: {e}. Using fallback fx=300.")
            return 300.0
    fx_cam = _load_fx_from_yaml()
    print(f"[INFO] fx for ECW: {fx_cam:.2f}")

    # Basic sanity on dirs
    for d in [FRAME_DIR, LIDAR_DIR]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"[ERR] Directory not found: {d}")

    FRONT_VIEW_DIR = os.path.join(FRAME_DIR, 'front')
    if not os.path.isdir(FRONT_VIEW_DIR):
        raise FileNotFoundError(f"[ERR] Front view directory not found: {FRONT_VIEW_DIR}")

    # Frame list (bounded + max_frames)
    frame_candidates = sorted([
        f for f in os.listdir(FRONT_VIEW_DIR)
        if f.endswith('.png') and args.camera_start <= int(os.path.splitext(f)[0]) <= args.camera_end
    ], key=lambda x: int(os.path.splitext(x)[0]))
    total_candidates = len(frame_candidates)
    num_to_process = min(args.max_frames, total_candidates)

    if num_to_process <= 0:
        frame_files = []
    elif total_candidates > num_to_process:
        if args.frame_sampling == 'random':
            sampled = perturb_rng.choice(frame_candidates, size=num_to_process, replace=False)
            frame_files = sorted(sampled.tolist(), key=lambda x: int(os.path.splitext(x)[0]))
            print(f"[INFO] Randomly sampled {num_to_process} frames (seed={args.perturb_seed}) "
                  f"from {total_candidates} candidates in range.")
        else:
            start_idx = int(perturb_rng.integers(0, total_candidates - num_to_process + 1))
            frame_files = frame_candidates[start_idx:start_idx + num_to_process]
            print(f"[INFO] Selected contiguous window [{start_idx}:{start_idx+num_to_process}) "
                  f"(seed={args.perturb_seed}) from {total_candidates} frames.")
    else:
        frame_files = frame_candidates
        print(f"[INFO] Using all {num_to_process} frames available in range.")

    planned_frames = len(frame_files)
    print(f"[INFO] Found {total_candidates} frames in range; will process {planned_frames}")

    # Load models
    print("\n[LOAD] YOLO weights:", YOLO_WEIGHTS)
    yolo_model = load_yolo_model(YOLO_WEIGHTS)
    print("[LOAD] YOLO ready")

    print(f"[LOAD] Depth backend: {args.depth_backend}")
    run_depth, device, depth_name = load_depth_backend(args.depth_backend)
    print(f"[LOAD] Depth ready → backend='{depth_name}', device='{device}'")


    # Build LiDAR index
    lidar_index = {}
    # Only index files within our frame range
    min_frame = args.lidar_start
    max_frame = args.lidar_end
    print(f"[INFO] Building LiDAR index for frames {min_frame} to {max_frame}")

    for p in glob(os.path.join(LIDAR_DIR, '*.npz')):
        stem = os.path.splitext(os.path.basename(p))[0]
        frame_num = int(norm_stem(stem))
        if min_frame <= frame_num <= max_frame:
            lidar_index[str(frame_num)] = p
            print(f"[DEBUG] Added NPZ: {stem} → {frame_num}")

    for p in glob(os.path.join(LIDAR_DIR, '*.pcd')):
        stem = os.path.splitext(os.path.basename(p))[0]
        frame_num = int(norm_stem(stem))
        if min_frame <= frame_num <= max_frame:
            key = str(frame_num)
            if key not in lidar_index:  # Only add PCD if no NPZ exists
                lidar_index[key] = p
                print(f"[DEBUG] Added PCD: {stem} → {frame_num}")

    print(f"[INFO] LiDAR index size: {len(lidar_index)} (npz preferred)")
    if lidar_index:
        sample_items = list(lidar_index.items())[:5]
        for k, v in sample_items:
            print(f"        key={k} → {os.path.basename(v)}")

    def find_matching_lidar(camera_frame: int):
        """Estimate LiDAR frame via FPS ratio; search ±15 frames for available file."""
        rel_cam  = camera_frame - args.camera_start
        guess    = args.lidar_start + int(round(rel_cam * (args.lidar_fps / args.camera_fps)))
        best_key, best_diff = None, 1e9
        for off in range(-15, 16):
            k = str(guess + off)
            if k in lidar_index and abs(off) < best_diff:
                best_key, best_diff = k, abs(off)
                
        # Also search for direct frame number match
        frame_key = str(guess)
        if frame_key in lidar_index:
            return lidar_index[frame_key]
            
        return lidar_index.get(best_key, None)

    # Trackers & constants
    ttc_tracker = TTCTracker()
    ttc_tracker.warning_states = {}  # For persistence/hysteresis tracking
    fps_for_ttc = args.camera_fps
    ego_speed   = 10.33  # m/s (tune for your rig)

    csv_rows = []
    timing_rows = []  # For performance metrics
    episode_states = {}
    episode_logs = []
    episode_id_counter = itertools.count(1)
    processed_frames = 0
    last_timestamp_processed = 0.0

    def finalize_episode(ep_key: str, reason: str):
        state = episode_states.get(ep_key)
        if not state:
            return
        active = state.get("active")
        if not active:
            return
        lead_time = (active.get("warn_time") - active.get("onset_time")) if (active.get("warn_time") is not None and active.get("onset_time") is not None) else np.nan
        episode_logs.append({
            "episode_id": active.get("id"),
            "obj_id": ep_key,
            "class": active.get("class"),
            "onset_frame": active.get("onset_frame"),
            "onset_time": active.get("onset_time"),
            "onset_ttc": active.get("onset_ttc"),
            "warning_frame": active.get("warn_frame"),
            "warning_time": active.get("warn_time"),
            "warning_ttc": active.get("warn_ttc"),
            "lead_time_s": lead_time,
            "warning_issued": bool(active.get("warn_frame") is not None),
            "dzdt_onset": active.get("dzdt_onset"),
            "end_frame": state.get("last_seen_frame"),
            "end_time": state.get("last_seen_time"),
            "end_reason": reason,
            "fusion_mode": args.fusion_mode,
            "ecw_source": args.ecw_source,
            "use_ema": USE_EMA,
            "use_sanity": USE_SANITY_CHECKS,
            "use_mining": USE_MISSED_MINING,
            "hysteresis": T_HYSTERESIS,
            "min_persistence_frames": active.get("min_frames"),
            "ego_speed": ego_speed,
            "depth_backend": depth_name,
            "run_out_dir": OUT_DIR
        })
        state["active"] = None
        state["consec_under"] = 0

    def update_episode_state(ep_key: str, cls_name: str, ttc: float, dzdt: float,
                              ecw_flag: bool, warn_raw: bool, warn_stable: bool,
                              frame_idx_val: int, time_seconds: float):
        state = episode_states.setdefault(ep_key, {
            "consec_under": 0,
            "active": None,
            "last_seen_frame": frame_idx_val,
            "last_seen_time": time_seconds,
            "last_class": cls_name,
        })
        state["last_seen_frame"] = frame_idx_val
        state["last_seen_time"] = time_seconds
        state["last_class"] = cls_name

        approaching = np.isfinite(dzdt) and (dzdt < 0)
        is_valid_ttc = np.isfinite(ttc)
        t_warn_local = get_ttc_threshold(cls_name)
        is_candidate = is_valid_ttc and ecw_flag and approaching

        is_vru = any(k in str(cls_name).lower() for k in VRU_TOKENS)
        min_frames_req = 2 if is_vru else 3

        if is_candidate and (ttc <= t_warn_local):
            state["consec_under"] += 1
        else:
            state["consec_under"] = 0

        if state["consec_under"] >= min_frames_req and state.get("active") is None:
            episode_id = next(episode_id_counter)
            state["active"] = {
                "id": episode_id,
                "class": cls_name,
                "onset_frame": frame_idx_val,
                "onset_time": time_seconds,
                "onset_ttc": float(ttc) if is_valid_ttc else np.nan,
                "dzdt_onset": float(dzdt) if np.isfinite(dzdt) else np.nan,
                "min_frames": min_frames_req,
                "warn_frame": None,
                "warn_time": None,
                "warn_ttc": None
            }

        active = state.get("active")
        if active and warn_stable and active.get("warn_frame") is None:
            active["warn_frame"] = frame_idx_val
            active["warn_time"] = time_seconds
            active["warn_ttc"] = float(ttc) if is_valid_ttc else np.nan

        active = state.get("active")
        if active:
            if not ecw_flag:
                finalize_episode(ep_key, "exit_ecw")
            elif not approaching or not is_valid_ttc:
                finalize_episode(ep_key, "not_approaching")
            elif (state["consec_under"] == 0) and (not warn_raw) and (not warn_stable):
                finalize_episode(ep_key, "ttc_cleared")

    print("\n========== PROCESSING ==========")
    for idx, fname in enumerate(frame_files, 1):
        frame_id = os.path.splitext(fname)[0]
        camera_frame = int(frame_id)
        frame_time_seconds = (camera_frame - args.camera_start) / fps_for_ttc

        # Check if outputs already exist for this frame
        outputs_exist = all(
            os.path.exists(os.path.join(DBG_DIR, f"{frame_id}_{suffix}"))
            for suffix in [
                "fused_depth.npy",
                "mono_depth.npy", 
                "lidar_depth.npy",
                "lidar_mask.npy"
            ]
        ) and os.path.exists(os.path.join(OUT_DIR, f"{frame_id}_overlay.png"))
        
        if outputs_exist:
            print(f"\n[FRAME {idx}/{len(frame_files)}] Camera {frame_id} - Outputs exist, skipping...")
            continue
            
        print(f"\n[FRAME {idx}/{len(frame_files)}] Camera {frame_id}")

        # Choose LiDAR
        lidar_path = find_matching_lidar(camera_frame)
        if lidar_path is None:
            print("  [LIDAR] No match in index → skipping")
            continue
        print("  [LIDAR] Matched file:", os.path.basename(lidar_path))

        # Load image (RGB)
        frame_path = os.path.join(FRONT_VIEW_DIR, fname)
        img = Image.open(frame_path).convert('RGB')
        img_np = np.array(img)
        H, W = img_np.shape[:2]
        print(f"  [IMG] size: {W}x{H}")

        # Detection timing
        t0 = time.perf_counter()
        det_json = os.path.join(DET_CACHE_DIR, f"{frame_id}.json")
        if os.path.exists(det_json):
            with open(det_json, "r") as f_det:
                pred_bboxes = json.load(f_det)
        else:
            raw_boxes = run_obstacle_detection(img_np, yolo_model)
            pred_bboxes = []
            for box in raw_boxes:
                x1, y1, x2, y2, cls_idx, conf = box
                pred_bboxes.append([
                    float(x1), float(y1), float(x2), float(y2), int(cls_idx), float(conf)
                ])
            with open(det_json, "w") as f_det:
                json.dump(pred_bboxes, f_det)
        t1 = time.perf_counter()
        print(f"  [DET] objects: {len(pred_bboxes)}")

        # Save detection visualization
        det_vis_img = img_np.copy()
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls_idx = int(box[4])
            conf = float(box[5])
            cv2.rectangle(det_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{CLASSES[cls_idx]} ({conf:.2f})"
            cv2.putText(det_vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save detection visualization to debug folder
        det_vis_path = os.path.join(DBG_DIR, f"{frame_id}_03_detection.png")
        cv2.imwrite(det_vis_path, cv2.cvtColor(det_vis_img, cv2.COLOR_RGB2BGR))

        # Build detection dicts
        dets = []
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(box[4]); conf = float(box[5])
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2 <= x1 or y2 <= y1:
                continue
            dets.append({'bbox':[x1,y1,x2,y2], 'cls': CLASSES[cls], 'conf': conf})

        # Assign stable IDs
        tracks_this_frame, prev_tracks, next_track_id = _assign_tracks(
            dets, prev_tracks, frame_idx=camera_frame, next_id_start=next_track_id
        )

        # Monocular depth → (H,W)
        depth_start_time = time.perf_counter()
        depth_map = run_depth(img_np)  # returns np.ndarray (H,W) in arbitrary units
        depth_end_time = time.perf_counter()
        depth_comp_time = depth_end_time - depth_start_time  # Computation time in seconds
        
        t2 = time.perf_counter()
        if depth_map.shape != (H, W):
            print(f"  [DEPTH] resize {depth_name} {depth_map.shape} → {(H, W)}")
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"  [PERF] Depth computation time: {depth_comp_time*1000:.2f} ms")

        # LiDAR projection products
        if lidar_path.endswith('.npz'):
            if not os.path.exists(lidar_path):
                print(f"  [LIDAR][ERR] Missing npz: {lidar_path}")
                continue
            lidar_data = np.load(lidar_path)
            if 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
                Dlidar = lidar_data['Dlidar']
                Mlidar = lidar_data['Mlidar'].astype(bool)
            elif 'depth' in lidar_data and 'mask' in lidar_data:
                Dlidar = lidar_data['depth']
                Mlidar = lidar_data['mask'].astype(bool)
            else:
                print(f"  [LIDAR][ERR] Unexpected keys in {os.path.basename(lidar_path)}")
                continue

        elif lidar_path.endswith('.pcd'):
            out_npz = os.path.join(LIDAR_DIR, f"{frame_id}.npz")
            debug_overlay = os.path.join(LIDAR_DIR, f"{frame_id}_overlay.png")
            cmd = (
                f"python lidar_projection/project_lidar.py "
                f"--pcd '{lidar_path}' "
                f"--cam_yaml optimized_calibration/camera_optimized.yaml "
                f"--ext_yaml optimized_calibration/extrinsics_optimized.yaml "
                f"--image {frame_path} "
                f"--out_npz {out_npz} "
                f"--debug_overlay {debug_overlay} "
            )
            print(f"  [LIDAR][RUN] {cmd}")
            ret = os.system(cmd)
            if ret != 0 or (not os.path.exists(out_npz)):
                print(f"  [LIDAR][ERR] projection failed (ret={ret})")
                continue
            lidar_data = np.load(out_npz)
            if 'Dlidar' in lidar_data and 'Mlidar' in lidar_data:
                Dlidar = lidar_data['Dlidar']
                Mlidar = lidar_data['Mlidar'].astype(bool)
            else:
                print(f"  [LIDAR][ERR] Bad npz after projection")
                continue
        else:
            print("  [LIDAR][ERR] Unknown file type")
            continue

        # Resize LiDAR to image grid
        if Dlidar.shape != (H, W):
            Dlidar_small = cv2.resize(Dlidar, (W, H), interpolation=cv2.INTER_NEAREST)
            Mlidar_small = cv2.resize(Mlidar.astype(np.uint8), (W, H),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            Dlidar_small, Mlidar_small = Dlidar, Mlidar

        if abs(args.lidar_yaw_jitter_deg) > 0:
            approx_deg_per_px = 0.2
            shift_px = int(round(args.lidar_yaw_jitter_deg / approx_deg_per_px))
            if shift_px != 0:
                Dlidar_small = np.roll(Dlidar_small, shift_px, axis=1)
                Mlidar_small = np.roll(Mlidar_small, shift_px, axis=1)
                if shift_px > 0:
                    Mlidar_small[:, :shift_px] = False
                    Dlidar_small[:, :shift_px] = 0.0
                else:
                    Mlidar_small[:, shift_px:] = False
                    Dlidar_small[:, shift_px:] = 0.0
                print(f"  [PERTURB] Applied yaw jitter shift_px={shift_px}")

        if args.lidar_dropout_pct > 0:
            dropout_mask = perturb_rng.random(Mlidar_small.shape) < args.lidar_dropout_pct
            Mlidar_small[dropout_mask] = False
            Dlidar_small[dropout_mask] = 0.0
            print(f"  [PERTURB] Applied LiDAR dropout {args.lidar_dropout_pct*100:.1f}%")

        def split_lidar_mask(mask_bool, ratio=0.3):
            rows, cols = np.nonzero(mask_bool)
            if rows.size == 0:
                return mask_bool.copy(), np.zeros_like(mask_bool, dtype=bool)
            h = (rows * 1315423911 + cols * 2654435761) & 0xFFFFFFFF
            frac = (h.astype(np.uint64) % 10000) / 10000.0
            m_gt = np.zeros_like(mask_bool, dtype=bool)
            m_gt[rows[frac < ratio], cols[frac < ratio]] = True
            m_prior = mask_bool & (~m_gt)
            return m_prior, m_gt

        if args.lidar_holdout == 'beam':
            holdout_ratio = float(np.clip(args.lidar_holdout_ratio, 0.0, 1.0))
            Mlidar_prior, Mlidar_gt = split_lidar_mask(Mlidar_small, ratio=holdout_ratio)
        else:
            holdout_ratio = 0.0
            Mlidar_prior = Mlidar_small.copy()
            Mlidar_gt = np.zeros_like(Mlidar_small, dtype=bool)

        Dlidar_prior = np.where(Mlidar_prior, Dlidar_small, np.nan).astype(np.float32)
        Dlidar_gt    = np.where(Mlidar_gt,    Dlidar_small, np.nan).astype(np.float32)

        valid = int(Mlidar_prior.sum())
        print(f"  [LIDAR] valid input px: {valid} (GT hold-out {int(Mlidar_gt.sum())}, ratio={holdout_ratio:.2f})")
        if valid == 0:
            print("  [LIDAR][WARN] 0 prior px — calibration/sync suspect")

        # --- Convert MiDaS relative to metric meters using LiDAR overlap ---
        eps = 1e-6
        if args.depth_backend == 'midas':
            overlap = Mlidar_prior & np.isfinite(depth_map) & np.isfinite(Dlidar_prior) & (Dlidar_prior > 0)
            if overlap.sum() >= 200:
                r = depth_map[overlap].astype(np.float32)
                z = Dlidar_prior[overlap].astype(np.float32)
                r_lo, r_hi = np.percentile(r, [2, 98])
                if r_hi <= r_lo:
                    r_lo, r_hi = float(r.min()), float(r.max())
                x = (r - r_lo) / (r_hi - r_lo + eps)
                x = np.clip(x, 0.0, 1.0)
                y = 1.0 / np.clip(z, 0.05, 200.0)
                y_med = np.median(y)
                mad = np.median(np.abs(y - y_med)) + eps
                keep = np.abs(y - y_med) <= 3.5 * mad
                x, y = x[keep], y[keep]
                if x.size > 30000:
                    idx = np.random.choice(x.size, 30000, replace=False)
                    x, y = x[idx], y[idx]
                # Robust regression using Huber loss
                from scipy.optimize import least_squares
                def huber_residual(params, x, y, delta=1.0):
                    pred = params[0] * x + params[1]
                    res = y - pred
                    abs_res = np.abs(res)
                    mask = abs_res <= delta
                    out = np.empty_like(res)
                    out[mask] = 0.5 * res[mask] ** 2
                    out[~mask] = delta * (abs_res[~mask] - 0.5 * delta)
                    return out

                def residual(params):
                    return y - (params[0] * x + params[1])

                # Use scipy's least_squares with 'huber' loss
                res = least_squares(residual, x0=[1.0, 0.0], loss='huber', f_scale=1.0)
                A, B = float(res.x[0]), float(res.x[1])
                r_full = depth_map.astype(np.float32)
                x_full = (r_full - r_lo) / (r_hi - r_lo + eps)
                x_full = np.clip(x_full, 0.0, 1.0)
                depth_mono_m = 1.0 / np.clip(A * x_full + B, 1e-4, 1e4)
                depth_mono_m = np.clip(depth_mono_m, 0.1, 120.0)
                print(f"  [CAL][Huber] fit AB using {x.size} pts: A={A:.6f}, B={B:.6f} -> min≈{1.0/(A+B+eps):.2f} m, max≈{1.0/(B+eps):.2f} m (from overlap)")
            else:
                print("  [CAL][WARN] insufficient overlap; using median-ratio scale fallback")
                overlap2 = Mlidar_prior & np.isfinite(depth_map) & (depth_map > 0)
                if overlap2.sum() >= 50:
                    scale = float(np.nanmedian(Dlidar_prior[overlap2] / (depth_map[overlap2] + eps)))
                else:
                    scale = 1.0
                depth_mono_m = np.clip(depth_map * scale, 0.1, 120.0)
        else:
            overlap = Mlidar_prior & np.isfinite(depth_map) & (depth_map > 0)
            if overlap.sum() >= 50:
                scale = float(np.nanmedian(Dlidar_prior[overlap] / (depth_map[overlap] + eps)))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
            else:
                scale = 1.0
            depth_mono_m = np.clip(depth_map * scale, 0.1, 80.0)
            print(f"  [CAL] overlap={int(overlap.sum())}, mono_scale={scale:.3f}")

        if jitter_dx != 0 or jitter_dy != 0:
            if jitter_dy != 0:
                depth_mono_m = np.roll(depth_mono_m, jitter_dy, axis=0)
                if jitter_dy > 0:
                    depth_mono_m[:jitter_dy, :] = np.nan
                else:
                    depth_mono_m[jitter_dy:, :] = np.nan
            if jitter_dx != 0:
                depth_mono_m = np.roll(depth_mono_m, jitter_dx, axis=1)
                if jitter_dx > 0:
                    depth_mono_m[:, :jitter_dx] = np.nan
                else:
                    depth_mono_m[:, jitter_dx:] = np.nan
            print(f"  [PERTURB] Applied principal point jitter dx={jitter_dx}px dy={jitter_dy}px")

        # Fuse depths
        fused_depth, Wlidar, Mfused = fuse_confidence(Dlidar_prior, Mlidar_prior, depth_mono_m)
        print(f"  [FUSE] mean LiDAR weight: {float(np.nanmean(Wlidar)):.3f}; fused finite px: {int(np.isfinite(fused_depth).sum())}")

        # Save dense depth maps for evaluation
        np.save(os.path.join(DBG_DIR, f"{frame_id}_fused_depth.npy"), fused_depth)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_mono_depth.npy"), depth_mono_m)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_depth.npy"), Dlidar_small)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_mask.npy"), Mlidar_small.astype(np.uint8))
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_prior_depth.npy"), Dlidar_prior)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_prior_mask.npy"), Mlidar_prior.astype(np.uint8))
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_gt_depth.npy"), Dlidar_gt)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_gt_mask.npy"), Mlidar_gt.astype(np.uint8))
        print(f"  [SAVE] depth maps → {frame_id}_*_depth.npy")
        t3 = time.perf_counter()  # After fusion and depth save

        # Prepare per-mode depth maps for downstream logic/metrics
        lidar_metric = np.full_like(Dlidar_prior, np.nan, dtype=np.float32)
        lidar_metric[Mlidar_prior] = Dlidar_prior[Mlidar_prior]

        late_depth = np.where(
            np.isfinite(lidar_metric),
            0.5 * (lidar_metric + depth_mono_m),
            depth_mono_m
        )

        depth_modes = {
            "mono": depth_mono_m,
            "lidar": lidar_metric,
            "late": late_depth,
            "ours": fused_depth
        }
        pred_depth_map = depth_modes.get(args.fusion_mode, fused_depth)
        ecw_depth_map = depth_modes.get(args.ecw_source, fused_depth)

        # ---------- Missed obstacle detection inside ECW (depth blobs not covered by YOLO) ----------
        # 3.1 ECW mask (trapezoid ahead of ego)
        ECW_mask = make_ecw_mask(H, W, ecw_polygon)
        ECW_poly = ecw_polygon

        # 3.2 Detected-objects mask (dilate to give YOLO some margin)
        det_mask = np.zeros((H, W), dtype=np.uint8)
        for box in pred_bboxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            det_mask[y1:y2+1, x1:x2+1] = 1
        if args.miss_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.miss_dilate, args.miss_dilate))
            det_mask = cv2.dilate(det_mask, k, iterations=1).astype(bool)
        else:
            det_mask = det_mask.astype(bool)

        # 3.3 Enhanced candidate detection for Indian traffic scenarios
        map_for_ecw = ecw_depth_map if ecw_depth_map is not None else fused_depth
        valid_depth = np.isfinite(map_for_ecw)
        
        # Fix depth range conditions (nearer than far limit, farther than near limit)
        near = map_for_ecw <= args.miss_far_m
        far  = map_for_ecw >= args.miss_near_m
        
        # Ground plane removal with slope consideration
        not_ground = remove_ground(map_for_ecw, fx_cam, fx_cam, slope=args.road_slope)
        
        # Motion detection (if enabled and have previous frame)
        motion_cue = np.zeros_like(ECW_mask)
        flow_magnitudes = None
        if args.use_motion:
            if 'prev_frame' in locals():
                motion_mask, flow_magnitudes = compute_motion_mask(prev_frame, img_np, min_flow=args.flow_thresh)
                motion_cue = motion_mask & ECW_mask & (~det_mask)
        prev_frame = img_np.copy()  # Save for next iteration
        
        missed_boxes = []
        if USE_MISSED_MINING:
            # Combine all cues
            depth_cue = ECW_mask & valid_depth & near & far & (~det_mask) & not_ground

            # Final candidate mask combines depth and motion cues
            cand = depth_cue | motion_cue

            # Optional smoothing / closing to connect sparse LiDAR projections
            if cand.any():
                k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, k2, iterations=1).astype(bool)

            if cand.any():
                # 3.4 Connected components to get blobs
                num, labels = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
                for lab in range(1, num):
                    blob = labels == lab
                    area = int(blob.sum())
                    if area < args.miss_min_px:
                        continue

                    # Bounding box
                    ys, xs = np.where(blob)
                    y1, y2 = int(ys.min()), int(ys.max())
                    x1, x2 = int(xs.min()), int(xs.max())

                    # Median fused depth (meters) within blob
                    z_vals = fused_depth[blob]
                    z_vals = z_vals[np.isfinite(z_vals)]
                    if z_vals.size == 0:
                        continue
                    z_med = float(np.median(z_vals))

                    # Physical size gating
                    width_m, height_m = box_size_meters(x1, y1, x2, y2, z_med, fx_cam, fx_cam)
                    if not (0.2 <= width_m <= 2.5 and 0.2 <= height_m <= 3.0):
                        continue  # Skip if outside human-like size range

                    # Compute hazard score
                    blob_mask = blob.astype(bool)

                    # Depth score (1 if inside valid range)
                    s_depth = 1.0 if (args.miss_near_m <= z_med <= args.miss_far_m) else 0.0

                    # Motion score (average flow magnitude in blob)
                    s_motion = 0.0
                    if args.use_motion and flow_magnitudes is not None:
                        flow_in_blob = flow_magnitudes[blob_mask]
                        s_motion = float(flow_in_blob.mean()) / args.flow_thresh if flow_in_blob.size > 0 else 0.0

                    # Combined hazard score
                    w_depth, w_motion = 0.6, 0.4  # Weights for depth and motion
                    hazard_score = w_depth * s_depth + w_motion * s_motion

                    if hazard_score >= 0.5:  # Only keep high-scoring candidates
                        missed_boxes.append((x1, y1, x2, y2, z_med, width_m, height_m, hazard_score))

        # Debug panels
        # Use depth_mono_m for visualization
        dm = depth_mono_m.copy()
        dm[~np.isfinite(dm)] = 0
        dm_norm = (dm - np.nanmin(dm)) / (np.nanmax(dm) - np.nanmin(dm) + 1e-6)
        mono_color = cv2.applyColorMap((dm_norm*255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        mono_color = cv2.cvtColor(mono_color, cv2.COLOR_BGR2RGB)

        dl = Dlidar_small.copy()
        dl[~Mlidar_small] = 0
        nonzero = dl[Mlidar_small]
        if nonzero.size > 0:
            lo, hi = np.percentile(nonzero, [2, 98])
            dl_clip = np.clip(dl, lo, hi)
        else:
            lo, hi = 0, 1
            dl_clip = dl
        dl_norm = (dl_clip - lo) / (hi - lo + 1e-6)
        lidar_color = cv2.applyColorMap((dl_norm*255).astype(np.uint8), cv2.COLORMAP_TURBO)
        lidar_color = cv2.cvtColor(lidar_color, cv2.COLOR_BGR2RGB)
        lidar_color[~Mlidar_small] = (0,0,0)

        overlay_mask = img_np.copy()
        overlay_mask[Mlidar_small] = (overlay_mask[Mlidar_small] * 0.3 + np.array([0,255,0])*0.7).astype(np.uint8)

        cv2.imwrite(os.path.join(DBG_DIR, f"{frame_id}_00_image.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(DBG_DIR, f"{frame_id}_01_mono_depth.png"), cv2.cvtColor(mono_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(DBG_DIR, f"{frame_id}_02_lidar_depth.png"), cv2.cvtColor(lidar_color, cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(DBG_DIR, f"{frame_id}_03_lidar_mask_on_image.png"), cv2.cvtColor(overlay_mask, cv2.COLOR_RGB2BGR))

        # Per-object metrics
        bboxes, classes, confidences = [], [], []
        lidar_depths, mono_depths, ttc_list, ecw_bubbles = [], [], [], []
        base_obj_count = 0  # Initialize counter for TTC tracking


        seen_episode_keys = set()

        for t in tracks_this_frame:
            x1, y1, x2, y2 = t['bbox']
            cls_name = t['cls']
            conf = t['conf']
            track_id = t['track_id']
            obj_key = f"det_{track_id}"
            seen_episode_keys.add(obj_key)
            # inner 40% patch (avoid edges)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            dx = max(1, (x2-x1)//5); dy = max(1, (y2-y1)//5)
            xi1, yi1 = cx - 2*dx, cy - 2*dy
            xi2, yi2 = cx + 2*dx, cy + 2*dy
            xi1 = max(0, xi1); yi1 = max(0, yi1); xi2 = min(W-1, xi2); yi2 = min(H-1, yi2)

            mono_patch = depth_mono_m[yi1:yi2, xi1:xi2]
            lidar_patch_prior = Dlidar_prior[yi1:yi2, xi1:xi2]
            mask_patch_prior  = Mlidar_prior[yi1:yi2, xi1:xi2]
            lidar_patch_gt = Dlidar_gt[yi1:yi2, xi1:xi2]
            mask_patch_gt  = Mlidar_gt[yi1:yi2, xi1:xi2]

            mono_vals  = mono_patch[np.isfinite(mono_patch)]
            lidar_vals = lidar_patch_gt[mask_patch_gt]
            mono_depth_val  = float(np.median(mono_vals))  if mono_vals.size  > 0 else np.nan
            lidar_depth_val = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan

            bboxes.append([x1, y1, x2, y2])
            classes.append(cls_name)
            confidences.append(conf)
            mono_depths.append(mono_depth_val)
            lidar_depths.append(lidar_depth_val)

            # TTC (time base = camera fps)
            lidar_hits = int(mask_patch_prior.sum())
            d_lidar = robust_box_depth(Dlidar_prior, [x1,y1,x2,y2], mask=Mlidar_prior)
            d_fused = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            d_lidar_gt = robust_box_depth(Dlidar_gt, [x1,y1,x2,y2], mask=Mlidar_gt)
            K_MIN = 30
            if lidar_hits < K_MIN or not np.isfinite(d_lidar):
                d_use = d_fused
            else:
                d_use = d_lidar

            # EMA smoothing before TTC
            state = ttc_tracker.warning_states.setdefault(
                f"det_{track_id}",
                {'consecutive_frames':0,'warning_active':False,'last_ttc':None,'ema_depth':None}
            )
            if USE_EMA:
                alpha = 0.4
                if state['ema_depth'] is None or not np.isfinite(state['ema_depth']):
                    d_smooth = d_use
                else:
                    d_smooth = alpha * d_use + (1 - alpha) * state['ema_depth']
                state['ema_depth'] = d_smooth
            else:
                d_smooth = d_use
                state['ema_depth'] = None

            t_now = frame_time_seconds
            if np.isfinite(d_smooth):
                ttc, dzdt = ttc_tracker.update_and_compute(track_id, d_smooth, t_now, ego_speed=ego_speed)
            else:
                ttc, dzdt = np.nan, np.nan
            ttc_list.append(ttc)

            ecw, _ = compute_ecw_bubble([x1, y1, x2, y2], ecw_depth_map, threshold=args.ecw_depth_threshold)
            ecw_bubbles.append(ecw)

            d_fused_box = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            d_pred_box = robust_box_depth(pred_depth_map, [x1,y1,x2,y2], mask=np.isfinite(pred_depth_map))
            depth_for_size = d_pred_box if np.isfinite(d_pred_box) else d_use
            width_m, height_m = box_size_meters(x1, y1, x2, y2, depth_for_size, fx_cam, fx_cam)
            meets_size = check_object_size(width_m, height_m)
            patch_mask = np.isfinite(fused_depth[yi1:yi2, xi1:xi2])
            depth_valid_px = int(patch_mask.sum())

            if USE_SANITY_CHECKS:
                warn_raw, warn_stable = compute_warning_state(
                    f"det_{track_id}", ttc, cls_name, ttc_tracker.warning_states
                )
            else:
                warn_raw = np.isfinite(ttc) and (ttc <= get_ttc_threshold(cls_name))
                warn_stable = warn_raw
                state['consecutive_frames'] = 1 if warn_stable else 0
                state['warning_active'] = warn_stable
                state['last_ttc'] = ttc

            update_episode_state(
                ep_key=obj_key,
                cls_name=cls_name,
                ttc=ttc,
                dzdt=dzdt,
                ecw_flag=bool(ecw),
                warn_raw=warn_raw,
                warn_stable=warn_stable,
                frame_idx_val=camera_frame,
                time_seconds=t_now
            )

            # Get detailed LiDAR stats for the box
            lidar_in_box = lidar_patch_prior[mask_patch_prior]
            lidar_stats = {
                'lidar_points': len(lidar_in_box),
                'lidar_density': len(lidar_in_box) / ((x2-x1+1) * (y2-y1+1)) if lidar_in_box.size else 0.0,
                'lidar_std': float(np.std(lidar_in_box)) if len(lidar_in_box) > 0 else np.nan,
                'lidar_completeness': len(lidar_in_box) / mask_patch_prior.size if mask_patch_prior.size else 0.0
            }
            
            # Confidence and fusion weight metrics
            d_mono_box = robust_box_depth(depth_mono_m, [x1,y1,x2,y2], mask=np.isfinite(depth_mono_m))
            depth_agreement = 1.0 - abs(d_pred_box - d_lidar_gt) / (d_lidar_gt + 1e-6) if np.isfinite(d_lidar_gt) and np.isfinite(d_pred_box) else 0.0
            
            # Calculate LiDAR confidence based on point density and completeness
            lidar_conf = (lidar_stats['lidar_density'] * lidar_stats['lidar_completeness']) if lidar_stats['lidar_points'] > 0 else 0.0
            
            # Calculate camera confidence based on detection confidence and depth consistency
            camera_conf = conf * (1.0 - (abs(mono_depth_val - d_fused_box) / (d_fused_box + 1e-6))) if np.isfinite(d_fused_box) and np.isfinite(mono_depth_val) else 0.0
            
            # Compute computational metrics
            if 'depth_comp_time' in locals():
                inference_time_ms = depth_comp_time * 1000  # Convert to ms
            else:
                inference_time_ms = np.nan
                
            # Calculate depth estimation stability
            if prev_tracks and track_id in prev_tracks:
                prev_depth = prev_tracks[track_id].get('last_depth', np.nan)
                depth_stability = 1.0 - abs(d_fused_box - prev_depth) / (prev_depth + 1e-6) if np.isfinite(prev_depth) else 0.0
            else:
                depth_stability = 0.0
            
            # Update track history
            if track_id in prev_tracks:
                prev_tracks[track_id]['last_depth'] = d_fused_box
            
            # Get fusion weights from the region, ensuring proper dimensions
            try:
                fusion_weights = Wlidar[y1:y2, x1:x2]
                # Ensure mask_patch matches fusion_weights dimensions
                mask_region = np.zeros_like(fusion_weights, dtype=bool)
                # Get the minimum dimensions to avoid index errors
                h, w = min(mask_patch_prior.shape[0], fusion_weights.shape[0]), min(mask_patch_prior.shape[1], fusion_weights.shape[1])
                mask_region[:h, :w] = mask_patch_prior[:h, :w]
                
                lidar_weight = float(np.mean(fusion_weights[mask_region])) if mask_region.any() else 0.0
                camera_weight = 1.0 - lidar_weight  # Camera weight is complement of LiDAR weight
                
                print(f"    [FUSION] Box dims: {fusion_weights.shape}, Mask dims: {mask_patch_prior.shape}, "
                      f"LiDAR weight: {lidar_weight:.3f}")
            except Exception as e:
                print(f"    [FUSION][WARN] Weight computation failed: {str(e)}")
                lidar_weight = 0.0
                camera_weight = 1.0
            
            # Motion analysis if available
            motion_score = 0.0
            if args.use_motion and 'flow_magnitudes' in locals() and flow_magnitudes is not None:
                box_flow = flow_magnitudes[y1:y2, x1:x2]
                motion_score = float(np.median(box_flow)) if box_flow.size > 0 else 0.0
            
            # Detection stability metrics
            track_info = ttc_tracker.warning_states.get(f"det_{track_id}", {})
            detection_history = track_info.get('consecutive_frames', 0)
            
            csv_rows.append({
                # Basic identifiers
                "frame": frame_id,
                "obj_id": f"det_{track_id}",
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "source": "det",
                
                # Depth measurements
                "mono_median_depth": mono_depth_val,
                "lidar_median_depth": lidar_depth_val, 
                "fused_median_depth": float(d_fused_box) if np.isfinite(d_fused_box) else np.nan,
                "pred_median_depth": float(d_pred_box) if np.isfinite(d_pred_box) else np.nan,
                "pred_depth_source": args.fusion_mode,
                "depth_agreement": depth_agreement,
                
                # Confidence scores
                "lidar_confidence": lidar_conf,
                "camera_confidence": camera_conf,
                "detection_confidence": conf,
                
                # Fusion weights
                "lidar_weight": lidar_weight,
                "camera_weight": camera_weight,
                
                # System performance metrics
                "inference_time_ms": inference_time_ms,
                "depth_stability": depth_stability,
                "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024) if 'psutil' in sys.modules else np.nan,
                
                # LiDAR detailed stats
                "lidar_point_count": lidar_stats['lidar_points'],
                "lidar_density": lidar_stats['lidar_density'],
                "lidar_std": lidar_stats['lidar_std'],
                "lidar_completeness": lidar_stats['lidar_completeness'],
                
                # Object metrics
                "ttc": ttc,
                "in_ecw": bool(ecw),
                "depth_valid_px": depth_valid_px,
                "meets_size": meets_size,
                "width_m": width_m,
                "height_m": height_m,
                
                # Warning states
                "warn_raw": warn_raw,
                "warn_stable": warn_stable,
                "warn_raw_fused": warn_raw,
                "warn_stable_fused": warn_stable,
                
                # Motion and stability
                "motion_score": motion_score,
                "detection_history": detection_history,
                
                # Timestamp info
                "timestamp": t_now,
                "ego_speed": ego_speed
            })

        print(f"    [OBJ][{track_id}] class={cls_name}, conf={conf:.2f}, mono_depth={mono_depth_val:.2f}m, lidar_depth={lidar_depth_val:.2f}m")

        missing_episode_keys = [key for key in list(episode_states.keys()) if key not in seen_episode_keys]
        for mkey in missing_episode_keys:
            mstate = episode_states.get(mkey)
            if not mstate:
                continue
            if mstate.get("active") is not None:
                finalize_episode(mkey, "track_lost")
            # drop states that are idle to keep map compact
            if episode_states.get(mkey, {}).get("active") is None:
                episode_states.pop(mkey, None)

        print(f"  [OBJ] boxes: {len(bboxes)}")

        # ---------- Add missed obstacles as pseudo-objects ----------
        base_obj_count = len(bboxes)  # Get current count for continuing indices
        for j, (x1, y1, x2, y2, z_med, width_m, height_m, hazard_score) in enumerate(missed_boxes):
            # guard tight coords
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2 <= x1 or y2 <= y1:
                continue

            # Per-box medians from maps (consistent with regular flow)
            mono_patch = depth_mono_m[y1:y2, x1:x2]
            lidar_patch_prior = Dlidar_prior[y1:y2, x1:x2]
            mask_patch_prior  = Mlidar_prior[y1:y2, x1:x2]
            lidar_patch_gt = Dlidar_gt[y1:y2, x1:x2]
            mask_patch_gt  = Mlidar_gt[y1:y2, x1:x2]

            mono_vals  = mono_patch[np.isfinite(mono_patch)]
            lidar_vals = lidar_patch_gt[mask_patch_gt]
            mono_depth_val  = float(np.median(mono_vals))  if mono_vals.size  > 0 else np.nan
            lidar_depth_val = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan

            # TTC using fused depth (preferred) or LiDAR-in-box if available
            t_now = frame_time_seconds
            d_fused = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            d_pred_box = robust_box_depth(pred_depth_map, [x1,y1,x2,y2], mask=np.isfinite(pred_depth_map))
            d_use = d_pred_box if np.isfinite(d_pred_box) else (d_fused if np.isfinite(d_fused) else z_med)
            if np.isfinite(d_use):
                ttc, dzdt = ttc_tracker.update_and_compute(base_obj_count + j, d_use, t_now, ego_speed=ego_speed)
            else:
                ttc = np.nan
                dzdt = np.nan
            ttc_list.append(ttc)
            # Append as a new “missed” class with low confidence
            bboxes.append([x1, y1, x2, y2])
            classes.append("Missed")         # or "Unknown"
            confidences.append(0.01)         # visualization only
            mono_depths.append(mono_depth_val)
            lidar_depths.append(lidar_depth_val)

            # Check if object is in ECW region
            in_ecw, _ = compute_ecw_bubble([x1, y1, x2, y2], ecw_depth_map, threshold=args.ecw_depth_threshold)
            ecw_bubbles.append(in_ecw)
            
            # Get depth confidence
            patch_mask = np.isfinite(fused_depth[y1:y2, x1:x2])
            depth_valid_px = int(patch_mask.sum())
            
            # Physical size check
            depth_for_size = d_pred_box if np.isfinite(d_pred_box) else d_use
            width_m, height_m = box_size_meters(x1, y1, x2, y2, depth_for_size, fx_cam, fx_cam)
            meets_size = check_object_size(width_m, height_m)
            
            # Compute warning states with persistence and hysteresis
            miss_id = f"miss_{base_obj_count + j}"
            if USE_SANITY_CHECKS:
                warn_raw, warn_stable = compute_warning_state(
                    miss_id, ttc, "Missed",
                    ttc_tracker.warning_states
                )
            else:
                warn_raw = np.isfinite(ttc) and (ttc <= get_ttc_threshold("Missed"))
                warn_stable = warn_raw
                state = ttc_tracker.warning_states.setdefault(
                    miss_id,
                    {'consecutive_frames':0,'warning_active':False,'last_ttc':None,'ema_depth':None}
                )
                state['consecutive_frames'] = 1 if warn_stable else 0
                state['warning_active'] = warn_stable
                state['last_ttc'] = ttc

            # For missed blobs, append CSV with fused depth
            csv_rows.append({
                "frame": frame_id,
                "obj_id": f"miss_{base_obj_count + j}",  # Unique object ID
                "class": "Missed",
                "confidence": 0.01,
                "bbox": (x1, y1, x2, y2),
                "mono_median_depth": mono_depth_val,
                "lidar_median_depth": lidar_depth_val,
                "fused_median_depth": float(d_fused) if np.isfinite(d_fused) else np.nan,
                "pred_median_depth": float(d_pred_box) if np.isfinite(d_pred_box) else np.nan,
                "pred_depth_source": args.fusion_mode,
                "ttc": ttc,
                "in_ecw": in_ecw,
                "depth_valid_px": depth_valid_px,
                "meets_size": meets_size,
                "width_m": width_m,
                "height_m": height_m,
                "warn_raw": warn_raw,
                "warn_stable": warn_stable,
                "source": "miss",
                "hazard_score": hazard_score,
                "flow_magnitude": float(s_motion) if args.use_motion else np.nan
            })

            print(f"    [MISS][{j}] bbox=({x1},{y1},{x2},{y2}) median_fused≈{z_med:.2f}m TTC={ttc:.2f}s")

        t4 = time.perf_counter()  # After ECW processing

        # Collect timing info
        timing_rows.append({
            "frame": frame_id,
            "t_det_ms": 1000*(t1-t0),
            "t_depth_ms": 1000*(t2-t1),
            "t_fuse_ms": 1000*(t3-t2),
            "t_ecw_ms": 1000*(t4-t3),
            "t_total_ms": 1000*(t4-t0),
            "backend": depth_name
        })

        # Final overlay
        base_rgb = paint_depth_background(img_np, depth_mono_m, mask=np.isfinite(depth_mono_m),
                                          cmap=cv2.COLORMAP_MAGMA)
        base_rgb = paint_depth_background(base_rgb, Dlidar_small, mask=Mlidar_small,
                                          alpha_fg=0.9, alpha_bg=0.1, cmap=cv2.COLORMAP_TURBO)

        # draw ECW polygon for debug (on base_rgb)
        base_dbg = base_rgb.copy()
        cv2.polylines(base_dbg, [ECW_poly], isClosed=True, color=(255,255,0), thickness=2)
        final_img = overlay_results(
            base_dbg, bboxes, classes, confidences,
            lidar_depths, mono_depths, ttc_list, ecw_bubbles
        )

        out_img_path = os.path.join(OUT_DIR, frame_id + '_overlay.png')
        cv2.imwrite(out_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE] overlay → {out_img_path}")
        processed_frames += 1
        last_timestamp_processed = frame_time_seconds

    for ep_key, ep_state in list(episode_states.items()):
        if ep_state.get("active") is not None:
            finalize_episode(ep_key, "end_of_run")
        if episode_states.get(ep_key, {}).get("active") is None and ep_key in episode_states:
            # remove idle holder to keep structure minimal
            episode_states.pop(ep_key, None)

    # Save metrics CSVs
    csv_path = os.path.join(OUT_DIR, 'object_depth_metrics.csv')
    
    # If there are existing metrics, load and combine them
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        new_df = pd.DataFrame(csv_rows)
        
        # Combine existing and new data, removing duplicates based on frame and obj_id
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['frame', 'obj_id'], keep='last')
        combined_df.to_csv(csv_path, index=False)
    else:
        # If no existing file, save new data directly
        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)
    
    print('\n[CSV] saved metrics:', csv_path)

    # Save timing CSV
    timing_path = os.path.join(OUT_DIR, 'timing.csv')
    
    # If there are existing timing metrics, load and combine them
    if os.path.exists(timing_path):
        existing_timing = pd.read_csv(timing_path)
        new_timing = pd.DataFrame(timing_rows)
        
        # Combine existing and new timing data, removing duplicates based on frame
        combined_timing = pd.concat([existing_timing, new_timing]).drop_duplicates(subset=['frame'], keep='last')
        combined_timing.to_csv(timing_path, index=False)
    else:
        # If no existing file, save new timing data directly
        df_timing = pd.DataFrame(timing_rows)
        df_timing.to_csv(timing_path, index=False)
    
    print('[CSV] saved timing:', timing_path)

    episodes_path = os.path.join(OUT_DIR, 'episodes.csv')
    if episode_logs:
        ep_df = pd.DataFrame(episode_logs)
        if os.path.exists(episodes_path):
            existing_ep = pd.read_csv(episodes_path)
            ep_df = pd.concat([existing_ep, ep_df], ignore_index=True)
        ep_df = ep_df.drop_duplicates(subset=['run_out_dir','episode_id'], keep='last')
        ep_df.to_csv(episodes_path, index=False)
        print('[CSV] saved episodes:', episodes_path)
    else:
        print('[CSV] no episodes to log')

    summary_path = os.path.join(OUT_DIR, 'run_summary.json')
    duration_est = processed_frames / args.camera_fps if args.camera_fps > 0 else 0.0
    duration_est = max(duration_est, last_timestamp_processed if processed_frames else 0.0)
    run_summary = {
        "camera_start": args.camera_start,
        "camera_end": args.camera_end,
        "lidar_start": args.lidar_start,
        "lidar_end": args.lidar_end,
        "frames_planned": planned_frames,
        "frames_processed": processed_frames,
        "duration_s": duration_est,
        "duration_hours": duration_est / 3600.0 if duration_est else 0.0,
        "fusion_mode": args.fusion_mode,
        "ecw_source": args.ecw_source,
        "depth_backend": depth_name,
        "use_ema": USE_EMA,
        "use_sanity": USE_SANITY_CHECKS,
        "use_mining": USE_MISSED_MINING,
        "hysteresis": T_HYSTERESIS,
        "lidar_dropout_pct": args.lidar_dropout_pct,
        "lidar_yaw_jitter_deg": args.lidar_yaw_jitter_deg,
        "principal_point_jitter_px": [jitter_dx, jitter_dy],
        "perturb_seed": args.perturb_seed,
        "lidar_holdout": args.lidar_holdout,
        "lidar_holdout_ratio": args.lidar_holdout_ratio,
        "run_out_dir": OUT_DIR
    }
    with open(summary_path, 'w', encoding='utf-8') as f_summary:
        json.dump(run_summary, f_summary, indent=2)
    print('[JSON] saved run summary:', summary_path)

    # Video
    video_path = os.path.join(OUT_DIR, 'output_visualization.mp4')
    create_video(OUT_DIR, video_path, fps=int(round(args.camera_fps)))

    # Calculate average performance metrics
    def _pred_depth(row):
        val = row.get('pred_median_depth', np.nan)
        if np.isfinite(val):
            return val
        return row.get('fused_median_depth', np.nan)

    abs_rel_vals = [
        abs(_pred_depth(row) - row['lidar_median_depth']) / (row['lidar_median_depth'] + 1e-6)
        for row in csv_rows
        if np.isfinite(row.get('lidar_median_depth')) and np.isfinite(_pred_depth(row))
    ]
    sq_rel_vals = [
        ((_pred_depth(row) - row['lidar_median_depth'])**2) / (row['lidar_median_depth'] + 1e-6)
        for row in csv_rows
        if np.isfinite(row.get('lidar_median_depth')) and np.isfinite(_pred_depth(row))
    ]

    avg_metrics = {
        # Total pipeline time including fusion
        'inference_time_ms': np.mean([
            row['t_depth_ms'] + row['t_fuse_ms']  # Only depth estimation + fusion time
            for row in timing_rows
        ]),
        
        # Absolute relative error using selected prediction depth as final output
        'abs_rel_error': float(np.mean(abs_rel_vals)) if abs_rel_vals else np.nan,
        
        # Peak memory usage
        'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024),
        
        # Square relative error using selected prediction depth
        'sq_rel_error': float(np.mean(sq_rel_vals)) if sq_rel_vals else np.nan
    }
    
    # Add confidence intervals
    for metric in ['abs_rel_error', 'sq_rel_error']:
        values = abs_rel_vals if metric == 'abs_rel_error' else sq_rel_vals
        if values:
            ci = np.percentile(values, [2.5, 97.5])
            avg_metrics[f'{metric}_ci_low'] = ci[0]
            avg_metrics[f'{metric}_ci_high'] = ci[1]

    # Save baseline comparison to CSV
    comparison_path = os.path.join(OUT_DIR, 'baseline_comparison.csv')
    comparison_rows = []
    
    # Add our system's performance
    comparison_rows.append({
        'method': 'Our System',
        **avg_metrics
    })
    
    # Add baseline performances
    for method, metrics in BASELINE_METRICS.items():
        comparison_rows.append({
            'method': method,
            **metrics
        })
    
    # Save comparison CSV
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    
    # Print performance comparison
    print_performance_comparison(avg_metrics, timing_rows)
    
    print("\n========== DONE ==========")
    print(f"[OUT] CSV:   {csv_path}")
    print(f"[OUT] Comparison CSV: {comparison_path}")
    print(f"[OUT] Video: {video_path}")
    print(f"[OUT] Debug: {DBG_DIR}")

def remove_ground(fused_depth, fx, fy, z_floor=0.0, slope=0.0):
    """Remove ground plane using image-space heuristics.
    
    Args:
        fused_depth: (H,W) depth map
        fx, fy: focal lengths
        z_floor: height of ground plane (default 0.0m)
        slope: road slope in degrees (default 0.0)
    
    Returns:
        Boolean mask where True indicates "not ground"
    """
    H, W = fused_depth.shape
    mask_valid = np.isfinite(fused_depth)
    
    # Simple heuristic: ignore bottom portion with linear depth increase
    row = np.arange(H)[:,None] / H
    road_band = (row > 0.80)  # bottom 20%
    
    # Add slope-based filtering
    if slope != 0.0:
        slope_rad = np.deg2rad(slope)
        expected_depth = (row * H * np.tan(slope_rad))
        road_mask = np.abs(fused_depth - expected_depth) < 0.5  # 0.5m tolerance
        road_band = road_band | road_mask
    
    return mask_valid & (~road_band)

def box_size_meters(x1, y1, x2, y2, z_med, fx, fy):
    """Convert pixel bbox to physical size using depth and intrinsics.
    
    Args:
        x1,y1,x2,y2: Pixel coordinates
        z_med: Median depth in meters
        fx,fy: Focal lengths in pixels
    
    Returns:
        width_m, height_m: Physical dimensions in meters
    """
    w_px = max(1, x2-x1+1)
    h_px = max(1, y2-y1+1)
    w_m = (w_px / fx) * z_med
    h_m = (h_px / fy) * z_med
    return w_m, h_m

def compute_motion_mask(prev_frame, curr_frame, min_flow=1.5):
    """Compute motion mask using optical flow.
    
    Args:
        prev_frame: Previous RGB frame
        curr_frame: Current RGB frame
        min_flow: Minimum flow magnitude threshold
    
    Returns:
        tuple: (motion_mask, flow_magnitudes)
        - motion_mask: Boolean mask where True indicates motion
        - flow_magnitudes: Float array of raw flow magnitudes
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    flow_mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
    return flow_mag > min_flow, flow_mag

def get_ttc_threshold(class_name):
    if class_name is None:
        return T_WARN_DEFAULT
    name = str(class_name).lower()
    return T_WARN_VRU if any(k in name for k in VRU_TOKENS) else T_WARN_VEHICLE

def check_object_size(width_m, height_m):
    """Check if object's physical size is within valid ranges"""
    return (MIN_SIZE_M['width'] <= width_m <= MAX_SIZE_M['width'] and 
            MIN_SIZE_M['height'] <= height_m <= MAX_SIZE_M['height'])

def compute_warning_state(obj_id, ttc, class_name, state_dict):
    if obj_id not in state_dict:
        state_dict[obj_id] = {
            'consecutive_frames': 0,
            'warning_active': False,
            'last_ttc': None,
            'ema_depth': None
        }
    state = state_dict[obj_id]

    t_warn = get_ttc_threshold(class_name)
    if not USE_SANITY_CHECKS:
        warn_raw = np.isfinite(ttc) and (ttc <= t_warn)
        state['warning_active'] = warn_raw
        state['consecutive_frames'] = 1 if warn_raw else 0
        state['last_ttc'] = ttc
        return warn_raw, warn_raw

    t_clear = t_warn + T_HYSTERESIS

    is_vru = any(k in str(class_name).lower() for k in VRU_TOKENS)
    min_frames = 2 if is_vru else 3

    warn_raw = np.isfinite(ttc) and (ttc <= t_warn)
    state['consecutive_frames'] = state['consecutive_frames'] + 1 if warn_raw else 0

    if not state['warning_active'] and state['consecutive_frames'] >= min_frames:
        state['warning_active'] = True

    if state['warning_active'] and np.isfinite(ttc) and (ttc > t_clear):
        state['warning_active'] = False
        state['consecutive_frames'] = 0

    state['last_ttc'] = ttc
    return warn_raw, state['warning_active']

def make_ecw_mask(H, W, ecw_polygon):
    """
    Returns boolean mask [H,W] for ECW polygon loaded from annotation.
    ecw_polygon: np.ndarray of shape (4,2) or more, int32
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [ecw_polygon], 1)
    return mask.astype(bool)

if __name__ == '__main__':
    # Main entry point. No mask creation here; use make_ecw_mask(H, W, ecw_polygon) in your pipeline.
    main()
