import os
import re
import argparse
import time
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
T_HYSTERESIS = 0.5  # additional seconds before clearing warning
MIN_VALID_PIXELS = 15  # minimum valid pixels in inner patch
MIN_PERSISTENCE_FRAMES = 3  # frames of continuous warning before triggering
MIN_SIZE_M = {'width': 0.2, 'height': 0.2}  # minimum object size in meters
MAX_SIZE_M = {'width': 2.5, 'height': 3.0}  # maximum object size in meters

# ── Paths ──────────────────────────────────────────────────────────────────────
FRAME_DIR = 'data/frames'
LIDAR_DIR = 'data/lidar'  # Changed to match where LiDAR files actually are
OUT_DIR   = 'data/fused_output'
YOLO_WEIGHTS = 'detection/best-e150 (1).pt'

os.makedirs(OUT_DIR, exist_ok=True)
DBG_DIR = os.path.join(OUT_DIR, "debug")
os.makedirs(DBG_DIR, exist_ok=True)

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
def main():
    parser = argparse.ArgumentParser(description="Camera–LiDAR fusion pipeline (debug-friendly)")
    parser.add_argument('--camera_start', type=int, default=15000, help='Start camera frame number')
    parser.add_argument('--camera_end',   type=int, default=16500, help='End camera frame number')
    parser.add_argument('--lidar_start',  type=int, default=6000,  help='Start LiDAR frame number')
    parser.add_argument('--lidar_end',    type=int, default=6600,  help='End LiDAR frame number')
    parser.add_argument('--camera_fps',   type=float, default=25.0)
    parser.add_argument('--lidar_fps',    type=float, default=10.0)
    parser.add_argument('--max_frames',   type=int,   default=10,  help='Process at most N frames')
    parser.add_argument('--depth_backend', type=str, default='midas',
                    choices=['fastdepth','zoe', 'midas'],
                    help="Monocular depth backend: 'zoe' (HuggingFace) or 'midas' or 'fastdepth'")
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

    args = parser.parse_args()

    print("\n========== PIPELINE BOOT ==========")
    print(f"[CFG] Camera frames: {args.camera_start} .. {args.camera_end} @ {args.camera_fps:.1f} fps")
    print(f"[CFG] LiDAR  frames: {args.lidar_start} .. {args.lidar_end} @ {args.lidar_fps:.1f} fps")
    print(f"[CFG] Max frames to process: {args.max_frames}")

    # Load fx from camera.yaml (for ECW etc.)
    def _load_fx_from_yaml(path="calibration/camera.yaml"):
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
    frame_files = sorted([
        f for f in os.listdir(FRONT_VIEW_DIR)
        if f.endswith('.png') and args.camera_start <= int(os.path.splitext(f)[0]) <= args.camera_end
    ], key=lambda x: int(os.path.splitext(x)[0]))
    total_candidates = len(frame_files)
    frame_files = frame_files[:args.max_frames]
    print(f"[INFO] Found {total_candidates} frames in range; will process {len(frame_files)}")

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

    print("\n========== PROCESSING ==========")
    for idx, fname in enumerate(frame_files, 1):
        frame_id = os.path.splitext(fname)[0]
        camera_frame = int(frame_id)
        
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
        pred_bboxes = run_obstacle_detection(img_np, yolo_model)
        t1 = time.perf_counter()
        print(f"  [DET] objects: {len(pred_bboxes)}")

        # Monocular depth → (H,W)
        depth_map = run_depth(img_np)  # returns np.ndarray (H,W) in arbitrary units
        t2 = time.perf_counter()
        if depth_map.shape != (H, W):
            print(f"  [DEPTH] resize {depth_name} {depth_map.shape} → {(H, W)}")
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

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
                f"--cam_yaml calibration/camera.yaml "
                f"--ext_yaml calibration/extrinsics_lidar_to_cam.yaml "
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

        valid = int(Mlidar_small.sum())
        print(f"  [LIDAR] valid px: {valid}")
        if valid == 0:
            print("  [LIDAR][WARN] 0 valid px — calibration/sync suspect")

        # --- Convert MiDaS relative to metric meters using LiDAR overlap ---
        eps = 1e-6
        if args.depth_backend == 'midas':
            overlap = Mlidar_small & np.isfinite(depth_map) & np.isfinite(Dlidar_small) & (Dlidar_small > 0)
            if overlap.sum() >= 200:
                r = depth_map[overlap].astype(np.float32)
                z = Dlidar_small[overlap].astype(np.float32)
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
                X = np.stack([x, np.ones_like(x)], axis=1)
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                A, B = float(sol[0]), float(sol[1])
                r_full = depth_map.astype(np.float32)
                x_full = (r_full - r_lo) / (r_hi - r_lo + eps)
                x_full = np.clip(x_full, 0.0, 1.0)
                depth_mono_m = 1.0 / np.clip(A * x_full + B, 1e-4, 1e4)
                depth_mono_m = np.clip(depth_mono_m, 0.1, 120.0)
                print(f"  [CAL] fit AB using {x.size} pts: A={A:.6f}, B={B:.6f} -> min≈{1.0/(A+B+eps):.2f} m, max≈{1.0/(B+eps):.2f} m (from overlap)")
            else:
                print("  [CAL][WARN] insufficient overlap; using median-ratio scale fallback")
                overlap2 = Mlidar_small & np.isfinite(depth_map) & (depth_map > 0)
                if overlap2.sum() >= 50:
                    scale = float(np.median(Dlidar_small[overlap2] / (depth_map[overlap2] + eps)))
                else:
                    scale = 1.0
                depth_mono_m = np.clip(depth_map * scale, 0.1, 120.0)
        else:
            overlap = Mlidar_small & np.isfinite(depth_map) & (depth_map > 0)
            if overlap.sum() >= 50:
                scale = float(np.median(Dlidar_small[overlap] / (depth_map[overlap] + eps)))
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
            else:
                scale = 1.0
            depth_mono_m = np.clip(depth_map * scale, 0.1, 80.0)
            print(f"  [CAL] overlap={int(overlap.sum())}, mono_scale={scale:.3f}")

        # Fuse depths
        fused_depth, Wlidar, Mfused = fuse_confidence(Dlidar_small, Mlidar_small, depth_mono_m)
        print(f"  [FUSE] mean LiDAR weight: {float(np.nanmean(Wlidar)):.3f}; fused finite px: {int(np.isfinite(fused_depth).sum())}")

        # Save dense depth maps for evaluation
        np.save(os.path.join(DBG_DIR, f"{frame_id}_fused_depth.npy"), fused_depth)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_mono_depth.npy"), depth_mono_m)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_depth.npy"), Dlidar_small)
        np.save(os.path.join(DBG_DIR, f"{frame_id}_lidar_mask.npy"), Mlidar_small.astype(np.uint8))
        print(f"  [SAVE] depth maps → {frame_id}_*_depth.npy")
        t3 = time.perf_counter()  # After fusion and depth save

        # ---------- Missed obstacle detection inside ECW (depth blobs not covered by YOLO) ----------
        # 3.1 ECW mask (trapezoid ahead of ego)
        ECW_mask, ECW_poly = make_ecw_mask(
            H, W, args.ecw_top, args.ecw_bot, args.ecw_top_w, args.ecw_bot_w
        )

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
        map_for_ecw = fused_depth if args.ecw_use_fused else depth_mono_m
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
        
        # Combine all cues
        depth_cue = ECW_mask & valid_depth & near & far & (~det_mask) & not_ground
        
        # Final candidate mask combines depth and motion cues
        cand = depth_cue | motion_cue

        # Optional smoothing / closing to connect sparse LiDAR projections
        if cand.any():
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, k2, iterations=1).astype(bool)

        missed_boxes = []
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

        for obj_idx, box in enumerate(pred_bboxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cls = int(box[4]); conf = float(box[5])
            x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
            y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
            if x2 <= x1 or y2 <= y1:
                continue

            # inner 40% patch (avoid edges)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            dx = max(1, (x2-x1)//5); dy = max(1, (y2-y1)//5)
            xi1, yi1 = cx - 2*dx, cy - 2*dy
            xi2, yi2 = cx + 2*dx, cy + 2*dy
            xi1 = max(0, xi1); yi1 = max(0, yi1); xi2 = min(W-1, xi2); yi2 = min(H-1, yi2)

            mono_patch  = depth_mono_m[yi1:yi2, xi1:xi2]
            lidar_patch = Dlidar_small[yi1:yi2, xi1:xi2]
            mask_patch  = Mlidar_small[yi1:yi2, xi1:xi2]

            mono_vals  = mono_patch[np.isfinite(mono_patch)]
            lidar_vals = lidar_patch[mask_patch]
            mono_depth_val  = float(np.median(mono_vals))  if mono_vals.size  > 0 else np.nan
            lidar_depth_val = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan

            bboxes.append([x1, y1, x2, y2])
            classes.append(CLASSES[cls])
            confidences.append(conf)
            mono_depths.append(mono_depth_val)
            lidar_depths.append(lidar_depth_val)

            # TTC (time base = camera fps)
            t_now = idx / fps_for_ttc
            d_lidar = robust_box_depth(Dlidar_small, [x1,y1,x2,y2], mask=Mlidar_small)
            d_fused = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            d_use = d_lidar if np.isfinite(d_lidar) else d_fused
            if np.isfinite(d_use):
                ttc, dzdt = ttc_tracker.update_and_compute(obj_idx, d_use, t_now, ego_speed=ego_speed)
            else:
                ttc = np.nan
                dzdt = np.nan
            ttc_list.append(ttc)

            ecw, _ = compute_ecw_bubble([x1, y1, x2, y2], fused_depth)
            ecw_bubbles.append(ecw)

            # CSV row with fused depth
            d_fused_box = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            # Physical size check
            width_m, height_m = box_size_meters(x1, y1, x2, y2, d_use, fx_cam, fx_cam)
            meets_size = check_object_size(width_m, height_m)
            
            # Get depth confidence
            patch_mask = np.isfinite(fused_depth[yi1:yi2, xi1:xi2])
            depth_valid_px = int(patch_mask.sum())
            
            # Compute warning states
            warn_raw, warn_stable = compute_warning_state(
                f"det_{obj_idx}", ttc, CLASSES[cls], 
                ttc_tracker.warning_states
            )
            
            csv_rows.append({
                "frame": frame_id,
                "obj_id": f"det_{obj_idx}",  # Unique object ID
                "class": CLASSES[cls],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "mono_median_depth": mono_depth_val,
                "lidar_median_depth": lidar_depth_val, 
                "fused_median_depth": float(d_fused_box) if np.isfinite(d_fused_box) else np.nan,
                "ttc": ttc,
                "in_ecw": bool(ecw),
                "depth_valid_px": depth_valid_px,
                "meets_size": meets_size,
                "width_m": width_m,
                "height_m": height_m,
                "warn_raw": warn_raw,
                "warn_stable": warn_stable,
                "source": "det"
            })

            # Print per-object depth for debug
            print(f"    [OBJ][{obj_idx}] class={CLASSES[cls]}, conf={conf:.2f}, mono_depth={mono_depth_val:.2f}m, lidar_depth={lidar_depth_val:.2f}m")

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
            lidar_patch = Dlidar_small[y1:y2, x1:x2]
            mask_patch  = Mlidar_small[y1:y2, x1:x2]

            mono_vals  = mono_patch[np.isfinite(mono_patch)]
            lidar_vals = lidar_patch[mask_patch]
            mono_depth_val  = float(np.median(mono_vals))  if mono_vals.size  > 0 else np.nan
            lidar_depth_val = float(np.median(lidar_vals)) if lidar_vals.size > 0 else np.nan

            # TTC using fused depth (preferred) or LiDAR-in-box if available
            t_now = idx / fps_for_ttc
            d_fused = robust_box_depth(fused_depth, [x1,y1,x2,y2], mask=np.isfinite(fused_depth))
            d_use = d_fused if np.isfinite(d_fused) else z_med
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
            in_ecw, _ = compute_ecw_bubble([x1, y1, x2, y2], fused_depth)
            ecw_bubbles.append(in_ecw)
            
            # Get depth confidence
            patch_mask = np.isfinite(fused_depth[y1:y2, x1:x2])
            depth_valid_px = int(patch_mask.sum())
            
            # Physical size check
            width_m, height_m = box_size_meters(x1, y1, x2, y2, d_use, fx_cam, fx_cam)
            meets_size = check_object_size(width_m, height_m)
            
            # Compute warning states with persistence and hysteresis
            warn_raw, warn_stable = compute_warning_state(
                f"miss_{base_obj_count + j}", ttc, "Missed", 
                ttc_tracker.warning_states
            )

            # For missed blobs, append CSV with fused depth
            csv_rows.append({
                "frame": frame_id,
                "obj_id": f"miss_{base_obj_count + j}",  # Unique object ID
                "class": "Missed",
                "confidence": 0.01,
                "bbox": (x1, y1, x2, y2),
                "mono_median_depth": mono_depth_val,
                "lidar_median_depth": lidar_depth_val,
                "fused_median_depth": float(d_use) if np.isfinite(d_use) else np.nan,
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

    # Video
    video_path = os.path.join(OUT_DIR, 'output_visualization.mp4')
    create_video(OUT_DIR, video_path, fps=int(round(args.camera_fps)))

    print("\n========== DONE ==========")
    print(f"[OUT] CSV:   {csv_path}")
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
    """Get class-specific TTC warning threshold"""
    if class_name is None:
        return T_WARN_DEFAULT
    name = str(class_name).lower()
    if any(k in name for k in ['person', 'pedestrian', 'bicycle', 'bike']):
        return T_WARN_VRU
    return T_WARN_VEHICLE

def check_object_size(width_m, height_m):
    """Check if object's physical size is within valid ranges"""
    return (MIN_SIZE_M['width'] <= width_m <= MAX_SIZE_M['width'] and 
            MIN_SIZE_M['height'] <= height_m <= MAX_SIZE_M['height'])

def compute_warning_state(obj_id, ttc, class_name, state_dict):
    """Compute warning state with persistence and hysteresis
    
    Args:
        obj_id: Unique object identifier
        ttc: Current TTC value
        class_name: Object class for threshold selection
        state_dict: Persistent state dictionary
        
    Returns:
        tuple: (warn_raw, warn_stable)
        - warn_raw: True if TTC <= T_warn this frame
        - warn_stable: True if warning criteria met with persistence/hysteresis
    """
    if obj_id not in state_dict:
        state_dict[obj_id] = {
            'consecutive_frames': 0,
            'warning_active': False,
            'last_ttc': None
        }
    
    state = state_dict[obj_id]
    t_warn = get_ttc_threshold(class_name)
    t_clear = t_warn + T_HYSTERESIS
    
    # Raw warning this frame
    warn_raw = np.isfinite(ttc) and ttc <= t_warn
    
    if warn_raw:
        state['consecutive_frames'] += 1
    else:
        state['consecutive_frames'] = 0
    
    # Warning becomes active after persistence threshold
    if state['consecutive_frames'] >= MIN_PERSISTENCE_FRAMES:
        state['warning_active'] = True
    
    # Clear warning only if TTC rises above t_clear
    if state['warning_active'] and np.isfinite(ttc) and ttc > t_clear:
        state['warning_active'] = False
    
    state['last_ttc'] = ttc
    return warn_raw, state['warning_active']

def make_ecw_mask(H, W, top_y, bot_y, top_w, bot_w):
    """
    Trapezoid in front of ego: widths are fractions of W; y are fractions of H.
    Returns boolean mask [H,W].
    """
    import numpy as np, cv2
    y1 = int(top_y * H); y2 = int(bot_y * H)
    half_top = int(0.5 * top_w * W)
    half_bot = int(0.5 * bot_w * W)
    cx = W // 2

    poly = np.array([
        [cx - half_top, y1],
        [cx + half_top, y1],
        [cx + half_bot, y2],
        [cx - half_bot, y2],
    ], dtype=np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool), poly

if __name__ == '__main__':
    main()
