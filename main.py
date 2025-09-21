import os
import re
import argparse
from glob import glob

import numpy as np
import pandas as pd
import cv2
import yaml
from PIL import Image

# ── Project modules ─────────────────────────────────────────────────────────────
from modules.detection import load_yolo_model, run_obstacle_detection, CLASSES
# NOTE: this import requires the __init__.py shim described above
from modules.depth import load_depth_backend
from modules.depth_fusion.fusion import fuse_confidence          # using the copy under modules/depth/fusion.py
from modules.metrics import compute_ecw_bubble
from modules.metrics import TTCTracker, robust_box_depth
from modules.visualization import overlay_results

# ── Paths ──────────────────────────────────────────────────────────────────────
FRAME_DIR = 'data/frames'
LIDAR_DIR = 'data/processed_lidar'
OUT_DIR   = 'data/fused_output'
YOLO_WEIGHTS = 'detection/best-e150 (1).pt'

os.makedirs(OUT_DIR, exist_ok=True)
DBG_DIR = os.path.join(OUT_DIR, "debug")
os.makedirs(DBG_DIR, exist_ok=True)

# ── Utilities ──────────────────────────────────────────────────────────────────
def norm_stem(s: str) -> str:
    """Extract digits from filename stem and strip leading zeros."""
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
    parser.add_argument('--depth_backend', type=str, default='zoe',
                    choices=['fastdepth','zoe', 'midas'],
                    help="Monocular depth backend: 'zoe' (HuggingFace) or 'midas' or 'fastdepth'")

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
    for p in glob(os.path.join(LIDAR_DIR, '*.npz')):
        stem = os.path.splitext(os.path.basename(p))[0]
        lidar_index[norm_stem(stem)] = p
    for p in glob(os.path.join(LIDAR_DIR, '*.pcd')):
        k = norm_stem(os.path.splitext(os.path.basename(p))[0])
        lidar_index.setdefault(k, p)

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
        return lidar_index.get(best_key, None)

    # Trackers & constants
    ttc_tracker = TTCTracker()
    fps_for_ttc = args.camera_fps
    ego_speed   = 5.0  # m/s (tune for your rig)

    csv_rows = []

    print("\n========== PROCESSING ==========")
    for idx, fname in enumerate(frame_files, 1):
        frame_id = os.path.splitext(fname)[0]
        camera_frame = int(frame_id)
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

        # Detection
        pred_bboxes = run_obstacle_detection(img_np, yolo_model)
        print(f"  [DET] objects: {len(pred_bboxes)}")

        # Monocular depth → (H,W)
        depth_map = run_depth(img_np)  # returns np.ndarray (H,W) in arbitrary units
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

        # Scale MiDaS to meters using overlap with LiDAR
        eps = 1e-6
        overlap = Mlidar_small & np.isfinite(depth_map) & (depth_map > 0)
        if overlap.sum() >= 50:
            scale = float(np.median(Dlidar_small[overlap] / (depth_map[overlap] + eps)))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        else:
            scale = 1.0
        depth_midas_m = np.clip(depth_map * scale, 0.1, 80.0)
        print(f"  [CAL] overlap={int(overlap.sum())}, mono_scale={scale:.3f}")

        # Fuse depths
        fused_depth, Wlidar, Mfused = fuse_confidence(Dlidar_small, Mlidar_small, depth_midas_m)
        print(f"  [FUSE] mean LiDAR weight: {float(np.nanmean(Wlidar)):.3f}; fused finite px: {int(np.isfinite(fused_depth).sum())}")

        # Debug panels
        dm = depth_midas_m.copy()
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

            mono_patch  = depth_midas_m[yi1:yi2, xi1:xi2]
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
            depth_for_ttc = d_lidar if np.isfinite(d_lidar) else d_fused
            ttc, dzdt = ttc_tracker.update_and_compute(obj_idx, depth_for_ttc, t_now, ego_speed=ego_speed)
            ttc_list.append(ttc)

            ecw, _ = compute_ecw_bubble([x1, y1, x2, y2], fused_depth)
            ecw_bubbles.append(ecw)

            # CSV row
            csv_rows.append({
                "frame": frame_id,
                "class": CLASSES[cls],
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "mono_median_depth": mono_depth_val,
                "lidar_median_depth": lidar_depth_val,
                "ttc": ttc,
                "ecw_bubble": ecw
            })

        print(f"  [OBJ] boxes: {len(bboxes)}")

        # Final overlay
        base_rgb = paint_depth_background(img_np, depth_midas_m, mask=np.isfinite(depth_midas_m),
                                          cmap=cv2.COLORMAP_MAGMA)
        base_rgb = paint_depth_background(base_rgb, Dlidar_small, mask=Mlidar_small,
                                          alpha_fg=0.9, alpha_bg=0.1, cmap=cv2.COLORMAP_TURBO)

        final_img = overlay_results(
            base_rgb, bboxes, classes, confidences,
            lidar_depths, mono_depths, ttc_list, ecw_bubbles
        )

        out_img_path = os.path.join(OUT_DIR, frame_id + '_overlay.png')
        cv2.imwrite(out_img_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
        print(f"  [SAVE] overlay → {out_img_path}")

    # CSV
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(OUT_DIR, 'object_depth_metrics.csv')
    df.to_csv(csv_path, index=False)
    print('\n[CSV] saved:', csv_path)

    # Video
    video_path = os.path.join(OUT_DIR, 'output_visualization.mp4')
    create_video(OUT_DIR, video_path, fps=int(round(args.camera_fps)))

    print("\n========== DONE ==========")
    print(f"[OUT] CSV:   {csv_path}")
    print(f"[OUT] Video: {video_path}")
    print(f"[OUT] Debug: {DBG_DIR}")

if __name__ == '__main__':
    main()
