import numpy as np
import cv2

def estimate_ground_mask(depth, horizon_y, slope_tol=0.08):
    """
    Simple ground estimate via vertical gradient threshold below horizon.
    """
    H, W = depth.shape
    d = depth.copy()
    d[~np.isfinite(d)] = 0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    ground = np.zeros_like(d, dtype=bool)
    horizon_y = int(np.clip(horizon_y, 0, H-1))
    ground[horizon_y:, :] = np.abs(gy[horizon_y:, :]) < slope_tol
    return ground

def ego_corridor_rows(depth, ground_mask, stride=8, y_start_frac=0.75):
    """
    Scan from bottom up; for each row, find gap from center to nearest obstacle
    on both sides. Returns list of (gap_px, depth_m) per row (NaN when invalid).
    """
    H, W = depth.shape
    y_start = int(H * y_start_frac)
    rows = []
    for y in range(H - 1, y_start, -stride):
        drow = depth[y]
        valid = np.isfinite(drow)
        obs = valid & (~ground_mask[y])
        if obs.sum() < 5:
            rows.append((np.nan, np.nan)); continue
        xs = np.where(obs)[0]
        left_gap = xs[xs < W // 2]
        right_gap = xs[xs > W // 2]
        if left_gap.size == 0 or right_gap.size == 0:
            rows.append((np.nan, np.nan)); continue
        gap_px = right_gap.min() - left_gap.max()
        d_c = np.median(drow[valid])
        rows.append((float(gap_px), float(d_c)))
    return rows

def ecw_from_rows(rows, fx, clip_m=(0.2, 6.0)):
    """
    Convert (gap_px, depth) to meters via w ≈ gap_px / fx * depth.
    Returns min_width_m (NaN if none), and list of widths per row.
    """
    widths_m = []
    for gap_px, d in rows:
        if not np.isfinite(gap_px) or not np.isfinite(d) or gap_px <= 0 or d <= 0:
            continue
        w_m = (gap_px / max(fx, 1e-6)) * d
        w_m = float(np.clip(w_m, *clip_m))
        widths_m.append(w_m)
    if not widths_m:
        return np.nan, widths_m
    return float(np.nanmin(widths_m)), widths_m

def early_warning_score(ttc, ecw_m, w_thresh=1.2):
    """
    Combine TTC and ECW into a single [0..1] risk score.
    """
    s_ttc = np.clip((3.0 - ttc) / 3.0, 0, 1) if np.isfinite(ttc) else 0.0
    s_ecw = np.clip((w_thresh - ecw_m) / w_thresh, 0, 1) if np.isfinite(ecw_m) else 0.0
    return float(0.6 * s_ttc + 0.4 * s_ecw)
