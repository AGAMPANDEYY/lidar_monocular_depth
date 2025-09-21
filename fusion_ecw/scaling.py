import numpy as np
from .configs import BOX_INNER, MIN_LIDAR_PTS, SCALE_CLAMP

def inner_crop(b, H, W, frac=BOX_INNER):
    x1,y1,x2,y2 = b
    w = x2-x1; h = y2-y1
    cx = (x1+x2)/2; cy = (y1+y2)/2
    iw = w*frac; ih = h*frac
    xi1 = int(max(0, cx - iw/2)); yi1 = int(max(0, cy - ih/2))
    xi2 = int(min(W-1, cx + iw/2)); yi2 = int(min(H-1, cy + ih/2))
    return xi1, yi1, xi2, yi2

def robust_median(arr):
    a = arr[np.isfinite(arr)]
    if a.size == 0: return np.nan
    # remove very small/large outliers via IQR
    q1,q3 = np.percentile(a, [25,75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    a = a[(a>=lo) & (a<=hi)]
    if a.size == 0: return np.nan
    return float(np.median(a))

def estimate_scale_per_box(Dmono, Dlidar, Mlidar, bbox):
    H, W = Dmono.shape
    xi1, yi1, xi2, yi2 = inner_crop(bbox, H, W)
    patch_m = Dmono[yi1:yi2+1, xi1:xi2+1]
    patch_l = Dlidar[yi1:yi2+1, xi1:xi2+1]
    patch_mask = Mlidar[yi1:yi2+1, xi1:xi2+1] > 0

    n_lidar = int(patch_mask.sum())
    z_mono = robust_median(patch_m[np.isfinite(patch_m)])
    z_lidar = robust_median(patch_l[patch_mask])

    if n_lidar < MIN_LIDAR_PTS or not np.isfinite(z_mono) or not np.isfinite(z_lidar) or z_mono <= 0:
        return np.nan, z_mono, z_lidar, n_lidar

    s = z_lidar / max(z_mono, 1e-6)
    s = float(np.clip(s, *SCALE_CLAMP))
    return s, z_mono, z_lidar, n_lidar
