import numpy as np
import cv2

def _density_confidence(mask, k=5):
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), k)

def _range_confidence(depth, eps=1e-3):
    d = depth.copy()
    d[~np.isfinite(d)] = np.nan
    w = 1.0 / (np.abs(d) + eps)
    w[~np.isfinite(w)] = 0.0
    return w

def fuse_confidence(Dlidar, Mlidar, Dmono):
    """
    Confidence-aware fusion (LiDAR dominates where dense/near).
    Inputs: depths in meters (H,W), mask bool (H,W).
    Returns: fused_depth (H,W), lidar_weight (H,W), mask_fused (H,W).
    """
    assert Dlidar.shape == Dmono.shape == Mlidar.shape, "Shape mismatch"
    H, W = Dmono.shape

    # weights
    Wd = _density_confidence(Mlidar) * _range_confidence(Dlidar)
    if Wd.max() > 0:
        Wd = Wd / Wd.max()
    W = np.clip(Wd, 0, 1).astype(np.float32)

    Dm = Dmono.copy().astype(np.float32)
    Dl = Dlidar.copy().astype(np.float32)

    Dm[~np.isfinite(Dm)] = 0
    Dl[~np.isfinite(Dl)] = 0

    fused = W * Dl + (1 - W) * Dm
    mask_fused = np.isfinite(Dm) | Mlidar
    fused[~mask_fused] = np.nan
    return fused, W, mask_fused
