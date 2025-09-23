import numpy as np
import cv2

# LiDAR fusion parameters based on recommended settings
LIDAR_MAX_RANGE = 100.0  # absolute maximum range in meters
LIDAR_MIN_RANGE = 0.5    # minimum valid range in meters
LIDAR_HIGH_CONF_RANGE = 50.0  # range until which confidence stays high
LIDAR_MIN_CONF = 0.3     # minimum confidence at max range
MONO_MID_RANGE = 30.0    # range where mono confidence drops to 0.5

def _density_confidence(mask, k=5):
    """Compute confidence based on local point density"""
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), k)

def _range_confidence(depth, eps=1e-3):
    """
    Enhanced range-based confidence following recommended parameters:
    - Full confidence (1.0) from min_range to high_conf_range (0.5m to 50m)
    - Gradual decay from high_conf_range to max_range (50m to 100m)
    - Minimum confidence of 0.3 at max range
    """
    d = depth.copy()
    w = np.ones_like(d)
    
    # Zero confidence outside valid range
    w[d < LIDAR_MIN_RANGE] = 0.0
    w[d > LIDAR_MAX_RANGE] = 0.0
    
    # High confidence region (0.5m to 50m)
    valid = (d >= LIDAR_MIN_RANGE) & (d <= LIDAR_MAX_RANGE)
    if valid.any():
        # Keep full confidence up to LIDAR_HIGH_CONF_RANGE
        far_zone = d > LIDAR_HIGH_CONF_RANGE
        if far_zone.any():
            # Gradual decay from high_conf_range to max_range
            decay_range = LIDAR_MAX_RANGE - LIDAR_HIGH_CONF_RANGE
            decay_factor = (d[far_zone] - LIDAR_HIGH_CONF_RANGE) / decay_range
            # Smooth decay from 1.0 to MIN_CONF
            w[far_zone] = 1.0 - ((1.0 - LIDAR_MIN_CONF) * decay_factor)
    
    w[~np.isfinite(d)] = 0.0
    return w

def _mono_confidence(depth):
    """
    Compute monocular confidence following recommended parameters:
    - Start near 1.0 at close range
    - Reduce quadratically to 0.5 at 30m
    - Taper toward 0.0 approaching max LiDAR range
    """
    d = depth.copy()
    # Quadratic decay to 0.5 at MONO_MID_RANGE (30m)
    w = 1.0 / (1.0 + (d/MONO_MID_RANGE)**2)
    
    # Additional taper toward max LiDAR range
    far_zone = d > MONO_MID_RANGE
    if far_zone.any():
        remain_range = LIDAR_MAX_RANGE - MONO_MID_RANGE
        taper = 1.0 - np.clip((d[far_zone] - MONO_MID_RANGE) / remain_range, 0, 1)
        w[far_zone] *= taper
    
    w[~np.isfinite(d)] = 0.0
    return w

def fuse_confidence(Dlidar, Mlidar, Dmono):
    """
    Enhanced confidence-aware fusion that better respects sensor characteristics.
    - LiDAR maintains high confidence through rated range
    - Mono confidence decreases with depth
    - Density still impacts overall weight
    
    Inputs: depths in meters (H,W), mask bool (H,W)
    Returns: fused_depth (H,W), lidar_weight (H,W), mask_fused (H,W)
    """
    assert Dlidar.shape == Dmono.shape == Mlidar.shape, "Shape mismatch"
    H, W = Dmono.shape

    # LiDAR confidence combines density and range
    Wl_range = _range_confidence(Dlidar)
    Wl_density = _density_confidence(Mlidar)
    Wlidar = Wl_density * Wl_range
    
    # Mono confidence
    Wmono = _mono_confidence(Dmono)
    
    # Combine confidences
    W = np.zeros_like(Dlidar, dtype=np.float32)
    
    # Where we have both, use ratio of confidences
    both_valid = Mlidar & np.isfinite(Dmono)
    if both_valid.any():
        W[both_valid] = Wlidar[both_valid] / (Wlidar[both_valid] + Wmono[both_valid] + 1e-6)
    
    # Where we only have LiDAR
    lidar_only = Mlidar & ~np.isfinite(Dmono)
    W[lidar_only] = Wlidar[lidar_only]
    
    # Normalize and clip weights
    if W.max() > 0:
        W = W / W.max()
    W = np.clip(W, 0, 1)

    Dm = Dmono.copy().astype(np.float32)
    Dl = Dlidar.copy().astype(np.float32)

    Dm[~np.isfinite(Dm)] = 0
    Dl[~np.isfinite(Dl)] = 0

    fused = W * Dl + (1 - W) * Dm
    mask_fused = np.isfinite(Dm) | Mlidar
    fused[~mask_fused] = np.nan
    return fused, W, mask_fused
