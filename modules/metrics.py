# modules/metrics.py
import numpy as np
from dataclasses import dataclass

def robust_box_depth(depth_m, bbox, mask=None, inner=0.4, reducer="median", p_lo=20, p_hi=80):
    """
    Robust depth summary inside bbox.
    - depth_m: (H,W) metric depth (e.g., fused or LiDAR-preferred)
    - mask: optional boolean valid mask
    - inner: use centered inner crop to avoid edges (0..1)
    """
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= x1 or y2 <= y1:
        return np.nan
    cx, cy = (x1+x2)//2, (y1+y2)//2
    dx, dy = int((x2-x1)*inner/2), int((y2-y1)*inner/2)
    xi1, yi1 = max(0, cx-dx), max(0, cy-dy)
    xi2, yi2 = min(depth_m.shape[1]-1, cx+dx), min(depth_m.shape[0]-1, cy+dy)

    patch = depth_m[yi1:yi2, xi1:xi2]
    if mask is not None:
        m = mask[yi1:yi2, xi1:xi2]
        vals = patch[m]
    else:
        vals = patch[np.isfinite(patch) & (patch > 0)]

    if vals.size < 10:
        return np.nan

    if reducer == "median":
        return float(np.median(vals))
    elif reducer == "trimmed":
        lo, hi = np.percentile(vals, [p_lo, p_hi])
        trimmed = vals[(vals >= lo) & (vals <= hi)]
        return float(np.median(trimmed)) if trimmed.size else float(np.median(vals))
    else:
        return float(np.median(vals))

@dataclass
class TTCParams:
    min_speed: float = 0.2     # m/s, floor to avoid dividing by ~0
    max_ttc:   float = 20.0    # s, clamp for display
    ema:       float = 0.7     # smoothing on dz/dt (0..1), higher = smoother

class TTCTracker:
    """
    Keeps one-step history per object id to estimate TTC from depth rate.
    You assign ids from a tracker (e.g., SORT/ByteTrack) or simple IoU matching.
    """
    def __init__(self, params=TTCParams()):
        self.params = params
        self.prev_depth = {}   # id -> Z_prev
        self.prev_time  = {}   # id -> t_prev
        self.prev_rate  = {}   # id -> dzdt_prev (EMA)

    def update_and_compute(self, obj_id, depth_now, t_now, ego_speed=None):
        """
        Returns TTC (seconds) and the dz/dt used (negative when approaching).
        If we have no history yet, falls back to ego_speed if provided.
        """
        p = self.params
        if not np.isfinite(depth_now):
            return np.inf, 0.0

        if obj_id in self.prev_depth and obj_id in self.prev_time:
            Z_prev = self.prev_depth[obj_id]
            t_prev = self.prev_time[obj_id]
            dt = max(1e-3, float(t_now - t_prev))
            dzdt = (depth_now - Z_prev) / dt  # >0 receding, <0 approaching

            # EMA smoothing
            if obj_id in self.prev_rate:
                dzdt = p.ema * self.prev_rate[obj_id] + (1 - p.ema) * dzdt
            self.prev_rate[obj_id] = dzdt

            approach_rate = max(p.min_speed, -dzdt) if dzdt < 0 else None
            if approach_rate is None:
                ttc = np.inf
            else:
                ttc = np.clip(depth_now / approach_rate, 0.0, p.max_ttc)
        else:
            # no history: use ego speed if available (stationary world assumption)
            if ego_speed is None or ego_speed <= 0:
                ttc, dzdt = np.inf, 0.0
            else:
                approach_rate = max(p.min_speed, float(ego_speed))
                ttc, dzdt = np.clip(depth_now / approach_rate, 0.0, p.max_ttc), -approach_rate

        # update history
        self.prev_depth[obj_id] = float(depth_now)
        self.prev_time[obj_id]  = float(t_now)
        return ttc, float(dzdt)

def compute_ecw_bubble(bbox, depth_map_m, threshold=10.0):
    """
    True if robust depth inside bbox is within threshold (meters).
    Uses a trimmed-median for stability.
    """
    d = robust_box_depth(depth_map_m, bbox, mask=np.isfinite(depth_map_m),
                         inner=0.4, reducer="trimmed")
    return (d < threshold), d
