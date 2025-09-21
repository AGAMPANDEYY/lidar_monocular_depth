import numpy as np
from .scaling import estimate_scale_per_box
from .configs import EMA_ALPHA, MIN_LIDAR_PTS

class OnlineFuser:
    def __init__(self):
        self.track_scale = {}   # id -> s_ema

    def _ema(self, old, new, alpha=EMA_ALPHA):
        return new if np.isnan(old) else (alpha*new + (1-alpha)*old)

    def fuse_frame(self, Dmono, Dlidar, Mlidar, dets):
        H,W = Dmono.shape
        Dfused = (Dmono.copy()).astype(np.float32)
        per_box_stats = []

        # global scale as fallback using all LiDAR hits over the image
        s_global = self._estimate_global_scale(Dmono, Dlidar, Mlidar)

        for d in dets:
            tid = d["id"]; x1,y1,x2,y2 = map(int, d["bbox"])
            s, zmono, zlidar, n = estimate_scale_per_box(Dmono, Dlidar, Mlidar, (x1,y1,x2,y2))
            if np.isnan(s):
                # fallback: per-track EMA or global
                s = self.track_scale.get(tid, np.nan)
                if np.isnan(s):
                    s = s_global

            # update track EMA if enough LiDAR
            if n >= MIN_LIDAR_PTS and np.isfinite(s):
                self.track_scale[tid] = self._ema(self.track_scale.get(tid, np.nan), s)
                s = self.track_scale[tid]

            # fuse inside box
            box = Dfused[y1:y2+1, x1:x2+1]
            mono_box = Dmono[y1:y2+1, x1:x2+1]
            lidar_box = Dlidar[y1:y2+1, x1:x2+1]
            mask_box = Mlidar[y1:y2+1, x1:x2+1] > 0

            # scale mono
            if np.isfinite(s):
                box[:] = mono_box * s
            else:
                box[:] = mono_box  # last resort

            # overwrite with lidar where available
            box[mask_box] = lidar_box[mask_box]

            Dfused[y1:y2+1, x1:x2+1] = box

            quality = "good" if n >= MIN_LIDAR_PTS else "weak"
            per_box_stats.append({
                "id": tid, "s_scale": float(s), "z_mono_med": float(zmono) if np.isfinite(zmono) else None,
                "z_lidar_med": float(zlidar) if np.isfinite(zlidar) else None,
                "n_lidar_pts": int(n), "quality": quality
            })

        # mask invalid or non-positive
        Dfused[~np.isfinite(Dfused)] = 0.0
        Dfused = np.clip(Dfused, 0.1, 200.0)
        return Dfused, per_box_stats

    def _estimate_global_scale(self, Dmono, Dlidar, Mlidar):
        mask = (Mlidar>0) & np.isfinite(Dmono) & (Dmono>0)
        if mask.sum() < 20: return np.nan
        zmono = np.median(Dmono[mask])
        zlidar = np.median(Dlidar[mask])
        if not np.isfinite(zmono) or zmono<=0: return np.nan
        s = zlidar / zmono
        return float(np.clip(s, 0.3, 3.5))
