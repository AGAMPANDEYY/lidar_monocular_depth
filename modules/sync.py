# modules/sync.py
import os, re, bisect
from typing import List, Tuple, Optional

_DIGITS = re.compile(r'(\d+)')

def _extract_id(name: str) -> Optional[int]:
    """
    Extract the first integer substring from a filename.
    Works for '06000.npz', 'input (Frame 6000).pcd', 'cam_15023.png', etc.
    """
    m = _DIGITS.search(name)
    return int(m.group(1)) if m else None

def _sorted_ids_and_paths(paths: List[str]) -> Tuple[list, list]:
    ids, pths = [], []
    for p in paths:
        i = _extract_id(os.path.basename(p))
        if i is not None:
            ids.append(i); pths.append(p)
    order = sorted(range(len(ids)), key=lambda k: ids[k])
    ids = [ids[k] for k in order]
    pths = [pths[k] for k in order]
    return ids, pths

def auto_detect_starts(
    camera_frame_files: List[str],
    lidar_files: List[str],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer the smallest numeric ID present in each stream (useful as the 'start').
    camera_frame_files: list of camera image filenames (not full paths required)
    lidar_files: list of LiDAR file *paths*
    """
    cam_ids = []
    for f in camera_frame_files:
        i = _extract_id(os.path.basename(f))
        if i is not None:
            cam_ids.append(i)
    lidar_ids = []
    for p in lidar_files:
        i = _extract_id(os.path.basename(p))
        if i is not None:
            lidar_ids.append(i)
    cam_start = min(cam_ids) if cam_ids else None
    lidar_start = min(lidar_ids) if lidar_ids else None
    return cam_start, lidar_start

def find_best_lidar_match(
    camera_frame: int,
    lidar_paths: List[str],
    camera_fps: float = 25.0,
    lidar_fps: float = 10.0,
    camera_start: Optional[int] = None,
    lidar_start: Optional[int] = None,
    max_time_diff_ms: float = 100.0,
) -> Tuple[Optional[str], float, Optional[int]]:
    """
    Map a camera frame ID to the nearest LiDAR frame using FPS ratio.
    Returns (lidar_path, time_diff_ms, lidar_frame_id) or (None, inf, None).

    - If camera_start/lidar_start are None, they are auto-detected from the inputs.
    - Rejects matches beyond max_time_diff_ms.
    """
    if not lidar_paths:
        return None, float('inf'), None

    # Build sorted LiDAR index once per call (fast enough for your batch sizes)
    lidar_ids, lidar_sorted_paths = _sorted_ids_and_paths(lidar_paths)
    if not lidar_ids:
        return None, float('inf'), None

    # Auto-detect starts if not provided
    if camera_start is None or lidar_start is None:
        # We only need lidar_start for the mapping; camera_start comes from the caller ideally,
        # but if not provided we estimate from the camera_frame itself (best-effort).
        # In your main, pass the real camera_start for consistency across frames.
        # Here, fallback to min seen id.
        if lidar_start is None:
            lidar_start = lidar_ids[0]
        if camera_start is None:
            # assume ids are absolute and camera_frame is one of them; fallback to itself
            camera_start = camera_frame

    # Compute expected LiDAR frame (float) via slope
    slope = lidar_fps / camera_fps  # e.g., 0.4 for 10/25
    expected = lidar_start + (camera_frame - camera_start) * slope

    # Find nearest available LiDAR id around the rounded expectation
    # Try 3 neighbors to be safe
    target = int(round(expected))
    pos = bisect.bisect_left(lidar_ids, target)
    candidate_idxs = []
    for k in (pos-1, pos, pos+1):
        if 0 <= k < len(lidar_ids):
            candidate_idxs.append(k)
    if not candidate_idxs:
        return None, float('inf'), None

    # Select candidate with smallest temporal error
    best = (None, float('inf'), None)  # (path, dt_ms, id)
    t_cam = (camera_frame - camera_start) / camera_fps
    for k in candidate_idxs:
        lid = lidar_ids[k]
        t_lid = (lid - lidar_start) / lidar_fps
        dt_ms = abs(t_cam - t_lid) * 1000.0
        if dt_ms < best[1]:
            best = (lidar_sorted_paths[k], dt_ms, lid)

    # Enforce time tolerance
    if best[0] is not None and best[1] <= max_time_diff_ms:
        return best
    return None, best[1], best[2]
