# tools/accumulate_pcd.py
import argparse, glob, os, re
import numpy as np
import open3d as o3d

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

ap = argparse.ArgumentParser(description="Accumulate multiple PCDs into one point cloud.")
ap.add_argument(
    "--glob", nargs="+", required=True,
    help='One or more glob patterns. Quote patterns with spaces/() e.g. "data/lidar/input (Frame 27[3-5][0-9]).pcd"'
)
ap.add_argument("--out", required=True, help="Output PCD path (e.g., calibration/accum.pcd)")
ap.add_argument("--voxel", type=float, default=0.03, help="Voxel size (meters); 0 disables downsampling")
args = ap.parse_args()

# Collect files from all patterns
all_files = []
for pat in args.glob:
    matches = glob.glob(pat)
    if not matches:
        print(f"[WARN] Pattern matched 0 files: {pat}")
    all_files.extend(matches)

# De-dup + sort naturally
files = sorted(set(all_files), key=natural_key)
if not files:
    raise SystemExit("[ERROR] No PCDs matched any supplied patterns.")

print(f"[INFO] Found {len(files)} files to merge.")
pcd_all = o3d.geometry.PointCloud()

merged = 0
for i, f in enumerate(files, 1):
    p = o3d.io.read_point_cloud(f)
    n = np.asarray(p.points).shape[0]
    if n == 0:
        print(f"[SKIP] Empty cloud: {os.path.basename(f)}"); continue
    pcd_all += p
    merged += 1
    if i % 10 == 0 or i == len(files):
        print(f"[INFO] Merged {i}/{len(files)}  (last={os.path.basename(f)}, pts={n})")

if args.voxel and args.voxel > 0:
    before = np.asarray(pcd_all.points).shape[0]
    pcd_all = pcd_all.voxel_down_sample(args.voxel)
    after = np.asarray(pcd_all.points).shape[0]
    print(f"[INFO] Downsampled with voxel={args.voxel:.3f} m: {before} → {after} pts")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
ok = o3d.io.write_point_cloud(args.out, pcd_all)
if not ok:
    raise SystemExit(f"[ERROR] Failed to write {args.out}")
print(f"[OK] Wrote {args.out} with {np.asarray(pcd_all.points).shape[0]} pts")
