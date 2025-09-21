import os, sys, csv, yaml, math
import numpy as np
import cv2
import argparse

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def read_csv_xy(path):
    xs = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if not row: continue
            vals = [float(v.strip()) for v in row[:2]]
            xs.append(vals)
    return np.array(xs, dtype=np.float64)

def read_csv_xyz(path):
    xs = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if not row: continue
            vals = [float(v.strip()) for v in row[:3]]
            xs.append(vals)
    return np.array(xs, dtype=np.float64)

def project_points(obj_xyz, rvec, tvec, K, dist):
    obj = obj_xyz.reshape(-1,1,3)
    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return img_pts.reshape(-1,2)

def mean_reproj_error(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=1).mean())

def solve_pnp(obj_xyz, img_uv, K, dist, flag=cv2.SOLVEPNP_SQPNP):
    ok, rvec, tvec = cv2.solvePnP(
        obj_xyz.reshape(-1,1,3),
        img_uv.reshape(-1,1,2),
        K, dist, flags=flag
    )
    if not ok:
        return None, None, None
    pred = project_points(obj_xyz, rvec, tvec, K, dist)
    err = mean_reproj_error(pred, img_uv)
    return rvec, tvec, err

def compose_extrinsics(R, t):
    """Return 4x4 [R|t] in camera-from-lidar (Pc = R*Pl + t)."""
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_yaml", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--pts3d_csv", required=True)  # LiDAR frame (raw)
    ap.add_argument("--pts2d_csv", required=True)
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--overlay", default="")
    ap.add_argument("--undistort_for_overlay", action="store_true",
                    help="Undistort the image before drawing (recommended).")
    args = ap.parse_args()

    # Intrinsics
    cam = load_yaml(args.cam_yaml)
    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam["dist"], dtype=np.float64).reshape(-1)

    # Data
    img = cv2.imread(args.image)
    if img is None:
        print("ERROR: cannot read image:", args.image); sys.exit(1)
    H, W = img.shape[:2]

    pts3d_raw = read_csv_xyz(args.pts3d_csv)
    pts2d = read_csv_xy(args.pts2d_csv)
    assert pts3d_raw.shape[0] == pts2d.shape[0], "3D/2D count mismatch."
    assert pts3d_raw.shape[0] >= 6, "Need >=6 correspondences."

    # Try both: RAW vs typical LiDAR->cam axis alignment
    T_align = np.array([[ 0, -1,  0],
                        [ 0,  0, -1],
                        [ 1,  0,  0]], dtype=np.float64)
    pts3d_aligned = (T_align @ pts3d_raw.T).T

    # Use robust PnP
    best = None
    for label, P3 in [("RAW", pts3d_raw), ("ALIGNED", pts3d_aligned)]:
        res = solve_pnp(P3, pts2d, K, dist, flag=cv2.SOLVEPNP_SQPNP)
        if res[0] is None:
            print(f"[{label}] solvePnP failed"); continue
        rvec, tvec, err = res
        R, _ = cv2.Rodrigues(rvec)
        print(f"[{label}] reprojection error = {err:.3f} px")
        if (best is None) or (err < best["err"]):
            best = {"label": label, "R": R, "t": tvec.reshape(3), "err": err}

    if best is None:
        print("ERROR: PnP failed for both raw and aligned."); sys.exit(2)

    # If ALIGNED wins, bake alignment: R_final = R * T_align, t_final = t
    R_final = best["R"].copy()
    t_final = best["t"].copy()
    if best["label"] == "ALIGNED":
        R_final = R_final @ T_align  # compose once; then you can use raw LiDAR thereafter

    # Save extrinsics
    T_cam_lidar = compose_extrinsics(R_final, t_final)
    out = {
        "T_cam_lidar": T_cam_lidar.tolist(),  # Pc = R*Pl + t
        "R": R_final.tolist(),
        "t": t_final.tolist(),
        "reproj_px": float(best["err"]),
        "note": "Pc = R * Pl + t; If label=='ALIGNED', T_align was baked: R := R @ T_align",
        "winner": best["label"]
    }
    save_yaml(args.out_yaml, out)
    print(f"[OK] Saved extrinsics to {args.out_yaml} (winner={best['label']}, err={best['err']:.3f}px)")

    # Optional overlay for visual sanity
    if args.overlay:
        img_draw = img.copy()
        base_img = img_draw
        if args.undistort_for_overlay:
            base_img = cv2.undistort(img_draw, K, dist, None, K)
        pred = project_points(pts3d_raw, cv2.Rodrigues(R_final)[0], t_final, K, dist)
        # draw GT (green) and PRED (red)
        for (u_gt, v_gt), (u_pr, v_pr) in zip(pts2d, pred):
            cv2.circle(base_img, (int(round(u_gt)), int(round(v_gt))), 4, (0,255,0), -1)
            cv2.circle(base_img, (int(round(u_pr)), int(round(v_pr))), 3, (0,0,255), -1)
        cv2.imwrite(args.overlay, base_img)
        print(f"[OK] Wrote overlay: {args.overlay}  (green=GT clicks, red=projected)")

if __name__ == "__main__":
    # Quiet down OpenCL warnings on some mac/conda builds
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    os.environ["OPENCV_OPENCL_CACHE_DISABLE"] = "1"
    main()
