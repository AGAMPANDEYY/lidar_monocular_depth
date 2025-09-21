# calibration/calibrate_extrinsics_optimize.py
import argparse, os, yaml, numpy as np, cv2
from scipy.optimize import least_squares

def load_camera(cam_yaml):
    with open(cam_yaml, "r") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam.get("dist", [0,0,0,0,0]), dtype=np.float64).reshape(-1)
    size = cam.get("size", None)  # [H,W]
    return K, dist, size

def read_points_csv(path, ncols, has_header=True):
    arr = np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0)
    if arr.ndim == 1: arr = arr[None, :]
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr.astype(np.float64)

def draw_indexed_overlay(img, proj, img_pts, out_path):
    vis = img.copy()
    for i, (p, q) in enumerate(zip(proj.astype(int), img_pts.astype(int))):
        u,v = p; u2,v2 = q
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(vis, (u,v), 4, (0,0,255), -1)   # projected (red)
            cv2.putText(vis, str(i), (u+4,v-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        if 0 <= u2 < img.shape[1] and 0 <= v2 < img.shape[0]:
            cv2.circle(vis, (u2,v2), 4, (0,255,0), -1) # clicked (green)
            cv2.putText(vis, str(i), (u2+4,v2-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.line(vis, (u,v), (u2,v2), (255,255,0), 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, vis)

def project_points(X3d, rvec, tvec, K, dist, use_undist_norm=False):
    if use_undist_norm:
        # Return normalized (undistorted) projections
        proj, _ = cv2.projectPoints(X3d.reshape(-1,1,3), rvec, tvec, np.eye(3), None)
        return proj.reshape(-1,2)  # these are normalized only if we also normalise u2d
    else:
        proj, _ = cv2.projectPoints(X3d.reshape(-1,1,3), rvec, tvec, K, dist)
        return proj.reshape(-1,2)

def residuals(params, X3d, u2d, K, dist, weights, use_undist_norm):
    rvec = params[:3].reshape(3,1)
    tvec = params[3:].reshape(3,1)
    if use_undist_norm:
        # compare in normalized, undistorted pixel coords
        proj = project_points(X3d, rvec, tvec, None, None, use_undist_norm=True)
        # normalise measured pixels too
        und = cv2.undistortPoints(u2d.reshape(-1,1,2), K, dist, P=np.eye(3)).reshape(-1,2)
        res = proj - und
    else:
        proj = project_points(X3d, rvec, tvec, K, dist, use_undist_norm=False)
        res = proj - u2d
    if weights is not None:
        res = (res * weights[:,None]).ravel()
    return res.ravel()

def subpix_refine_if_possible(img_path, img_pts):
    # optional helper if your 2D points are approximate grid corners; skip for arbitrary clicks
    # Provide your own corner detection if you use a checkerboard/charuco
    return img_pts

def main():
    ap = argparse.ArgumentParser(description="Refine LiDAR->Camera extrinsics with robust LS and RANSAC init")
    ap.add_argument("--cam_yaml", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--pts3d_csv", required=True)
    ap.add_argument("--pts2d_csv", required=True)
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--no_header", action="store_true")
    ap.add_argument("--init_yaml", default=None, help="YAML with {R,t} to start from; else use RANSAC init")
    ap.add_argument("--use_undist_norm", action="store_true", help="optimise in undistorted-normalised pixel space")
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None: raise FileNotFoundError(args.img)
    H, W = img.shape[:2]

    K, dist, size = load_camera(args.cam_yaml)
    if size is not None and (int(size[0]) != H or int(size[1]) != W):
        sw = W / float(size[1]); sh = H / float(size[0])
        K[0,0] *= sw; K[0,2] *= sw; K[1,1] *= sh; K[1,2] *= sh
        print(f"[INFO] Scaled K to {(H,W)}: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")

    X = read_points_csv(args.pts3d_csv, 3, has_header=not args.no_header)
    U = read_points_csv(args.pts2d_csv, 2, has_header=not args.no_header)

    if X.shape[0] != U.shape[0] or X.shape[0] < 6:
        raise ValueError(f"Need ≥6 1:1 pairs; got {X.shape[0]} and {U.shape[0]}")

    # Optional: sub-pixel refine (plug your own if you used a board)
    U = subpix_refine_if_possible(args.img, U)

    # Initial estimate: from file or RANSAC (recommended)
    if args.init_yaml:
        init = yaml.safe_load(open(args.init_yaml))
        R0 = np.array(init["R"], dtype=np.float64)
        t0 = np.array(init["t"], dtype=np.float64).reshape(3,1)
        rvec0, _ = cv2.Rodrigues(R0)
        print("[INFO] Using initial R,t from", args.init_yaml)
    else:
        ok, rvec0, t0, inliers = cv2.solvePnPRansac(
            objectPoints=X.reshape(-1,1,3),
            imagePoints=U.reshape(-1,1,2),
            cameraMatrix=K,
            distCoeffs=dist,
            reprojectionError=4.0,
            confidence=0.999,
            iterationsCount=5000,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok or inliers is None or len(inliers) < 6:
            raise RuntimeError("RANSAC failed — check ordering/quality of pairs.")
        inl = inliers.reshape(-1)
        print(f"[INFO] RANSAC inliers: {len(inl)}/{len(X)}")
        # Use only inliers for refinement
        X, U = X[inl], U[inl]

    # Per-point weights (optional): weight edges/corners higher if you know which are cleaner
    weights = np.ones(len(X), dtype=np.float64)

    # Report initial error (consistent space)
    if args.use_undist_norm:
        proj0, _ = cv2.projectPoints(X.reshape(-1,1,3), rvec0, t0, np.eye(3), None)
        und = cv2.undistortPoints(U.reshape(-1,1,2), K, dist, P=np.eye(3))
        err0 = np.linalg.norm(proj0.reshape(-1,2) - und.reshape(-1,2), axis=1)
    else:
        proj0, _ = cv2.projectPoints(X.reshape(-1,1,3), rvec0, t0, K, dist)
        err0 = np.linalg.norm(proj0.reshape(-1,2) - U, axis=1)
    print("Initial (refine-set) px: mean={:.2f}, median={:.2f}, max={:.2f}".format(
        err0.mean(), np.median(err0), err0.max()))

    # Robust LM refinement
    params0 = np.hstack([rvec0.ravel(), t0.ravel()])
    res = least_squares(
        residuals, params0,
        args=(X, U, K, dist, weights, args.use_undist_norm),
        method="trf", loss="soft_l1", f_scale=2.0, max_nfev=200
    )
    rvec = res.x[:3].reshape(3,1)
    tvec = res.x[3:].reshape(3,1)
    R_opt, _ = cv2.Rodrigues(rvec)

    # Final error
    if args.use_undist_norm:
        proj = cv2.projectPoints(X.reshape(-1,1,3), rvec, tvec, np.eye(3), None)[0].reshape(-1,2)
        und = cv2.undistortPoints(U.reshape(-1,1,2), K, dist, P=np.eye(3)).reshape(-1,2)
        err = np.linalg.norm(proj - und, axis=1)
        # For overlay, convert normalized back to pixels via K:
        proj_px = np.c_[proj[:,0]*K[0,0]+K[0,2], proj[:,1]*K[1,1]+K[1,2]]
    else:
        proj = cv2.projectPoints(X.reshape(-1,1,3), rvec, tvec, K, dist)[0].reshape(-1,2)
        err = np.linalg.norm(proj - U, axis=1)
        proj_px = proj

    print("Final (refine-set) px:  mean={:.2f}, median={:.2f}, max={:.2f}".format(
        err.mean(), np.median(err), err.max()))
    # Print worst offenders to go fix in the CSVs
    worst_idx = np.argsort(-err)[:5]
    print("[DEBUG] Top-5 residuals (idx, px):", [(int(i), float(err[i])) for i in worst_idx])

    # Draw overlay on original image (using pixel coords)
    draw_indexed_overlay(img, proj_px, U, "calib_debug/opt_overlay_indexed.png")
    print("Wrote calib_debug/opt_overlay_indexed.png")

    # Save refined extrinsics
    out = {"R": R_opt.tolist(), "t": tvec.reshape(3).tolist()}
    os.makedirs(os.path.dirname(args.out_yaml) or ".", exist_ok=True)
    yaml.safe_dump(out, open(args.out_yaml, "w"))
    print(f"Saved refined LiDAR->Camera extrinsics to {args.out_yaml}")

if __name__ == "__main__":
    main()
