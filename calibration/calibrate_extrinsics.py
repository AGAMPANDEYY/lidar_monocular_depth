# calibrate_extrinsics.py
import argparse, yaml, numpy as np, cv2, os

def load_camera(cam_yaml):
    with open(cam_yaml, "r") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64)
    dist = np.array(cam.get("dist", [0,0,0,0,0]), dtype=np.float64).reshape(-1)
    size = cam.get("size", None)  # [H, W]
    return K, dist, size

def read_points_csv(path, cols):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)  # Skip header row
    if arr.ndim == 1:
        arr = arr[None, :]
    assert arr.shape[1] == cols, f"{path}: expected {cols} columns"
    return arr.astype(np.float64)

def solve_extrinsics(pts3d_lidar, pts2d_img, K, dist):
    assert pts3d_lidar.shape[0] == pts2d_img.shape[0] and pts3d_lidar.shape[0] >= 6, \
        "Need ≥6 matched 3D↔2D points"
    obj = pts3d_lidar.reshape(-1,1,3).astype(np.float64)
    img = pts2d_img.reshape(-1,1,2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("solvePnP failed. Check correspondences and intrinsics.")
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)
    return R, t

def write_extrinsics_yaml(out_yaml, R, t):
    data = {
        "R": R.tolist(),
        "t": t.reshape(3).tolist()
    }
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f)
    print(f"Saved LiDAR->Camera extrinsics to {out_yaml}")

def main():
    ap = argparse.ArgumentParser(description="Solve LiDAR->Camera extrinsics from 3D↔2D correspondences")
    ap.add_argument("--cam_yaml", required=True, help="camera.yaml with K, dist, size")
    ap.add_argument("--img", required=True, help="image used for 2D points (for sanity)")
    ap.add_argument("--pts3d_csv", required=True, help="Nx3 CSV of LiDAR points (meters)")
    ap.add_argument("--pts2d_csv", required=True, help="Nx2 CSV of image pixels (u,v)")
    ap.add_argument("--out_yaml", required=True, help="output extrinsics YAML (R,t) LiDAR->Camera")
    args = ap.parse_args()

    # Read image first to get dimensions
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(args.img)
    H, W = img.shape[:2]

    # Then load camera parameters
    K, dist, size = load_camera(args.cam_yaml)

    # Check if we need to scale the camera matrix
    if size is not None and (int(size[0]) != H or int(size[1]) != W):
        scale_w = W / float(size[1])
        scale_h = H / float(size[0])
        # Usually same scale both directions after crop, but compute both:
        K[0,0] *= scale_w  # fx
        K[0,2] *= scale_w  # cx
        K[1,1] *= scale_h  # fy
        K[1,2] *= scale_h  # cy
        print(f"[INFO] Scaled K to {(H,W)}: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    if size is not None and (int(size[0]) != H or int(size[1]) != W):
        print(f"[WARN] camera.yaml size={size} but image is {(H,W)}. "
              f"Use the same size everywhere or scale K accordingly.")

    pts3d = read_points_csv(args.pts3d_csv, 3)
    pts2d = read_points_csv(args.pts2d_csv, 2)

    R, t = solve_extrinsics(pts3d, pts2d, K, dist)
    print("R=\n", R, "\nt=\n", t.reshape(-1))

    # quick overlay sanity check
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts3d, rvec, t.reshape(3), K, dist)
    err = np.linalg.norm(proj.reshape(-1,2)-pts2d, axis=1)
    print("Reprojection error px: mean={:.2f}, max={:.2f}".format(err.mean(), err.max()))
    vis = img.copy()
    for p in proj.reshape(-1,2):
        u,v = map(int, p)
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u,v), 5, (0,0,255), -1)
    os.makedirs("calib_debug", exist_ok=True)
    cv2.imwrite("calib_debug/pnp_projected_points.png", vis)
    print("Wrote calib_debug/pnp_projected_points.png")

    write_extrinsics_yaml(args.out_yaml, R, t)

if __name__ == "__main__":
    main()
