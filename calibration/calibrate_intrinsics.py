# calibration/calibrate_intrinsics.py
import glob, os, cv2, numpy as np, yaml, argparse, sys

# --- tame OpenCL cache issues on macOS ---
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass
os.environ["OPENCV_OPENCL_CACHE_DISABLE"] = "1"  # extra belt & suspenders

ap = argparse.ArgumentParser()
ap.add_argument("--imgs_glob", required=True, help='e.g. "calibration/calib_imgs_front/*.png"')
ap.add_argument("--cols", type=int, required=True, help="inner corners across")
ap.add_argument("--rows", type=int, required=True, help="inner corners down")
ap.add_argument("--square_size", type=float, required=True, help="square size in meters")
ap.add_argument("--out_yaml", default="calibration/camera.yaml")
args = ap.parse_args()

pattern_size = (args.cols, args.rows)

# Prepare object points (0..cols-1, 0..rows-1, 0) scaled by square_size
objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
objp *= args.square_size

objpoints, imgpoints = [], []

img_paths = sorted(glob.glob(args.imgs_glob))
if not img_paths:
    print("No images matched:", args.imgs_glob)
    sys.exit(1)

def detect_corners(gray):
    # Light pre-processing helps with outdoor frames
    g = cv2.equalizeHist(gray)
    g = cv2.GaussianBlur(g, (3,3), 0)

    # 1) Robust SB detector first (if available)
    try:
        ret, corners = cv2.findChessboardCornersSB(g, pattern_size)
        if ret and corners is not None:
            return True, corners
    except AttributeError:
        pass  # SB not available in this build

    # 2) Fallback: classic detector + flags + subpixel refine
    ret, corners = cv2.findChessboardCorners(
        g, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    return ret, corners

H = W = None
for p in img_paths:
    img = cv2.imread(p)
    if img is None:
        print("[skip] read fail", os.path.basename(p)); continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = detect_corners(gray)
    if not ret:
        print("[skip] No board in", os.path.basename(p)); continue

    objpoints.append(objp.copy())
    imgpoints.append(corners)
    H, W = gray.shape[:2]

if len(objpoints) < 8:
    print(f"Only {len(objpoints)} detections. Try more views, better lighting, full board visible.")
    sys.exit(2)

print(f"Using {len(objpoints)} images for calibration at size {(H, W)}")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (W, H), None, None)
print(f"RMS reprojection error: {rms:.6f}")
print("K=\n", K)
print("dist=", dist.ravel())

os.makedirs(os.path.dirname(args.out_yaml) or ".", exist_ok=True)
with open(args.out_yaml, "w") as f:
    yaml.safe_dump({
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "size": [int(H), int(W)],
        "pattern_inner_corners": [args.cols, args.rows],
        "square_size_m": float(args.square_size),
        "rms": float(rms),
    }, f)
print(f"Saved intrinsics to {args.out_yaml}")
