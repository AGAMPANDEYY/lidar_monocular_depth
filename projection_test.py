import argparse, os, yaml
import numpy as np
import cv2
import open3d as o3d

def load_cam(cam_yaml):
    with open(cam_yaml, "r") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam.get("dist", [0,0,0,0,0]), dtype=np.float64)
    size = cam.get("size", None)
    if size is not None:
        size = [int(size[0]), int(size[1])]
    return K, dist, size

def load_ext(ext_yaml):
    with open(ext_yaml, "r") as f:
        ext = yaml.safe_load(f)
    R = np.array(ext["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(ext["t"], dtype=np.float64).reshape(3, 1)
    return R, t

def filter_fov(points, fov_degrees=120):
    """
    Filter points to keep only those within the specified horizontal field of view.
    Assumes standard Velodyne coordinates: X=forward, Y=left, Z=up
    
    Args:
        points: Nx3 array of points (X, Y, Z) in standard Velodyne coordinates
        fov_degrees: Total horizontal field of view in degrees
    
    Returns:
        mask: Boolean array indicating which points are within FOV
    """
    # Calculate angles in the X-Y plane where X is forward
    # arctan2(Y, X) gives angle from forward axis (X) to left axis (Y)
    angles = np.arctan2(points[:, 1], points[:, 0])  # Y relative to X (forward)
    angles_deg = np.degrees(angles)
    
    # Calculate FOV limits
    half_fov = fov_degrees / 2
    
    # Keep points within ±half_fov degrees from forward direction
    mask = (angles_deg >= -half_fov) & (angles_deg <= half_fov)
    
    return mask

def apply_manual_rotation_correction(R, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """
    Apply manual rotation corrections to the rotation matrix.
    """
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # Create individual rotation matrices
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad),  np.cos(roll_rad), 0],
        [0,                 0,               1]
    ])
    
    R_pitch = np.array([
        [1, 0,                 0                ],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw_rad),  0, np.sin(yaw_rad)],
        [0,                1, 0               ],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])
    
    R_correction = R_roll @ R_pitch @ R_yaw
    R_corrected = R_correction @ R
    return R_corrected

def lidar_3d_to_camera_2d_projection(lidar_points, R, t, K, dist=None):
    """
    Project 3D LiDAR points to 2D camera image coordinates.
    
    Args:
        lidar_points: Nx3 array of LiDAR points [x, y, z] in LiDAR frame
        R: 3x3 rotation matrix (LiDAR to Camera)  
        t: 3x1 translation vector (LiDAR to Camera)
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients (optional)
    
    Returns:
        uv_points: Nx2 array of image coordinates [u, v]
        depths: N array of depths (Z values in camera frame)
        valid_mask: N boolean array indicating valid projections
    """
    
    # Transform LiDAR points to camera coordinate frame
    points_3d = lidar_points.T  # 3xN
    camera_points = R @ points_3d + t  # 3xN
    camera_points = camera_points.T    # Nx3
    
    # Filter points behind camera (negative Z in camera frame)
    valid_z = camera_points[:, 2] > 0.1  # min depth threshold
    
    if np.sum(valid_z) == 0:
        print("[WARNING] No points have positive Z (in front of camera)")
        return np.array([]).reshape(0, 2), np.array([]), valid_z
    
    valid_points = camera_points[valid_z]
    
    # Project to 2D using pinhole camera model
    X, Y, Z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
    
    # Apply camera intrinsics matrix K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Project to image plane
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    
    uv_points = np.column_stack([u, v])
    
    # Apply distortion correction if provided
    if dist is not None and np.any(dist != 0):
        uv_distorted = uv_points.reshape(-1, 1, 2).astype(np.float32)
        uv_undistorted = cv2.undistortPoints(uv_distorted, K, dist, P=K)
        uv_points = uv_undistorted.reshape(-1, 2)
    
    return uv_points, Z, valid_z

def test_checkerboard_projection(R, t, K, dist=None):
    """
    Test projection using your checkerboard calibration data
    """
    print("\n[TEST] Testing checkerboard projection...")
    
    # Your LiDAR 3D points (X=left/right, Y=forward, Z=up/down)
    lidar_3d = np.array([
        [-0.3390885, 3.573442, -0.568522],
        [-0.7454906, 3.477403, -0.311146],
        [-0.364554, 3.591593, -0.443259],
        [-0.2483596, 3.59366, -0.315155],
        [-0.1818541, 3.716697, -0.723318],
        [-0.6471004, 3.651254, -0.455303],
        [-0.8174115, 3.61537, -0.720495],
        [-0.7178828, 3.770423, -0.607904]
    ])
    
    # Your corresponding 2D image points
    image_2d = np.array([
        [167.00, 117.00],
        [206.00, 116.00],
        [208.00, 145.00],
        [169.00, 148.00],
        [174.00, 124.00],
        [199.00, 122.00],
        [174.00, 131.00],
        [200.00, 131.00]
    ])
    
    print(f"[DEBUG] Your extrinsic assumes Velodyne standard: X=forward, Y=left, Z=up")
    print(f"[DEBUG] But your data shows: X=left/right, Y=forward, Z=up/down")
    print(f"[DEBUG] We need to transform your data to match the expected coordinate system")
    
    # Transform your coordinate system to match the extrinsic expectation
    # Your system: (X_lr, Y_fwd, Z_ud) -> Expected: (X_fwd, Y_left, Z_up)
    # Transformation: X_fwd = Y, Y_left = -X, Z_up = -Z
    lidar_3d_transformed = np.zeros_like(lidar_3d)
    lidar_3d_transformed[:, 0] = lidar_3d[:, 1]   # X_fwd = Y_original (forward)
    lidar_3d_transformed[:, 1] = -lidar_3d[:, 0]  # Y_left = -X_original (left)
    lidar_3d_transformed[:, 2] = -lidar_3d[:, 2]  # Z_up = -Z_original (up)
    
    print(f"\n[DEBUG] Coordinate transformation applied:")
    print(f"  Original system: X=lr, Y=fwd, Z=ud")
    print(f"  Expected system: X=fwd, Y=left, Z=up")
    print(f"  Transform: [X_fwd, Y_left, Z_up] = [Y, -X, -Z]")
    
    # Project using current calibration
    projected_uv, depths, valid_mask = lidar_3d_to_camera_2d_projection(
        lidar_3d_transformed, R, t, K, dist
    )
    
    if len(projected_uv) == 0:
        print("[ERROR] No valid projections!")
        return
    
    print(f"\n[TEST] Projection results:")
    print(f"  Original 3D points: {len(lidar_3d)}")
    print(f"  Valid projections: {len(projected_uv)}")
    
    # Calculate reprojection errors
    errors = []
    print(f"\n[TEST] Point-by-point comparison:")
    print(f"{'Point':<5} {'Original (X,Y,Z)':<25} {'Transformed (X,Y,Z)':<25} {'Expected (u,v)':<15} {'Projected (u,v)':<15} {'Error':<10}")
    print("-" * 110)
    
    for i in range(len(lidar_3d)):
        if i < len(projected_uv):
            expected = image_2d[i]
            proj_uv = projected_uv[i]
            error = np.linalg.norm(proj_uv - expected)
            errors.append(error)
            
            print(f"{i:<5} {str(tuple(lidar_3d[i])):<25} {str(tuple(lidar_3d_transformed[i])):<25} {str(tuple(expected)):<15} {str(tuple(proj_uv.round(1))):<15} {error:.1f}")
        else:
            print(f"{i:<5} {str(tuple(lidar_3d[i])):<25} {str(tuple(lidar_3d_transformed[i])):<25} {str(tuple(image_2d[i])):<15} {'INVALID':<15} {'N/A':<10}")
    
    if errors:
        print(f"\n[TEST] Reprojection error statistics:")
        print(f"  Mean error: {np.mean(errors):.2f} pixels")
        print(f"  Max error: {np.max(errors):.2f} pixels")
        print(f"  Min error: {np.min(errors):.2f} pixels")
        
        if np.mean(errors) > 50:
            print("[WARNING] High reprojection errors suggest calibration issues!")
        elif np.mean(errors) < 10:
            print("[SUCCESS] Good reprojection accuracy!")
    
    return projected_uv, errors

def create_depth_map(uv_points, depths, H, W, radius=1):
    """Create a depth map from projected points using z-buffer"""
    depth_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    
    valid_uv = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(valid_uv) == 0:
        return depth_map, mask
    
    uv_valid = uv_points[valid_uv]
    depths_valid = depths[valid_uv]
    
    for i, ((u, v), depth) in enumerate(zip(uv_valid, depths_valid)):
        u_int, v_int = int(round(u)), int(round(v))
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = u_int + dx, v_int + dy
                if 0 <= x < W and 0 <= y < H:
                    if not mask[y, x] or depth < depth_map[y, x]:
                        depth_map[y, x] = depth
                        mask[y, x] = True
    
    return depth_map, mask

def visualize_projection(img, uv_points, depths, valid_mask=None):
    """Create visualization of LiDAR projection on image"""
    if valid_mask is not None:
        uv_points = uv_points[valid_mask]
        depths = depths[valid_mask]
    
    H, W = img.shape[:2]
    
    inside = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
              (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(inside) == 0:
        return img.copy()
    
    uv_inside = uv_points[inside]
    depths_inside = depths[inside]
    
    result = img.copy()
    
    if len(depths_inside) > 0:
        d_min, d_max = np.percentile(depths_inside, [5, 95])
        depths_norm = np.clip((depths_inside - d_min) / (d_max - d_min + 1e-6), 0, 1)
        colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        for (u, v), color in zip(uv_inside.astype(int), colors):
            cv2.circle(result, (u, v), 2, color[0].tolist(), -1)  # Larger circles for visibility
    
    return result

def main():
    ap = argparse.ArgumentParser(description="Fixed LiDAR 3D to Camera 2D Projection")
    ap.add_argument("--pcd", required=True, help="Path to LiDAR point cloud file")
    ap.add_argument("--cam_yaml", required=True, help="Camera intrinsics YAML file")
    ap.add_argument("--ext_yaml", required=True, help="LiDAR-Camera extrinsics YAML file")
    ap.add_argument("--image", required=True, help="Camera image file")
    ap.add_argument("--out_npz", required=True, help="Output NPZ file")
    ap.add_argument("--debug_overlay", default=None, help="Debug visualization output")
    ap.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth threshold")
    ap.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth threshold")
    ap.add_argument("--manual_roll", type=float, default=0.0, help="Manual roll correction in degrees")
    ap.add_argument("--manual_cy_offset", type=float, default=0.0, help="Manual vertical principal point correction")
    ap.add_argument("--manual_cx_offset", type=float, default=0.0, help="Manual horizontal principal point correction")
    ap.add_argument("--manual_pitch", type=float, default=0.0, help="Manual pitch correction in degrees")
    ap.add_argument("--manual_yaw", type=float, default=0.0, help="Manual yaw correction in degrees")
    ap.add_argument("--splat_radius", type=int, default=1, help="Radius for splatting points")
    ap.add_argument("--forward_positive", action='store_true', help="Use positive Z as forward (default: negative Z)")
    ap.add_argument("--test_checkerboard", action='store_true', help="Test with checkerboard calibration data")
    args = ap.parse_args()
    
    # Load data
    print("[INFO] Loading camera parameters...")
    K, dist, size_hw = load_cam(args.cam_yaml)
    print(f"[INFO] Camera matrix K:\n{K}")
    print(f"[INFO] Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    print("[INFO] Loading extrinsics...")
    R, t = load_ext(args.ext_yaml)
    print(f"[INFO] Original rotation matrix R:\n{R}")
    print(f"[INFO] Translation vector t: {t.flatten()}")
    
    # Apply manual corrections
    if args.manual_roll != 0.0 or args.manual_pitch != 0.0 or args.manual_yaw != 0.0:
        print(f"[INFO] Applying manual rotation corrections:")
        print(f"  Roll: {args.manual_roll}°, Pitch: {args.manual_pitch}°, Yaw: {args.manual_yaw}°")
        R = apply_manual_rotation_correction(R, args.manual_roll, args.manual_pitch, args.manual_yaw)
    
    print("[INFO] Loading image...")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    H, W = img.shape[:2]
    print(f"[INFO] Image size: {H}x{W}")
    
    if args.manual_cx_offset != 0.0 or args.manual_cy_offset != 0.0:
        print(f"[INFO] Applying manual principal point corrections: Δcx={args.manual_cx_offset}, Δcy={args.manual_cy_offset}")
        K[0, 2] += args.manual_cx_offset
        K[1, 2] += args.manual_cy_offset
    
    # Test with checkerboard data first
    if args.test_checkerboard:
        test_checkerboard_projection(R, t, K, dist)
    
    print("[INFO] Loading LiDAR point cloud...")
    pcd = o3d.io.read_point_cloud(args.pcd)
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"[INFO] Loaded {len(points)} LiDAR points")
    
    print(f"[DEBUG] LiDAR coordinate ranges:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] (left/right)")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] (forward/back)")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] (up/down)")
    
    print(f"[INFO] Your coordinate system: X=left/right, Y=forward, Z=up/down")
    print(f"[INFO] Extrinsic expects: X=forward, Y=left, Z=up")
    print(f"[INFO] Applying coordinate transformation: [X_fwd, Y_left, Z_up] = [Y, -X, -Z]")
    
    # Transform points to match extrinsic coordinate system expectation
    points_transformed = np.zeros_like(points)
    points_transformed[:, 0] = points[:, 1]   # X_forward = Y_original
    points_transformed[:, 1] = -points[:, 0]  # Y_left = -X_original  
    points_transformed[:, 2] = -points[:, 2]  # Z_up = -Z_original
    points = points_transformed
    
    # Apply FOV filtering - Now using standard Velodyne coordinates (X=forward)
    print("[INFO] Filtering points by 120° horizontal FOV (standard Velodyne: X=forward)...")
    fov_mask = filter_fov(points, fov_degrees=120)
    points = points[fov_mask]
    print(f"[INFO] After FOV filtering: {len(points)} points")
    
    # Apply depth filtering - X is forward in standard Velodyne coordinates
    print(f"[INFO] Applying depth filtering ({args.min_depth}m to {args.max_depth}m) on X-axis...")
    depth_mask = (points[:, 0] > args.min_depth) & (points[:, 0] < args.max_depth)
    
    points_filtered = points[depth_mask]
    print(f"[INFO] After depth filtering: {len(points_filtered)} points")
    
    if len(points_filtered) == 0:
        print("[ERROR] No points remain after filtering!")
        return
    
    # Project LiDAR points
    print("[INFO] Projecting LiDAR points to camera image...")
    uv_points, depths_cam, valid_mask = lidar_3d_to_camera_2d_projection(
        points_filtered, R, t, K, dist
    )
    
    print(f"[INFO] Projection results:")
    print(f"  Valid projections: {len(uv_points)}")
    if len(uv_points) > 0:
        print(f"  Camera depth range: [{depths_cam.min():.2f}, {depths_cam.max():.2f}]")
        
        inside_image = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                       (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
        print(f"  Points inside image: {np.sum(inside_image)}")
        
        if np.sum(inside_image) > 0:
            uv_inside = uv_points[inside_image]
            print(f"  UV coordinate ranges:")
            print(f"    U: [{uv_inside[:, 0].min():.1f}, {uv_inside[:, 0].max():.1f}]")
            print(f"    V: [{uv_inside[:, 1].min():.1f}, {uv_inside[:, 1].max():.1f}]")
    
    # Create depth map
    print("[INFO] Creating depth map...")
    depth_map, mask = create_depth_map(uv_points, depths_cam, H, W, args.splat_radius)
    valid_pixels = np.sum(mask)
    print(f"[INFO] Depth map: {valid_pixels} valid pixels")
    
    # Save results
    print("[INFO] Saving results...")
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(args.out_npz, Dlidar=depth_map, Mlidar=mask)
    print(f"[OK] Saved depth map to: {args.out_npz}")
    
    # Create debug visualization
    if args.debug_overlay and len(uv_points) > 0:
        print("[INFO] Creating debug visualization...")
        visualization = visualize_projection(img, uv_points, depths_cam)
        
        os.makedirs(os.path.dirname(args.debug_overlay) or ".", exist_ok=True)
        cv2.imwrite(args.debug_overlay, visualization)
        print(f"[OK] Saved visualization to: {args.debug_overlay}")

if __name__ == "__main__":
    main()