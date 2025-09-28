import argparse, os, yaml
import numpy as np
import cv2
import open3d as o3d

def load_cam(cam_yaml):
    with open(cam_yaml, "r") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam.get("dist", [0,0,0,0,0]), dtype=np.float64)
    return K, dist

def load_ext(ext_yaml):
    with open(ext_yaml, "r") as f:
        ext = yaml.safe_load(f)
    R = np.array(ext["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(ext["t"], dtype=np.float64).reshape(3, 1)
    return R, t

def test_checkerboard_projection(R, t, K, dist=None):
    """Test with checkerboard data using [Y, X, Z] transformation"""
    print("\n[TEST] Testing checkerboard projection...")
    
    # Your original data
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
    
    image_2d = np.array([
        [167.00, 117.00], [206.00, 116.00], [208.00, 145.00], [169.00, 148.00],
        [174.00, 124.00], [199.00, 122.00], [174.00, 131.00], [200.00, 131.00]
    ])
    
    # Apply [Y, X, Z] transformation
    lidar_transformed = np.column_stack([lidar_3d[:, 1], lidar_3d[:, 0], lidar_3d[:, 2]])
    
    # Project
    projected_uv, _, _ = project_points(lidar_transformed, R, t, K, dist)
    
    if len(projected_uv) > 0:
        errors = [np.linalg.norm(proj - exp) for proj, exp in zip(projected_uv, image_2d)]
        print(f"[TEST] Mean reprojection error: {np.mean(errors):.1f} pixels")
        
        # Show detailed results
        for i, (orig, proj, exp, err) in enumerate(zip(lidar_3d, projected_uv, image_2d, errors)):
            print(f"  Point {i}: {orig} -> Expected {exp}, Got {proj.round(1)}, Error: {err:.1f}px")
        
        return np.mean(errors)
    return float('inf')

def apply_manual_corrections(R, t, roll=0, pitch=0, yaw=0, tx=0, ty=0, tz=0):
    """Apply manual rotation and translation corrections"""
    if roll != 0 or pitch != 0 or yaw != 0:
        # Create rotation matrices
        roll_rad, pitch_rad, yaw_rad = np.radians([roll, pitch, yaw])
        
        R_roll = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad),  np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw_rad),  0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        R_correction = R_roll @ R_pitch @ R_yaw
        R = R_correction @ R
    
    # Apply translation corrections
    t = t + np.array([[tx], [ty], [tz]])
    
    return R, t

def project_points(points, R, t, K, dist=None):
    """Project 3D points to 2D image coordinates"""
    if len(points) == 0:
        return np.array([]).reshape(0, 2), np.array([]), np.array([], dtype=bool)
    
    # Transform to camera frame
    points_3d = points.T  # 3xN
    camera_points = R @ points_3d + t  # 3xN
    camera_points = camera_points.T    # Nx3
    
    # Filter points behind camera
    valid_z = camera_points[:, 2] > 0.1
    
    if np.sum(valid_z) == 0:
        return np.array([]).reshape(0, 2), np.array([]), valid_z
    
    valid_points = camera_points[valid_z]
    X, Y, Z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
    
    # Project to image
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    uv_points = np.column_stack([u, v])
    
    # Apply distortion correction
    if dist is not None and np.any(dist != 0):
        uv_distorted = uv_points.reshape(-1, 1, 2).astype(np.float32)
        uv_undistorted = cv2.undistortPoints(uv_distorted, K, dist, P=K)
        uv_points = uv_undistorted.reshape(-1, 2)
    
    return uv_points, Z, valid_z

def filter_and_sample_points(points, max_points=500, min_depth=0.5, max_depth=50.0):
    """Aggressively filter and sample points for 16-beam LiDAR"""
    print(f"[INFO] Starting with {len(points)} points")
    
    # Apply coordinate transformation [Y, X, Z] 
    points_transformed = np.column_stack([points[:, 1], points[:, 0], points[:, 2]])
    print(f"[INFO] Applied coordinate transformation [Y, X, Z]")
    
    # 1. Distance filtering (X is forward after transformation)
    depth_mask = (points_transformed[:, 0] > min_depth) & (points_transformed[:, 0] < max_depth)
    points_transformed = points_transformed[depth_mask]
    print(f"[INFO] After depth filtering ({min_depth}-{max_depth}m): {len(points_transformed)} points")
    
    # 2. Height filtering (remove ground and very high points)
    height_mask = (points_transformed[:, 2] > -2.0) & (points_transformed[:, 2] < 5.0)
    points_transformed = points_transformed[height_mask]
    print(f"[INFO] After height filtering (-2 to 5m): {len(points_transformed)} points")
    
    # 3. FOV filtering (120 degrees horizontal)
    angles = np.arctan2(points_transformed[:, 1], points_transformed[:, 0])  # Y relative to X
    angles_deg = np.degrees(angles)
    fov_mask = (angles_deg >= -60) & (angles_deg <= 60)  # ±60° = 120° total
    points_transformed = points_transformed[fov_mask]
    print(f"[INFO] After FOV filtering (120°): {len(points_transformed)} points")
    
    # 4. Distance-based subsampling
    if len(points_transformed) > max_points:
        distances = np.sqrt(np.sum(points_transformed**2, axis=1))
        
        # Keep closer points preferentially
        close_mask = distances <= 15.0
        far_mask = distances > 15.0
        
        close_points = points_transformed[close_mask]
        far_points = points_transformed[far_mask]
        
        # Sample far points more sparsely
        if len(far_points) > max_points // 3:
            indices = np.random.choice(len(far_points), max_points // 3, replace=False)
            far_points = far_points[indices]
        
        # Sample close points
        if len(close_points) > (max_points * 2) // 3:
            indices = np.random.choice(len(close_points), (max_points * 2) // 3, replace=False)
            close_points = close_points[indices]
        
        points_transformed = np.vstack([close_points, far_points]) if len(far_points) > 0 else close_points
        print(f"[INFO] After distance-based subsampling: {len(points_transformed)} points")
    
    # 5. Final random sampling if still too many
    if len(points_transformed) > max_points:
        indices = np.random.choice(len(points_transformed), max_points, replace=False)
        points_transformed = points_transformed[indices]
        print(f"[INFO] After final sampling: {len(points_transformed)} points")
    
    return points_transformed

def create_depth_map(uv_points, depths, H, W, radius=2):
    """Create depth map from projected points"""
    depth_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    
    # Filter points within image bounds
    valid_uv = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(valid_uv) == 0:
        return depth_map, mask
    
    uv_valid = uv_points[valid_uv]
    depths_valid = depths[valid_uv]
    
    # Splat points with radius
    for (u, v), depth in zip(uv_valid, depths_valid):
        u_int, v_int = int(round(u)), int(round(v))
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = u_int + dx, v_int + dy
                if 0 <= x < W and 0 <= y < H:
                    if not mask[y, x] or depth < depth_map[y, x]:
                        depth_map[y, x] = depth
                        mask[y, x] = True
    
    return depth_map, mask

def create_visualization(img, uv_points, depths):
    """Create colored visualization"""
    H, W = img.shape[:2]
    result = img.copy()
    
    # Filter points inside image
    inside = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
              (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(inside) == 0:
        return result
    
    uv_inside = uv_points[inside]
    depths_inside = depths[inside]
    
    # Color by depth
    if len(depths_inside) > 0:
        d_min, d_max = np.percentile(depths_inside, [5, 95])
        depths_norm = np.clip((depths_inside - d_min) / (d_max - d_min + 1e-6), 0, 1)
        colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        for (u, v), color in zip(uv_inside.astype(int), colors):
            cv2.circle(result, (u, v), 2, color[0].tolist(), -1)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Fixed LiDAR-Camera Projection")
    parser.add_argument("--pcd", required=True, help="LiDAR point cloud file")
    parser.add_argument("--cam_yaml", required=True, help="Camera intrinsics file")
    parser.add_argument("--ext_yaml", required=True, help="Extrinsics file")
    parser.add_argument("--image", required=True, help="Camera image file")
    parser.add_argument("--out_npz", required=True, help="Output depth map file")
    parser.add_argument("--debug_overlay", help="Debug visualization output")
    parser.add_argument("--test_checkerboard", action='store_true', help="Test with checkerboard data")
    
    # Manual correction parameters
    parser.add_argument("--manual_roll", type=float, default=0.0, help="Manual roll correction (degrees)")
    parser.add_argument("--manual_pitch", type=float, default=0.0, help="Manual pitch correction (degrees)")
    parser.add_argument("--manual_yaw", type=float, default=0.0, help="Manual yaw correction (degrees)")
    parser.add_argument("--manual_tx", type=float, default=0.0, help="Manual translation X (meters)")
    parser.add_argument("--manual_ty", type=float, default=0.0, help="Manual translation Y (meters)")
    parser.add_argument("--manual_tz", type=float, default=0.0, help="Manual translation Z (meters)")
    parser.add_argument("--manual_cx", type=float, default=0.0, help="Manual principal point X offset")
    parser.add_argument("--manual_cy", type=float, default=0.0, help="Manual principal point Y offset")
    
    # Filtering parameters
    parser.add_argument("--max_points", type=int, default=500, help="Maximum number of points to project")
    parser.add_argument("--min_depth", type=float, default=0.5, help="Minimum depth (meters)")
    parser.add_argument("--max_depth", type=float, default=50.0, help="Maximum depth (meters)")
    
    args = parser.parse_args()
    
    # Load calibration data
    print("[INFO] Loading calibration data...")
    K, dist = load_cam(args.cam_yaml)
    R, t = load_ext(args.ext_yaml)
    
    print(f"[INFO] Camera matrix K:\n{K}")
    print(f"[INFO] Distortion: {dist}")
    print(f"[INFO] Rotation matrix R:\n{R}")
    print(f"[INFO] Translation vector t: {t.flatten()}")
    
    # Apply manual corrections
    if any([args.manual_roll, args.manual_pitch, args.manual_yaw, args.manual_tx, args.manual_ty, args.manual_tz]):
        print(f"[INFO] Applying manual corrections:")
        print(f"  Rotation: roll={args.manual_roll}°, pitch={args.manual_pitch}°, yaw={args.manual_yaw}°")
        print(f"  Translation: tx={args.manual_tx}m, ty={args.manual_ty}m, tz={args.manual_tz}m")
        R, t = apply_manual_corrections(R, t, args.manual_roll, args.manual_pitch, args.manual_yaw,
                                      args.manual_tx, args.manual_ty, args.manual_tz)
    
    # Apply principal point corrections
    if args.manual_cx != 0.0 or args.manual_cy != 0.0:
        print(f"[INFO] Applying principal point corrections: cx+={args.manual_cx}, cy+={args.manual_cy}")
        K[0, 2] += args.manual_cx
        K[1, 2] += args.manual_cy
    
    # Test checkerboard projection if requested
    if args.test_checkerboard:
        error = test_checkerboard_projection(R, t, K, dist)
        if error > 20:
            print(f"[WARNING] High checkerboard error ({error:.1f}px) - consider manual corrections")
    
    # Load image
    print(f"[INFO] Loading image...")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    H, W = img.shape[:2]
    print(f"[INFO] Image size: {H}x{W}")
    
    # Load LiDAR data
    print(f"[INFO] Loading LiDAR point cloud...")
    pcd = o3d.io.read_point_cloud(args.pcd)
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"[INFO] Loaded {len(points)} LiDAR points")
    
    # Filter and sample points
    points_filtered = filter_and_sample_points(points, args.max_points, args.min_depth, args.max_depth)
    
    if len(points_filtered) == 0:
        print("[ERROR] No points remain after filtering!")
        return
    
    # Project points to image
    print(f"[INFO] Projecting {len(points_filtered)} points to image...")
    uv_points, depths, valid_mask = project_points(points_filtered, R, t, K, dist)
    
    print(f"[INFO] Projection results:")
    print(f"  Valid projections: {len(uv_points)}")
    
    if len(uv_points) > 0:
        inside_image = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                       (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
        print(f"  Points inside image: {np.sum(inside_image)}")
        print(f"  Depth range: [{depths.min():.1f}, {depths.max():.1f}] meters")
        
        # Create depth map
        print(f"[INFO] Creating depth map...")
        depth_map, mask = create_depth_map(uv_points, depths, H, W)
        valid_pixels = np.sum(mask)
        print(f"[INFO] Depth map: {valid_pixels} valid pixels")
        
        # Save results
        print(f"[INFO] Saving results...")
        os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
        np.savez_compressed(args.out_npz, Dlidar=depth_map, Mlidar=mask)
        print(f"[OK] Saved depth map to: {args.out_npz}")
        
        # Create debug visualization
        if args.debug_overlay:
            print(f"[INFO] Creating debug visualization...")
            visualization = create_visualization(img, uv_points, depths)
            os.makedirs(os.path.dirname(args.debug_overlay) or ".", exist_ok=True)
            cv2.imwrite(args.debug_overlay, visualization)
            print(f"[OK] Saved visualization to: {args.debug_overlay}")
    else:
        print("[ERROR] No valid projections!")

if __name__ == "__main__":
    main()