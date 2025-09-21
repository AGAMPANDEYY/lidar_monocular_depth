"""
Lidar to Camera Projection Correction Script test 2 after debuig script.py 
"""

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

def lidar_3d_to_camera_2d_projection(lidar_points, R, t, K, dist=None, z_forward="positive"):
    """
    Project 3D LiDAR points to 2D camera image coordinates.
    
    This follows the standard computer vision pipeline:
    1. Transform LiDAR points to camera coordinate frame
    2. Project 3D camera points to 2D image plane
    3. Apply distortion correction if needed
    
    Args:
        lidar_points: Nx3 array of LiDAR points [x, y, z] in LiDAR frame
        R: 3x3 rotation matrix (LiDAR to Camera)  
        t: 3x1 translation vector (LiDAR to Camera)
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients (optional)
        z_forward: "positive" or "negative" - direction convention for forward in camera frame
    
    Returns:
        uv_points: Nx2 array of image coordinates [u, v]
        depths: N array of depths (Z values in camera frame)
        valid_mask: N boolean array indicating valid projections
    """
    
    # Step 1: Transform LiDAR points to camera coordinate frame
    points_3d = lidar_points.T  # 3xN
    
    # Apply rotation and translation: P_cam = R * P_lidar + t
    camera_points = R @ points_3d + t  # 3xN
    camera_points = camera_points.T    # Nx3
    
    # Step 2: Handle different Z-forward conventions
    if z_forward == "negative":
        # If negative Z is forward in your setup, flip the Z axis
        camera_points[:, 2] = -camera_points[:, 2]
        print(f"[INFO] Flipped Z axis - new Z range: [{camera_points[:, 2].min():.2f}, {camera_points[:, 2].max():.2f}]")
    
    # Filter points behind camera (should have positive Z after convention handling)
    valid_z = camera_points[:, 2] > 0.1  # min depth threshold
    
    if np.sum(valid_z) == 0:
        print(f"[WARNING] No points have positive Z (in front of camera)")
        print(f"[DEBUG] Camera Z range: [{camera_points[:, 2].min():.2f}, {camera_points[:, 2].max():.2f}]")
        # Try with negative Z convention automatically
        if z_forward == "positive":
            print("[INFO] Trying negative Z convention...")
            return lidar_3d_to_camera_2d_projection(lidar_points, R, t, K, dist, "negative")
        return np.array([]).reshape(0, 2), np.array([]), valid_z
    
    valid_points = camera_points[valid_z]
    
    # Step 3: Project to 2D using pinhole camera model
    X, Y, Z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
    
    # Apply camera intrinsics matrix K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Project to image plane
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    
    # Combine u, v coordinates
    uv_points = np.column_stack([u, v])
    
    # Step 4: Apply distortion correction if provided
    if dist is not None and np.any(dist != 0):
        # OpenCV expects normalized coordinates for undistortion
        # First convert to normalized coordinates
        uv_normalized = np.column_stack([
            (uv_points[:, 0] - cx) / fx,
            (uv_points[:, 1] - cy) / fy
        ])
        
        # Apply distortion correction
        uv_distorted = uv_normalized.reshape(-1, 1, 2).astype(np.float32)
        uv_undistorted = cv2.undistortPoints(uv_distorted, np.eye(3), dist)
        uv_normalized_corrected = uv_undistorted.reshape(-1, 2)
        
        # Convert back to pixel coordinates
        uv_points = np.column_stack([
            uv_normalized_corrected[:, 0] * fx + cx,
            uv_normalized_corrected[:, 1] * fy + cy
        ])
    
    return uv_points, Z, valid_z

def create_depth_map(uv_points, depths, H, W, radius=1):
    """Create a depth map from projected points using z-buffer"""
    depth_map = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)
    
    # Filter points within image bounds
    valid_uv = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(valid_uv) == 0:
        return depth_map, mask
    
    uv_valid = uv_points[valid_uv]
    depths_valid = depths[valid_uv]
    
    # Z-buffer rasterization
    for i, ((u, v), depth) in enumerate(zip(uv_valid, depths_valid)):
        u_int, v_int = int(round(u)), int(round(v))
        
        # Splat with radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                x, y = u_int + dx, v_int + dy
                if 0 <= x < W and 0 <= y < H:
                    if not mask[y, x] or depth < depth_map[y, x]:
                        depth_map[y, x] = depth
                        mask[y, x] = True
    
    return depth_map, mask

def check_coordinate_alignment(lidar_points, camera_points, R, t):
    """Debug function to check coordinate system alignment"""
    print("\n[DEBUG] Coordinate System Analysis:")
    
    # Transform a few sample points
    sample_indices = np.random.choice(len(lidar_points), min(5, len(lidar_points)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        lidar_pt = lidar_points[idx]
        # Manual transformation
        cam_pt = (R @ lidar_pt.reshape(3, 1) + t).flatten()
        
        print(f"  Point {i+1}:")
        print(f"    LiDAR: [{lidar_pt[0]:6.2f}, {lidar_pt[1]:6.2f}, {lidar_pt[2]:6.2f}]")
        print(f"    Camera: [{cam_pt[0]:6.2f}, {cam_pt[1]:6.2f}, {cam_pt[2]:6.2f}]")

def visualize_projection(img, uv_points, depths, valid_mask=None):
    """Create visualization of LiDAR projection on image"""
    if valid_mask is not None:
        uv_points = uv_points[valid_mask]
        depths = depths[valid_mask]
    
    H, W = img.shape[:2]
    
    # Filter points inside image
    inside = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
              (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
    
    if np.sum(inside) == 0:
        return img.copy()
    
    uv_inside = uv_points[inside]
    depths_inside = depths[inside]
    
    # Create colored visualization
    result = img.copy()
    
    if len(depths_inside) > 0:
        # Normalize depths for coloring
        d_min, d_max = np.percentile(depths_inside, [5, 95])
        depths_norm = np.clip((depths_inside - d_min) / (d_max - d_min + 1e-6), 0, 1)
        
        # Apply colormap
        colors = cv2.applyColorMap((depths_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Draw points with different sizes based on depth
        for (u, v), color, depth in zip(uv_inside.astype(int), colors, depths_inside):
            # Closer points are larger
            size = max(1, int(3 - depth/10))  # Adjust scaling as needed
            cv2.circle(result, (u, v), size, color[0].tolist(), -1)
    
    return result

def main():
    ap = argparse.ArgumentParser(description="Correct LiDAR 3D to Camera 2D Projection")
    ap.add_argument("--pcd", required=True, help="Path to LiDAR point cloud file")
    ap.add_argument("--cam_yaml", required=True, help="Camera intrinsics YAML file")
    ap.add_argument("--ext_yaml", required=True, help="LiDAR-Camera extrinsics YAML file")
    ap.add_argument("--image", required=True, help="Camera image file")
    ap.add_argument("--out_npz", required=True, help="Output NPZ file")
    ap.add_argument("--debug_overlay", default=None, help="Debug visualization output")
    ap.add_argument("--min_depth", type=float, default=0.1, help="Minimum depth threshold")
    ap.add_argument("--max_depth", type=float, default=100.0, help="Maximum depth threshold")
    ap.add_argument("--manual_cy_offset", type=float, default=0.0, 
                    help="Manual vertical principal point correction")
    ap.add_argument("--manual_cx_offset", type=float, default=0.0, 
                    help="Manual horizontal principal point correction")
    ap.add_argument("--splat_radius", type=int, default=2, 
                    help="Radius for splatting LiDAR points in depth map (pixels)")
    ap.add_argument("--z_forward", choices=["positive", "negative", "auto"], default="auto",
                    help="Set camera Z forward direction: 'positive', 'negative', or 'auto'")
    ap.add_argument("--debug_coords", action="store_true", 
                    help="Print coordinate transformation debug info")
    
    args = ap.parse_args()
    
    # Load data
    print("[INFO] Loading camera parameters...")
    K, dist, size_hw = load_cam(args.cam_yaml)
    print(f"[INFO] Camera matrix K:\n{K}")
    print(f"[INFO] Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    print("[INFO] Loading image...")
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    H, W = img.shape[:2]
    print(f"[INFO] Image size: {H}x{W}")
    print(f"[INFO] Image center would be: cx={W/2:.1f}, cy={H/2:.1f}")
    print(f"[INFO] Principal point offset: Δcx={K[0,2]-W/2:.1f}, Δcy={K[1,2]-H/2:.1f}")
    if dist is not None and np.any(dist != 0):
        print(f"[INFO] Distortion coefficients: {dist}")
    
    print("[INFO] Loading extrinsics...")
    R, t = load_ext(args.ext_yaml)
    print(f"[INFO] Rotation matrix R:\n{R}")
    print(f"[INFO] Translation vector t: {t.flatten()}")
    
    # Apply manual corrections if specified
    if args.manual_cx_offset != 0.0 or args.manual_cy_offset != 0.0:
        print(f"[INFO] Applying manual principal point corrections: Δcx={args.manual_cx_offset}, Δcy={args.manual_cy_offset}")
        K[0, 2] += args.manual_cx_offset  # cx correction
        K[1, 2] += args.manual_cy_offset  # cy correction
        print(f"[INFO] Corrected principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    print("[INFO] Loading LiDAR point cloud...")
    pcd = o3d.io.read_point_cloud(args.pcd)
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"[INFO] Loaded {len(points)} LiDAR points")
    
    # Print coordinate ranges for debugging
    print(f"[DEBUG] LiDAR coordinate ranges:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Debug coordinate transformations if requested
    if args.debug_coords:
        check_coordinate_alignment(points[:100], None, R, t)  # Check first 100 points
    
    # Apply basic filtering first (remove obvious outliers)
    distance_from_origin = np.linalg.norm(points, axis=1)
    reasonable_distance = (distance_from_origin > 0.1) & (distance_from_origin < args.max_depth)
    points_filtered = points[reasonable_distance]
    print(f"[INFO] After basic filtering: {len(points_filtered)} points")
    
    if len(points_filtered) == 0:
        print("[ERROR] No points remain after basic filtering!")
        return
    
    # Determine Z direction
    z_direction = args.z_forward
    if z_direction == "auto":
        # Try both directions and see which gives more valid projections
        print("[INFO] Auto-detecting Z direction...")
        z_direction = "positive"  # Start with positive
    
    # Project LiDAR points to camera image
    print(f"[INFO] Projecting LiDAR points to camera image (Z forward: {z_direction})...")
    uv_points, depths_cam, valid_mask = lidar_3d_to_camera_2d_projection(
        points_filtered, R, t, K, dist, z_direction
    )
    
    print(f"[INFO] Projection results:")
    print(f"  Valid projections: {len(uv_points)}")
    if len(depths_cam) > 0:
        print(f"  Camera depth range: [{depths_cam.min():.2f}, {depths_cam.max():.2f}]")
    
    if len(uv_points) > 0:
        inside_image = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                       (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
        print(f"  Points inside image: {np.sum(inside_image)}")
        
        if np.sum(inside_image) > 0:
            uv_inside = uv_points[inside_image]
            print(f"  UV coordinate ranges:")
            print(f"    U: [{uv_inside[:, 0].min():.1f}, {uv_inside[:, 0].max():.1f}]")
            print(f"    V: [{uv_inside[:, 1].min():.1f}, {uv_inside[:, 1].max():.1f}]")
        else:
            print("[WARNING] No points project inside the image bounds!")
            print("  This might indicate:")
            print("  - Incorrect extrinsic calibration")
            print("  - Wrong coordinate system conventions")
            print("  - Principal point offset issues")
    
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
        
        # Also create scatter plot
        scatter_path = args.debug_overlay.replace(".png", "_scatter.png")
        scatter = img.copy()
        inside = ((uv_points[:, 0] >= 0) & (uv_points[:, 0] < W) & 
                 (uv_points[:, 1] >= 0) & (uv_points[:, 1] < H))
        for (u, v) in uv_points[inside].astype(int):
            cv2.circle(scatter, (u, v), 1, (0, 255, 255), -1)
        cv2.imwrite(scatter_path, scatter)
        print(f"[OK] Saved scatter plot to: {scatter_path}")

if __name__ == "__main__":
    main()