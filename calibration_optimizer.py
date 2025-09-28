import numpy as np
import cv2
import yaml
import itertools
from pathlib import Path

def load_calibration(cam_yaml, ext_yaml):
    """Load camera and extrinsic parameters"""
    with open(cam_yaml, "r") as f:
        cam = yaml.safe_load(f)
    K = np.array(cam["K"], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam.get("dist", [0,0,0,0,0]), dtype=np.float64)
    
    with open(ext_yaml, "r") as f:
        ext = yaml.safe_load(f)
    R = np.array(ext["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(ext["t"], dtype=np.float64).reshape(3, 1)
    
    return K, dist, R, t

def apply_corrections(R, t, K, roll=0, pitch=0, yaw=0, tx=0, ty=0, tz=0, cx_offset=0, cy_offset=0):
    """Apply manual corrections to calibration parameters"""
    R_corrected = R.copy()
    t_corrected = t.copy()
    K_corrected = K.copy()
    
    # Apply rotation corrections
    if roll != 0 or pitch != 0 or yaw != 0:
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
        R_corrected = R_correction @ R
    
    # Apply translation corrections
    t_corrected += np.array([[tx], [ty], [tz]])
    
    # Apply principal point corrections
    K_corrected[0, 2] += cx_offset  # cx
    K_corrected[1, 2] += cy_offset  # cy
    
    return R_corrected, t_corrected, K_corrected

def project_checkerboard_points(R, t, K, dist):
    """Project checkerboard points and calculate reprojection error"""
    # Your checkerboard data (using [Y, X, Z] transformation)
    lidar_3d_original = np.array([
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
    
    # Apply [Y, X, Z] coordinate transformation
    lidar_3d = np.column_stack([lidar_3d_original[:, 1], lidar_3d_original[:, 0], lidar_3d_original[:, 2]])
    
    # Project to camera
    points_3d = lidar_3d.T  # 3xN
    camera_points = R @ points_3d + t  # 3xN
    camera_points = camera_points.T    # Nx3
    
    # Check if points are in front of camera
    valid_z = camera_points[:, 2] > 0.1
    if not np.all(valid_z):
        return float('inf')  # Some points behind camera
    
    # Project to image plane
    X, Y, Z = camera_points[:, 0], camera_points[:, 1], camera_points[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    projected_uv = np.column_stack([u, v])
    
    # Apply distortion correction
    if dist is not None and np.any(dist != 0):
        uv_distorted = projected_uv.reshape(-1, 1, 2).astype(np.float32)
        uv_undistorted = cv2.undistortPoints(uv_distorted, K, dist, P=K)
        projected_uv = uv_undistorted.reshape(-1, 2)
    
    # Calculate reprojection errors
    errors = [np.linalg.norm(proj - exp) for proj, exp in zip(projected_uv, image_2d)]
    mean_error = np.mean(errors)
    
    return mean_error

def grid_search_optimization(K, dist, R, t):
    """Perform grid search to find optimal manual corrections"""
    print("[OPTIMIZER] Starting grid search optimization...")
    
    # Define search ranges (start small to avoid divergence)
    translation_range = np.arange(-0.2, 0.25, 0.05)  # ±20cm in 5cm steps
    rotation_range = np.arange(-3.0, 3.5, 0.5)       # ±3° in 0.5° steps
    principal_point_range = np.arange(-15, 20, 5)    # ±15 pixels in 5px steps
    
    best_error = float('inf')
    best_params = None
    
    # Test translation corrections first (most impactful)
    print("[OPTIMIZER] Testing translation corrections...")
    for tx, ty, tz in itertools.product(translation_range[:3], translation_range[:3], translation_range[:3]):
        R_corr, t_corr, K_corr = apply_corrections(R, t, K, tx=tx, ty=ty, tz=tz)
        error = project_checkerboard_points(R_corr, t_corr, K_corr, dist)
        
        if error < best_error:
            best_error = error
            best_params = {'tx': tx, 'ty': ty, 'tz': tz, 'roll': 0, 'pitch': 0, 'yaw': 0, 'cx_offset': 0, 'cy_offset': 0}
    
    print(f"[OPTIMIZER] Best translation correction: {best_params}, Error: {best_error:.1f}px")
    
    # Refine with rotation corrections around best translation
    if best_params and best_error < 30:  # Only if translation helped significantly
        print("[OPTIMIZER] Testing rotation corrections...")
        base_params = best_params.copy()
        
        for roll, pitch, yaw in itertools.product(rotation_range[::2], rotation_range[::2], rotation_range[::2]):  # Coarser grid
            params = base_params.copy()
            params.update({'roll': roll, 'pitch': pitch, 'yaw': yaw})
            
            R_corr, t_corr, K_corr = apply_corrections(R, t, K, **params)
            error = project_checkerboard_points(R_corr, t_corr, K_corr, dist)
            
            if error < best_error:
                best_error = error
                best_params = params
    
    print(f"[OPTIMIZER] Best rotation correction: {best_params}, Error: {best_error:.1f}px")
    
    # Refine with principal point corrections
    if best_params and best_error < 25:
        print("[OPTIMIZER] Testing principal point corrections...")
        base_params = best_params.copy()
        
        for cx_offset, cy_offset in itertools.product(principal_point_range, principal_point_range):
            params = base_params.copy()
            params.update({'cx_offset': cx_offset, 'cy_offset': cy_offset})
            
            R_corr, t_corr, K_corr = apply_corrections(R, t, K, **params)
            error = project_checkerboard_points(R_corr, t_corr, K_corr, dist)
            
            if error < best_error:
                best_error = error
                best_params = params
    
    print(f"[OPTIMIZER] Final best parameters: {best_params}, Error: {best_error:.1f}px")
    
    return best_params, best_error

def save_optimized_calibration(K, R, t, best_params, output_dir="optimized_calibration"):
    """Save optimized calibration parameters to YAML files"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Apply best corrections
    R_opt, t_opt, K_opt = apply_corrections(R, t, K, **best_params)
    
    # Save optimized camera parameters
    cam_data = {
        'size': [279, 352],  # Your image size
        'K': K_opt.tolist(),
        'dist': [-0.57554184, 1.26948481, 0.0, 0.0, 0.0]  # Keep original distortion
    }
    
    with open(f"{output_dir}/camera_optimized.yaml", 'w') as f:
        yaml.dump(cam_data, f, default_flow_style=False)
    
    # Save optimized extrinsics
    ext_data = {
        'R': R_opt.tolist(),
        't': t_opt.flatten().tolist()
    }
    
    with open(f"{output_dir}/extrinsics_optimized.yaml", 'w') as f:
        yaml.dump(ext_data, f, default_flow_style=False)
    
    # Save correction parameters for reference
    correction_data = {
        'original_error_px': 41.1,
        'optimized_error_px': float(project_checkerboard_points(R_opt, t_opt, K_opt, 
                                                                np.array([-0.57554184, 1.26948481, 0.0, 0.0, 0.0]))),
        'corrections_applied': best_params
    }
    
    with open(f"{output_dir}/optimization_report.yaml", 'w') as f:
        yaml.dump(correction_data, f, default_flow_style=False)
    
    print(f"[OPTIMIZER] Saved optimized calibration to {output_dir}/")
    print(f"[OPTIMIZER] Use these files with your LiDAR projection script")
    
    return R_opt, t_opt, K_opt

def main():
    # Your calibration files
    cam_yaml = "calibration/camera.yaml"
    ext_yaml = "calibration/extrinsics_lidar_to_cam.yaml"
    
    print("[OPTIMIZER] Loading original calibration...")
    K, dist, R, t = load_calibration(cam_yaml, ext_yaml)
    
    # Calculate baseline error
    baseline_error = project_checkerboard_points(R, t, K, dist)
    print(f"[OPTIMIZER] Baseline checkerboard error: {baseline_error:.1f}px")
    
    if baseline_error > 15.0:  # Only optimize if error is significant
        # Run optimization
        best_params, best_error = grid_search_optimization(K, dist, R, t)
        
        if best_params and best_error < baseline_error * 0.8:  # At least 20% improvement
            print(f"[OPTIMIZER] Optimization successful! Error reduced from {baseline_error:.1f}px to {best_error:.1f}px")
            
            # Save optimized calibration
            R_opt, t_opt, K_opt = save_optimized_calibration(K, R, t, best_params)
            
            # Print command line for testing
            print(f"\n[OPTIMIZER] Test optimized calibration with:")
            print(f"python working_script.py --cam_yaml optimized_calibration/camera_optimized.yaml --ext_yaml optimized_calibration/extrinsics_optimized.yaml --test_checkerboard ...")
            
        else:
            print(f"[OPTIMIZER] Optimization did not significantly improve results.")
            print(f"[OPTIMIZER] Manual calibration refinement may be needed.")
    else:
        print(f"[OPTIMIZER] Baseline error is already quite good ({baseline_error:.1f}px)")

if __name__ == "__main__":
    main()