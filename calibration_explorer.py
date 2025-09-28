import argparse, os
import subprocess
import numpy as np
from pathlib import Path

def run_diagnostic(pcd, image, base_output_dir, param_combination, cam_yaml, ext_yaml):
    """Run diagnostic tool with given parameter combination"""
    # Create unique name for this combination
    param_str = "_".join([f"{k}{v}" for k, v in param_combination.items()])
    output_dir = Path(base_output_dir) / param_str
    output_dir.mkdir(parents=True, exist_ok=True)
    
    out_npz = str(output_dir / "depth.npz")
    debug_overlay = str(output_dir / "overlay.png")
    param_file = output_dir / "params.txt"
    
    # Build command
    cmd = [
        "/Users/agampandey/work/lidar_monocular_depth/.venv/bin/python", "diagnostic_tool.py",
        "--pcd", pcd,
        "--image", image,
        "--cam_yaml", cam_yaml,
        "--ext_yaml", ext_yaml,
        "--out_npz", out_npz,
        "--debug_overlay", debug_overlay,
        "--test_checkerboard"
    ]
    
    # Add manual corrections
    for param, value in param_combination.items():
        cmd.extend([f"--{param}", str(value)])
    
    # Save parameters to file
    with open(param_file, 'w') as f:
        for param, value in param_combination.items():
            f.write(f"{param}: {value}\n")
    
    # Run command
    print(f"\nTrying combination: {param_str}")
    subprocess.run(cmd)
    
    return debug_overlay

def main():
    parser = argparse.ArgumentParser(description="Explore LiDAR-Camera calibration parameters")
    parser.add_argument("--pcd", required=True, help="LiDAR point cloud file")
    parser.add_argument("--image", required=True, help="Camera image file")
    parser.add_argument("--output_dir", required=True, help="Base output directory")
    parser.add_argument("--cam_yaml", required=True, help="Camera intrinsics file")
    parser.add_argument("--ext_yaml", required=True, help="Extrinsics file")
    
    args = parser.parse_args()
    
    # Parameter ranges to explore
    param_ranges = {
        'manual_yaw': [-2, 0, 2, 4],
        'manual_pitch': [-1.5, -0.5, 0, 0.5, 1.5],
        'manual_ty': [-0.3, 0, 0.3, 0.6, 1.2],
        'manual_tz': [-0.1, 0, 0.1, 0.2],
        'manual_cx': [-10, 0, 10, 20]
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try combinations
    kept_combinations = []
    discarded_combinations = []
    
    # Generate combinations (one parameter at a time to start)
    for param, values in param_ranges.items():
        print(f"\nTesting {param} variations...")
        for value in values:
            param_combination = {param: value}
            
            # Run diagnostic
            overlay_path = run_diagnostic(
                args.pcd, args.image, args.output_dir, 
                param_combination, args.cam_yaml, args.ext_yaml
            )
            
            # Ask for user input
            print(f"\nResult saved to: {overlay_path}")
            response = input("Keep this combination? (y/n): ").lower()
            
            if response == 'y':
                kept_combinations.append(param_combination)
            else:
                discarded_combinations.append(param_combination)
    
    # Now try combinations of parameters that worked well
    if len(kept_combinations) > 1:
        print("\nTrying combinations of successful parameters...")
        
        # Combine parameters from successful single-parameter tests
        combined_params = {}
        for combo in kept_combinations:
            combined_params.update(combo)
        
        overlay_path = run_diagnostic(
            args.pcd, args.image, args.output_dir,
            combined_params, args.cam_yaml, args.ext_yaml
        )
        
        print(f"\nFinal combined result saved to: {overlay_path}")
        print("Combined parameters:")
        for param, value in combined_params.items():
            print(f"  {param}: {value}")
    
    # Print summary
    print("\nExploration complete!")
    print("\nKept combinations:")
    for combo in kept_combinations:
        print(f"  {combo}")
    
    print("\nDiscarded combinations:")
    for combo in discarded_combinations:
        print(f"  {combo}")

if __name__ == "__main__":
    main()