#!/usr/bin/env python3
"""
Generate ROIs YAML file with proper dimensions from camera calibration.
This ensures each view (front, right, rear, left) is cropped to exactly
the camera calibration dimensions (352x279).
"""
import os
import yaml
import argparse

def load_camera_dimensions(camera_yaml):
    """Load camera dimensions from calibration YAML."""
    if not os.path.exists(camera_yaml):
        raise FileNotFoundError(f"Camera calibration file not found: {camera_yaml}")
        
    with open(camera_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'size' not in data:
        raise ValueError(f"No 'size' field found in {camera_yaml}")
    
    width, height = data['size']
    return width, height

def generate_rois_yaml(camera_yaml, output_yaml, input_resolution=(1408, 558)):
    """
    Generate ROIs YAML with proper dimensions.
    
    Args:
        camera_yaml: Path to camera calibration YAML
        output_yaml: Path to save ROIs YAML
        input_resolution: Full input image resolution (width, height)
    """
    # Get target dimensions from camera calibration
    width, height = load_camera_dimensions(camera_yaml)
    print(f"\nCamera calibration dimensions: {width}x{height}")
    
    # Input image is typically 4 views arranged in 2x2 grid
    # Verify input resolution can accommodate this
    if input_resolution[0] < width * 2 or input_resolution[1] < height * 2:
        raise ValueError(
            f"Input resolution {input_resolution} too small for 2x2 grid "
            f"of {width}x{height} views"
        )
    
    # Define ROIs for each view
    rois = {
        'rois': {
            'front': [0, 0, width, height],               # Top-left
            'right': [width, 0, width*2, height],         # Top-right
            'rear':  [0, height, width, height*2],        # Bottom-left
            'left': [width, height, width*2, height*2]    # Bottom-right
        },
        'crop_dimensions': {
            view: {'width': width, 'height': height}
            for view in ['front', 'right', 'rear', 'left']
        }
    }
    
    # Save to YAML
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)
    with open(output_yaml, 'w') as f:
        yaml.safe_dump(rois, f, default_flow_style=None)
    
    print(f"\nGenerated ROIs YAML at: {output_yaml}")
    print("ROI dimensions:")
    for view, roi in rois['rois'].items():
        print(f"- {view}: ({roi[0]}, {roi[1]}) to ({roi[2]}, {roi[3]})")
    print(f"\nEach view will be cropped to exactly {width}x{height}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera_yaml', default='calibration/camera.yaml',
                      help='Path to camera calibration YAML')
    parser.add_argument('--output_yaml', default='scripts/config/rois.yaml',
                      help='Path to save ROIs YAML')
    parser.add_argument('--input_width', type=int, default=1408,
                      help='Input image width (typically 2*crop_width)')
    parser.add_argument('--input_height', type=int, default=558,
                      help='Input image height (typically 2*crop_height)')
    
    args = parser.parse_args()
    
    generate_rois_yaml(
        args.camera_yaml,
        args.output_yaml,
        (args.input_width, args.input_height)
    )

if __name__ == '__main__':
    main()