#!/usr/bin/env python3
"""
Run comparison with baseline models and generate comparison metrics
"""

import os
import argparse
import pandas as pd
from models import run_baseline_comparison

def main():
    parser = argparse.ArgumentParser(description="Run baseline model comparisons")
    parser.add_argument('--img_dir', default='data/frames/front',
                       help='Directory containing input images')
    parser.add_argument('--lidar_dir', default='data/lidar',
                       help='Directory containing LiDAR data')
    parser.add_argument('--out_dir', default='data/baseline_results',
                       help='Output directory for results')
    parser.add_argument('--num_frames', type=int, default=100,
                       help='Number of frames to process')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Get list of image files
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.endswith('.png')])[:args.num_frames]
    
    all_results = []
    
    for img_file in img_files:
        frame_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(args.img_dir, img_file)
        lidar_path = os.path.join(args.lidar_dir, f"{frame_id}.npz")
        
        if not os.path.exists(lidar_path):
            print(f"Skipping frame {frame_id} - no LiDAR data")
            continue
            
        print(f"\nProcessing frame {frame_id}")
        results = run_baseline_comparison(img_path, lidar_path)
        
        # Add frame info
        for method, metrics in results.items():
            metrics['frame'] = frame_id
            metrics['method'] = method
            all_results.append(metrics)
    
    # Create DataFrame and save results
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.out_dir, 'baseline_comparison_results.csv')
    df.to_csv(csv_path, index=False)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    summary = df.groupby('method').mean()
    print(summary)
    
    print(f"\nResults saved to: {csv_path}")

if __name__ == '__main__':
    main()