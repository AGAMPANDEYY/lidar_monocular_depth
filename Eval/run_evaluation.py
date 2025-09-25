import os
import argparse
import pandas as pd
from glob import glob
from dense_metrics import process_dense_metrics

def main():
    parser = argparse.ArgumentParser(description='Run depth evaluation with confidence intervals')
    parser.add_argument('--data_dir', type=str, required=True, help='Base directory with depth data')
    parser.add_argument('--output', type=str, default='depth_metrics.csv', help='Output CSV file')
    args = parser.parse_args()

    # Define range bins
    range_bins = ['0-10m', '10-25m', '25-50m', '>50m']
    
    # Get list of frames to process
    frames = sorted(glob(os.path.join(args.data_dir, 'lidar', '*_lidar.npy')))
    if not frames:
        print(f"No depth frames found in {args.data_dir}/lidar/")
        return
    
    print(f"Found {len(frames)} frames to process")
    
    # Process metrics
    results = process_dense_metrics(range_bins, frames, args.data_dir)
    
    # Convert results to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print("\nSummary:")
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()