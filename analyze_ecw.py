import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('data/fused_output/object_depth_metrics.csv')

# Print basic statistics
print("\n=== ECW Analysis ===")
print(f"Total records: {len(df)}")

# Warning flags analysis
print("\n=== Warning Flags Distribution ===")
print("warn_raw:", df['warn_raw'].value_counts())
print("warn_stable:", df['warn_stable'].value_counts())
print("warn_raw_fused:", df['warn_raw_fused'].value_counts())
print("warn_stable_fused:", df['warn_stable_fused'].value_counts())

# TTC Analysis
print("\n=== TTC Statistics ===")
print("TTC distribution:")
print(df['ttc'].describe())

# Count objects by class with warnings
print("\n=== Objects with Warnings by Class ===")
warnings = df[df['warn_raw'] | df['warn_raw_fused']]
print(warnings['class'].value_counts())

# Analyze depth ranges for warning cases
print("\n=== Depth Analysis for Warning Cases ===")
warnings_depth = df[df['warn_raw'] | df['warn_raw_fused']]
print("\nDepth statistics for warning cases:")
print("Mono depth:", warnings_depth['mono_median_depth'].describe())
print("Lidar depth:", warnings_depth['lidar_median_depth'].describe())
print("Fused depth:", warnings_depth['fused_median_depth'].describe())