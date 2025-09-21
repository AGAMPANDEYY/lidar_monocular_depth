import os
import argparse
import numpy as np
from velodyne_pcap import VelodyneCapture
import open3d as o3d

def main(pcap_path, output_dir, max_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    # Open Velodyne PCAP
    capture = VelodyneCapture(pcap_path)
    for idx, scan in enumerate(capture):
        if idx >= max_frames:
            break
        # scan.points is (N, 4): x, y, z, intensity
        xyz = scan.points[:, :3]
        # Save as .npy
        np.save(os.path.join(output_dir, f"{idx:05d}.npy"), xyz)
        # Save as .pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"{idx:05d}.pcd"), pcd)
        print(f"Saved {idx:05d}.pcd and .npy")
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Velodyne PCAP to PCD files")
    parser.add_argument("--pcap", required=True, help="Path to Velodyne .pcap file")
    parser.add_argument("--output_dir", required=True, help="Output directory for point clouds")
    parser.add_argument("--max_frames", type=int, default=10, help="Number of frames to extract")
    args = parser.parse_args()
    main(args.pcap, args.output_dir, args.max_frames)
