"""
Data Preprocessing Module for ECW-Fusion-India

- Extracts frames from multi-view MP4 video
- Crops each frame into individual views using ROIs from YAML
- (Optional) Resizes each crop for downstream models
- (Optional) Converts PCAP LiDAR data to PCD files

Usage:
  python scripts/data_preprocessing.py \
    --video input.mp4 \
    --rois_yaml data/frames/rois.yaml \
    --output_dir data \
    --max_frames 200 \
    --resize_h 288 --resize_w 360 \
    --pcap input.pcap
"""

import os
import cv2
import argparse
import yaml

VIEW_NAMES = ["front", "right", "rear", "left"]  # expected keys in the YAML

# --- Frame Extraction ---
def extract_frames(video_path, out_dir, max_frames=200, fps=None):
    frames_dir = os.path.join(out_dir, "frames", "all")
    os.makedirs(frames_dir, exist_ok=True)
    fps_arg = f"-r {fps}" if fps else ""
    cmd = f'ffmpeg -y -i "{video_path}" {fps_arg} -vframes {max_frames} -q:v 2 "{frames_dir}/%05d.png"'
    print("[ffmpeg]", cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError("ffmpeg failed to extract frames.")
    return frames_dir

# --- Load ROIs from YAML ---
def load_rois(yaml_path):
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    rois = {}
    for k in VIEW_NAMES:
        if k not in d:
            raise KeyError(f"ROI for view '{k}' not found in {yaml_path}")
        x1, y1, x2, y2 = map(int, d[k])
        rois[k] = (x1, y1, x2, y2)
    return rois

# --- Crop frames using ROIs ---
def crop_with_rois(frames_dir, out_dir, rois, resize_hw=None):
    for v in VIEW_NAMES:
        os.makedirs(os.path.join(out_dir, "frames", v), exist_ok=True)

    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    if not files:
        print("No frames found in", frames_dir)
        return

    # sanity check on first frame size and ROI validity
    first_img = cv2.imread(os.path.join(frames_dir, files[0]))
    if first_img is None:
        raise FileNotFoundError("Failed to read first extracted frame.")
    H, W = first_img.shape[:2]
    for v, (x1, y1, x2, y2) in rois.items():
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            raise ValueError(f"ROI for '{v}' is out of bounds of the frame size {W}x{H}: {x1,y1,x2,y2}")

    # crop all frames
    for fname in files:
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue
        for v in VIEW_NAMES:
            x1, y1, x2, y2 = rois[v]
            crop = img[y1:y2, x1:x2]
            if resize_hw is not None:
                h, w = resize_hw
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, "frames", v, fname), crop)

    # write a debug mosaic on the first frame
    vis = first_img.copy()
    colors = {
        "front": (0, 255, 0),
        "right": (0, 200, 255),
        "rear":  (255, 200, 0),
        "left":  (255, 0, 0),
    }
    for v, (x1, y1, x2, y2) in rois.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), colors[v], 2)
        cv2.putText(vis, v, (x1+6, y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[v], 2, cv2.LINE_AA)
    dbg = os.path.join(out_dir, "frames", "rois_debug.png")
    cv2.imwrite(dbg, vis)
    print("Wrote ROI debug overlay:", dbg)

# --- LiDAR PCAP to PCD Conversion (placeholder) ---
def convert_pcap_to_pcd(pcap_path, out_dir):
    lidar_dir = os.path.join(out_dir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    # Example command (replace with your actual tool):
    # os.system(f'velodyne_decoder -i "{pcap_path}" -o "{lidar_dir}"')
    print(f"[info] Convert PCAP to PCD with your preferred tool into: {lidar_dir}")
    print(f"[info] Input PCAP: {pcap_path}")

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input MP4 video path")
    parser.add_argument("--rois_yaml", required=True, help="YAML with ROIs for front/right/rear/left")
    parser.add_argument("--output_dir", default="data", help="Output data directory")
    parser.add_argument("--max_frames", type=int, default=200, help="Number of frames to extract")
    parser.add_argument("--fps", type=int, default=None, help="Optional re-encode FPS during extraction")
    parser.add_argument("--resize_h", type=int, default=None, help="Resize height for each crop")
    parser.add_argument("--resize_w", type=int, default=None, help="Resize width for each crop")
    parser.add_argument("--pcap", type=str, default=None, help="Optional PCAP for LiDAR conversion")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = extract_frames(args.video, args.output_dir, max_frames=args.max_frames, fps=args.fps)

    rois = load_rois(args.rois_yaml)
    resize_hw = None
    if args.resize_h is not None and args.resize_w is not None:
        resize_hw = (args.resize_h, args.resize_w)  # (H, W)

    crop_with_rois(frames_dir, args.output_dir, rois, resize_hw=resize_hw)

    if args.pcap:
        convert_pcap_to_pcd(args.pcap, args.output_dir)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
