#!/usr/bin/env python3
"""
Extract frames from video with frame range support and ROI cropping.
"""
import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path

VIEW_NAMES_DEFAULT = ["front", "right", "rear", "left"]  # TL, TR, BL, BR

def extract_frames(video_path, output_dir, start_frame=None, end_frame=None, max_frames=None):
    """Extract frames from video with frame range support."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate frame range
    start = start_frame if start_frame is not None else 0
    end = min(end_frame if end_frame is not None else total_frames, total_frames)
    if max_frames:
        end = min(start + max_frames, end)
    
    print(f"\nVideo info:")
    print(f"- Total frames: {total_frames}")
    print(f"- FPS: {fps}")
    print(f"- Extracting frames {start} to {end}")
    
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    frame_count = 0
    current_frame = start
    
    while cap.isOpened() and current_frame < end:
        ret, frame = cap.read()
        if not ret:
            break
            
        out_path = os.path.join(output_dir, f"{current_frame:05d}.png")
        cv2.imwrite(out_path, frame)
        
        frame_count += 1
        current_frame += 1
        
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count} frames...")
    
    cap.release()
    print(f"\nExtracted {frame_count} frames to {output_dir}")
    print(f"Frame range: {start:05d}.png to {(current_frame-1):05d}.png")
    return True

def manual_select_rois(img_bgr, names=VIEW_NAMES_DEFAULT):
    """Manually select ROIs for different camera views."""
    print("\nSelect 4 ROIs for camera views:")
    print("1. Click and drag to select each ROI")
    print("2. Press ENTER to confirm selection")
    print("3. Press 'c' to cancel and retry if you made a mistake\n")
    
    rois = []
    for i in range(4):
        while True:
            print(f"Select ROI for {names[i]} view...")
            r = cv2.selectROI("Select ROI", img_bgr, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI")
            x,y,w,h = r
            
            # Check if ROI is valid (not zero size)
            if w > 0 and h > 0:
                rois.append((x, y, x+w, y+h))
                print(f"{names[i]} ROI: ({x}, {y}, {x+w}, {y+h})")
                break
            else:
                print(f"Invalid ROI selection for {names[i]}. Please try again.")
                print("Make sure to click and drag to create a non-zero size selection.")
    return rois

def save_rois(rois, out_yaml, names=VIEW_NAMES_DEFAULT):
    """Save ROIs to YAML file."""
    data = { names[i]: [int(v) for v in rois[i]] for i in range(4) }
    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f)
    print(f"Saved ROIs to {out_yaml}")

def load_rois(yaml_path, names=VIEW_NAMES_DEFAULT):
    """Load ROIs from YAML file."""
    with open(yaml_path, "r") as f:
        d = yaml.safe_load(f)
    return [tuple(d[n]) for n in names]

def draw_rois_debug(img, rois, names=VIEW_NAMES_DEFAULT, out_path=None):
    """Draw ROIs on image for visualization."""
    vis = img.copy()
    colors = [(0,255,0),(0,200,255),(255,200,0),(255,0,0)]
    for i, (x1,y1,x2,y2) in enumerate(rois):
        cv2.rectangle(vis, (x1,y1), (x2,y2), colors[i%4], 2)
        cv2.putText(vis, names[i], (x1+6,y1+24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i%4], 2, cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, vis)
        print("Wrote", out_path)
    return vis

def crop_with_rois(frames_dir, out_dir, rois, names=VIEW_NAMES_DEFAULT, limit=None, resize_to=None):
    """Crop frames using the defined ROIs."""
    # Validate ROIs
    for i, (x1,y1,x2,y2) in enumerate(rois):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid ROI for {names[i]}: ({x1},{y1},{x2},{y2}). Ensure proper selection.")
    
    # Create output directories
    for name in names:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    
    # Get and sort input files
    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if limit: files = files[:limit]
    
    print(f"\nCropping {len(files)} frames into views...")
    for fname in files:
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            print(f"Warning: Could not read frame {fname}")
            continue
            
        for i, (x1,y1,x2,y2) in enumerate(rois):
            crop = img[y1:y2, x1:x2]
            if resize_to:
                crop = cv2.resize(crop, resize_to, interpolation=cv2.INTER_AREA)
            out_path = os.path.join(out_dir, names[i], fname)
            if not cv2.imwrite(out_path, crop):
                print(f"Warning: Failed to save crop to {out_path}")
                
        if len(files) > 100 and len(files) % 100 == 0:
            print(f"Processed {len(files)} frames...")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video and crop views")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output_dir", default="data/frames",
                      help="Output directory for frames")
    parser.add_argument("--start_frame", type=int, help="Start frame number")
    parser.add_argument("--end_frame", type=int, help="End frame number")
    parser.add_argument("--max_frames", type=int, help="Maximum number of frames")
    parser.add_argument("--rois_yaml", help="Path to ROIs YAML file (if exists)")
    parser.add_argument("--resize_w", type=int, help="Resize crops to this width")
    parser.add_argument("--resize_h", type=int, help="Resize crops to this height")
    
    args = parser.parse_args()
    
    # Extract frames first
    frames_dir = os.path.join(args.output_dir, "raw")  # Store raw frames in 'raw' subdirectory
    success = extract_frames(args.video, frames_dir,
                           args.start_frame, args.end_frame,
                           args.max_frames)
    if not success:
        return 1

    # Get first frame for ROI selection
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not frame_files:
        print("No frames were extracted!")
        return 1
    
    img0 = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    
    # Load or select ROIs
    rois = None
    if args.rois_yaml and os.path.exists(args.rois_yaml):
        print(f"Loading ROIs from {args.rois_yaml}")
        rois = load_rois(args.rois_yaml)
    else:
        rois = manual_select_rois(img0)
        if args.rois_yaml:
            save_rois(rois, args.rois_yaml)
    
    # Draw debug visualization
    roi_debug_path = os.path.join(args.output_dir, "rois_debug.png")
    draw_rois_debug(img0, rois, out_path=roi_debug_path)
    
    # Crop frames with ROIs
    resize_to = None
    if args.resize_w and args.resize_h:
        resize_to = (args.resize_w, args.resize_h)
    
    crop_with_rois(frames_dir, args.output_dir, rois, resize_to=resize_to)
    print("Frame extraction and cropping complete.")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
