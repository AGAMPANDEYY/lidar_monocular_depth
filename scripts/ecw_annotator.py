#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
from PIL import Image
import argparse

class ECWAnnotator:
    def __init__(self, window_name="ECW Region Annotator"):
        self.window_name = window_name
        self.points = []  # [top_left, top_right, bottom_right, bottom_left]
        self.current_point = None
        self.dragging_point = None
        self.point_radius = 5
        self.alpha = 0.3

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking near existing point
            for i, point in enumerate(self.points):
                if np.sqrt(((x - point[0])**2 + (y - point[1])**2)) < self.point_radius * 2:
                    self.dragging_point = i
                    return
            
            # Add new point if we have less than 4
            if len(self.points) < 4:
                self.points.append([x, y])
            self.draw_current()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_point is not None:
                self.points[self.dragging_point] = [x, y]
                self.draw_current()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_point = None

    def draw_current(self):
        img_copy = self.image.copy()
        
        # Draw trapezoid if we have all points
        if len(self.points) == 4:
            overlay = img_copy.copy()
            pts = np.array(self.points, np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, self.alpha, img_copy, 1 - self.alpha, 0, img_copy)
            
            # Draw connecting lines
            for i in range(4):
                cv2.line(img_copy, 
                        tuple(map(int, self.points[i])),
                        tuple(map(int, self.points[(i+1)%4])),
                        (0, 255, 0), 2)

        # Draw points
        for i, point in enumerate(self.points):
            cv2.circle(img_copy, tuple(map(int, point)), self.point_radius, (0, 0, 255), -1)
            cv2.putText(img_copy, f"P{i+1}", (point[0]+10, point[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow(self.window_name, img_copy)

    def annotate(self, image):
        self.image = image
        H, W = image.shape[:2]
        
        # Initialize with default trapezoid if no points exist
        if not self.points:
            top_y = int(H * 0.55)
            bot_y = int(H * 0.85)
            top_w = int(W * 0.20)
            bot_w = int(W * 0.90)
            top_left_x = W//2 - top_w//2
            top_right_x = W//2 + top_w//2
            bot_left_x = W//2 - bot_w//2
            bot_right_x = W//2 + bot_w//2
            
            self.points = [
                [top_left_x, top_y],     # Top left
                [top_right_x, top_y],    # Top right
                [bot_right_x, bot_y],    # Bottom right
                [bot_left_x, bot_y]      # Bottom left
            ]

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nECW Annotator Controls:")
        print("- Click and drag points to adjust shape")
        print("- 'r' to reset to default trapezoid")
        print("- 'c' to clear all points")
        print("- '+'/'-' to adjust overlay transparency")
        print("- 's' to save and exit")
        print("- 'q' to quit without saving")
        
        self.draw_current()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # Reset
                self.points = []
                self.annotate(image)
                
            elif key == ord('c'):  # Clear
                self.points = []
                self.draw_current()
                
            elif key == ord('+') or key == ord('='):  # Increase transparency
                self.alpha = min(1.0, self.alpha + 0.1)
                self.draw_current()
                
            elif key == ord('-') or key == ord('_'):  # Decrease transparency
                self.alpha = max(0.0, self.alpha - 0.1)
                self.draw_current()
                
            elif key == ord('s'):  # Save
                cv2.destroyAllWindows()
                return self.points
                
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return None

def save_ecw_region(points, output_path):
    """Save ECW region points"""
    data = {
        'points': points,
        'format': 'x,y points clockwise from top-left'
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved ECW region to {output_path}")

def load_ecw_region(filepath):
    """Load saved ECW region"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['points']
    return None

def main():
    parser = argparse.ArgumentParser(description='ECW Region Annotator')
    parser.add_argument('--frame', type=int, required=True,
                      help='Frame number to annotate')
    parser.add_argument('--video', type=str, default='data/input.avi',
                      help='Path to input video')
    args = parser.parse_args()
    
    # Extract frame
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, args.video)
    
    # Use extract_frame from generate_paper_figures
    from generate_paper_figures import extract_frame
    frame_rgb = extract_frame(video_path, args.frame, None)
    
    # Setup annotator
    annotator = ECWAnnotator()
    
    # Check for existing annotations
    annotations_dir = os.path.join(project_root, 'data', 'ecw_annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_file = os.path.join(annotations_dir, f'ecw_frame_{args.frame}.json')
    
    existing_points = load_ecw_region(annotation_file)
    if existing_points:
        print(f"Found existing ECW annotation for frame {args.frame}")
        annotator.points = existing_points
    
    # Run annotation
    points = annotator.annotate(frame_rgb)
    
    if points is not None:
        save_ecw_region(points, annotation_file)
        print("ECW region saved successfully!")
    else:
        print("Annotation cancelled.")

if __name__ == '__main__':
    main()